"""Il cervello: decide quando guardare, cosa proporre, come e dove parlare con Dario."""
from __future__ import annotations

import asyncio
import io
import logging
import time
from datetime import datetime
from typing import Awaitable, Callable

from PIL import Image

from . import prompts
from .config import Config, Room
from .llm import LLM, Tool
from .memory import Memory, Task

log = logging.getLogger("hup.brain")

ESITI = ("fatto", "dopo", "non_serve", "altro")
# Differenza media (0-255) sotto la quale la stanza si considera invariata dall'ultimo controllo.
SAME_SCENE_THRESHOLD = 6.0

TextSender = Callable[[str], Awaitable[None]]
PhotoSender = Callable[[bytes, str], Awaitable[None]]


def thumb(img: bytes) -> list[int]:
    """Impronta 16x16 in scala di grigi, per capire se la stanza è cambiata senza chiamare l'AI."""
    with Image.open(io.BytesIO(img)) as im:
        return list(im.convert("L").resize((16, 16)).tobytes())


def shrink(img: bytes, max_side: int = 1280) -> bytes:
    """Riduce la foto prima di mandarla all'AI: molto più veloce e costa meno, i dettagli bastano."""
    with Image.open(io.BytesIO(img)) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=80)
        return out.getvalue()


def thumb_diff(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 255.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


class Brain:
    def __init__(self, cfg: Config, ha, llm: LLM, mem: Memory, clock=time.time):
        self.cfg, self.ha, self.llm, self.mem, self.clock = cfg, ha, llm, mem, clock
        self.telegram_text: TextSender | None = None
        self.telegram_photo: PhotoSender | None = None
        self._lock: dict[str, asyncio.Lock] = {r.name: asyncio.Lock() for r in cfg.rooms}
        self._chat_lock = asyncio.Lock()
        self._motion_due: dict[str, float] = {}
        self._voice_wait: dict[str, float] = {}  # task_id -> scadenza attesa risposta vocale
        self._last_app_task: str | None = None
        self._seen: dict[str, float] = {}  # stanza -> quando la telecamera ha visto Dario
        for room in cfg.rooms:
            self.mem.goal(room.name, room.goal)

    # ------------------------------------------------------------------ utilità
    def now_dt(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), self.cfg.tz)

    def now_text(self) -> str:
        giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
        d = self.now_dt()
        return f"{giorni[d.weekday()]} {d:%d/%m %H:%M}"

    def in_quiet_hours(self) -> bool:
        t = self.now_dt().time()
        start, end = self.cfg.quiet_start, self.cfg.quiet_end
        if start == end:
            return False
        if start < end:
            return start <= t < end
        return t >= start or t < end

    async def availability(self) -> tuple[bool, str]:
        if await self.ha.state(self.cfg.person_entity) != "home":
            return False, "fuori casa"
        if await self.privacy_on():
            return False, "ha messo le telecamere in privacy"
        if self.cfg.sleep_entity and await self.ha.state(self.cfg.sleep_entity) == "on":
            return False, "sta dormendo"
        if self.in_quiet_hours():
            return False, "fascia di silenzio"
        return True, "in casa e sveglio"

    async def privacy_on(self) -> bool:
        return bool(self.cfg.privacy_entity) and await self.ha.state(self.cfg.privacy_entity) == "on"

    async def camera_ok(self, room: Room) -> bool:
        return await self.ha.state(room.camera) not in (None, "unavailable", "unknown")

    def goal(self, room: Room) -> str:
        return self.mem.goal(room.name, room.goal)

    def _last_check(self, room: str) -> float:
        return float(self.mem.get(f"last_check:{room}", 0))

    # ------------------------------------------------------------------ ciclo principale
    async def tick(self) -> None:
        now = self.clock()
        await self._voice_timeouts(now)
        available, reason = await self.availability()
        if not available:
            if self.mem.get("last_unavailable_reason") != reason:
                self.mem.set("last_unavailable_reason", reason)
                self.mem.log("pausa", f"In pausa: Dario {reason}")
            return
        if self.mem.get("last_unavailable_reason"):
            self.mem.set("last_unavailable_reason", None)
            self.mem.log("pausa", "Di nuovo attivo")
        for room in self.cfg.rooms:
            try:
                await self._tick_room(room, now)
            except Exception as exc:  # noqa: BLE001  una stanza in errore non ferma le altre
                log.warning("Controllo %s non riuscito: %r", room.name, exc)
                self.mem.log("errore", f"Controllo non riuscito: {exc!r}"[:300], room.name)
                self.mem.set(f"last_check:{room.name}", now)
                for t in self.mem.open_tasks(room.name):  # riprova tra 5 minuti, non a raffica
                    if t.next_action_at <= now:
                        self.mem.update_task(t.id, next_action_at=now + 300)

    async def _tick_room(self, room: Room, now: float) -> None:
        if not await self.camera_ok(room):
            return
        due_tasks = [t for t in self.mem.open_tasks(room.name)
                     if t.next_action_at <= now and t.id not in self._voice_wait]
        since_last = now - self._last_check(room.name)
        motion_at = self._motion_due.get(room.name)
        if due_tasks:
            await self.check_room(room, "verifica")
        elif since_last < self.cfg.min_minutes_between_checks * 60:
            return
        elif motion_at and motion_at <= now:
            self._motion_due.pop(room.name, None)
            await self.check_room(room, "movimento")
        elif since_last >= self.cfg.check_interval_minutes * 60:
            await self.check_room(room, "periodico")

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("Errore nel ciclo principale")
            await asyncio.sleep(30)

    # ------------------------------------------------------------------ controllo stanza
    async def check_room(self, room: Room, reason: str, force: bool = False) -> dict:
        async with self._lock[room.name]:
            return await self._check_room(room, reason, force)

    async def _check_room(self, room: Room, reason: str, force: bool) -> dict:
        now = self.clock()
        open_tasks = self.mem.open_tasks(room.name)
        allow_new = force or len(self.mem.open_tasks()) < self.cfg.max_open_tasks
        if not open_tasks and not allow_new:
            # Ci sono già abbastanza compiti aperti in casa: non serve nemmeno guardare.
            self.mem.set(f"last_check:{room.name}", now)
            return {"esito": "in_attesa", "dettaglio": "troppi compiti già aperti in casa"}
        try:
            img = await self.ha.camera_snapshot(room.camera)
        except Exception as exc:  # noqa: BLE001
            self.mem.log("errore", f"Snapshot non riuscito ({exc})", room.name)
            self.mem.set(f"last_check:{room.name}", now)
            return {"esito": "errore", "dettaglio": "snapshot non riuscito"}
        fp = thumb(img)
        diff = thumb_diff(self.mem.get(f"thumb:{room.name}"), fp)
        self.mem.set(f"last_check:{room.name}", now)
        any_due = any(t.next_action_at <= now for t in open_tasks)
        if not force and not any_due and reason in ("periodico", "movimento") and diff < SAME_SCENE_THRESHOLD:
            log.info("%s invariata (diff %.1f), salto l'analisi", room.name, diff)
            return {"esito": "invariata"}
        self.mem.set(f"thumb:{room.name}", fp)

        t0 = time.monotonic()
        result = await self.llm.vision_json(
            prompts.check_system(self.cfg.assistant_name, self.cfg.personality),
            prompts.check_prompt(
                room.name, self.goal(room), self.mem.rules(room.name), [t.as_dict() for t in open_tasks],
                [t.as_dict() for t in self.mem.recent_closed(room.name)], self.now_text()),
            [shrink(img)],
            prompts.CHECK_SCHEMA,
        )
        log.info("Controllo %s (%s, AI %.1fs): %s", room.name, reason, time.monotonic() - t0,
                 result.get("descrizione"))
        if result.get("persona_presente"):
            self._seen[room.name] = now
        if not result.get("immagine_utile", True):
            self.mem.log("controllo", "Immagine non utilizzabile", room.name)
            for t in open_tasks:  # riprovo più tardi senza contare un promemoria
                self.mem.update_task(t.id, next_action_at=now + self.cfg.reminder_minutes * 60)
            return {"esito": "immagine_non_utile"}

        verdicts = {v.get("task_id", "").strip().lower(): v for v in result.get("verifiche", [])}
        for task in open_tasks:
            v = verdicts.get(task.id)
            if v is None:
                continue
            if v.get("completato"):
                self.mem.update_task(task.id, status="fatto", next_action_at=now)
                self.mem.log("completato", task.text, room.name)
                await self._say(task, v.get("messaggio") or "Perfetto, grazie!", interactive=False)
            elif task.next_action_at <= now:
                await self._remind(task, v.get("messaggio") or f"Ti ricordo: {task.text}")

        new = result.get("nuovo_compito") or {}
        if not open_tasks and allow_new and new.get("presente") and new.get("compito"):
            task = self.mem.add_task(room.name, new["compito"], now + self.cfg.reminder_minutes * 60)
            self.mem.log("nuovo", task.text, room.name)
            await self._say(task, new.get("messaggio") or task.text, interactive=True)
            return {"esito": "nuovo_compito", "compito": task.text, "task_id": task.id,
                    "descrizione": result.get("descrizione")}
        if not open_tasks:
            self.mem.log("controllo", f"Tutto ok: {result.get('descrizione', '')}", room.name)
        return {"esito": "ok", "descrizione": result.get("descrizione"),
                "compiti_aperti": [t.as_dict() for t in self.mem.open_tasks(room.name)]}

    async def _remind(self, task: Task, message: str) -> None:
        now = self.clock()
        if task.reminders >= self.cfg.max_reminders:
            self.mem.update_task(task.id, status="lasciato", next_action_at=now)
            self.mem.log("lasciato", f"Lascio perdere: {task.text}", task.room)
            return
        task = self.mem.update_task(task.id, reminders=task.reminders + 1,
                                    next_action_at=now + self.cfg.reminder_minutes * 60)
        self.mem.log("promemoria", message, task.room)
        await self._say(task, message, interactive=True)

    # ------------------------------------------------------------------ consegna messaggi
    async def _recent(self, entity: str | None) -> float | None:
        """Quando il sensore ha visto movimento l'ultima volta (adesso se è attivo), entro la finestra utile."""
        if not entity:
            return None
        st = await self.ha.get_state(entity)
        if not st:
            return None
        now = self.clock()
        if st["state"] == "on":
            return now
        if st["state"] == "off":
            changed = datetime.fromisoformat(st["last_changed"]).timestamp()
            if now - changed <= self.cfg.voice_presence_minutes * 60:
                return changed
        return None

    async def _usable(self, entity: str | None) -> bool:
        if not entity:
            return False
        if entity.startswith("notify."):
            return True  # i servizi notify non hanno stato: in caso di errore ripiego sull'app
        return await self.ha.state(entity) not in (None, "unavailable", "unknown")

    async def locate(self) -> tuple[str, str] | None:
        """Dove parlare a Dario: ("satellite", entità) o ("speaker", entità), in base a dove si trova."""
        candidates: list[tuple[float, str, str]] = []
        for sat in self.cfg.satellites:
            for sensor in sat.motion:
                ts = await self._recent(sensor)
                if ts is not None:
                    candidates.append((ts, "satellite", sat.entity))
        horizon = self.cfg.voice_presence_minutes * 60
        for room in self.cfg.rooms:
            times = [await self._recent(room.motion)]
            seen = self._seen.get(room.name)
            if seen and self.clock() - seen <= horizon:
                times.append(seen)
            ts = max((t for t in times if t is not None), default=None)
            if ts is None:
                continue
            if room.satellite:
                candidates.append((ts, "satellite", room.satellite))
            elif room.speaker:
                candidates.append((ts, "speaker", room.speaker))
        for _ts, kind, entity in sorted(candidates, reverse=True):
            if await self._usable(entity):
                return kind, entity
        return None

    async def satellite_near(self) -> str | None:
        where = await self.locate()
        return where[1] if where and where[0] == "satellite" else None

    async def _say(self, task: Task, message: str, interactive: bool) -> None:
        where = await self.locate()
        if where and where[0] == "satellite":
            if await self._voice(where[1], task, message, interactive):
                return
        if where and where[0] == "speaker":
            spoke = await self._speak(where[1], message)
            if spoke:
                self.mem.log("voce", message, task.room)
                if not interactive:
                    return
                # L'altoparlante non ascolta: la risposta arriva dai pulsanti della notifica.
        await self._app(task, message, interactive)

    async def _speak(self, speaker: str, message: str) -> bool:
        try:
            if speaker.startswith("notify."):
                await self.ha.call_service("notify", speaker.split(".", 1)[1],
                                           {"message": message, "data": {"type": "announce"}})
            else:
                await self.ha.call_service("tts", "speak", {"entity_id": self.cfg.tts_entity,
                                                            "media_player_entity_id": speaker,
                                                            "message": message})
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Altoparlante %s non disponibile: %s", speaker, exc)
            return False

    async def _voice(self, sat: str, task: Task, message: str, interactive: bool) -> bool:
        try:
            if interactive:
                data = {
                    "entity_id": sat,
                    "start_message": message,
                    "extra_system_prompt": prompts.voice_context(
                        self.cfg.assistant_name, self.cfg.personality, task.id, task.room, task.text, message,
                        self.cfg.feedback_script.split(".", 1)[-1]),
                    "preannounce": True,
                }
                self._voice_wait[task.id] = self.clock() + self.cfg.voice_reply_timeout_minutes * 60
                asyncio.create_task(self._fire("assist_satellite", "start_conversation", data))
            else:
                asyncio.create_task(self._fire("assist_satellite", "announce",
                                               {"entity_id": sat, "message": message, "preannounce": True}))
            self.mem.update_task(task.id, channel="voce")
            self.mem.log("voce", message, task.room)
            return True
        except Exception as exc:  # noqa: BLE001
            self._voice_wait.pop(task.id, None)
            log.warning("Voce non disponibile (%s)", exc)
            return False

    async def _fire(self, domain: str, service: str, data: dict) -> None:
        try:
            await self.ha.call_service(domain, service, data)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s.%s non riuscito: %s", domain, service, exc)

    async def _app(self, task: Task, message: str, interactive: bool) -> None:
        sent = False
        if self.cfg.notify_service:
            domain, service = self.cfg.notify_service.split(".", 1)
            room = self.cfg.room(task.room)
            data: dict = {"tag": f"hup_{task.id}", "group": "hup"}
            if room:
                data["image"] = f"/api/camera_proxy/{room.camera}"
            if interactive:
                data["actions"] = [
                    {"action": f"HUP_FATTO_{task.id}", "title": "Fatto"},
                    {"action": f"HUP_DOPO_{task.id}", "title": "Dopo"},
                    {"action": "REPLY", "title": "Rispondi"},
                ]
                self._last_app_task = task.id
            try:
                await self.ha.call_service(domain, service, {
                    "title": f"{self.cfg.assistant_name} · {task.room}", "message": message, "data": data})
                sent = True
            except Exception as exc:  # noqa: BLE001
                log.warning("Notifica app non riuscita: %s", exc)
        if not sent and self.telegram_text:
            await self.telegram_text(message)
            sent = True
        if sent:
            self.mem.update_task(task.id, channel="app" if self.cfg.notify_service else "telegram")
            self.mem.log("messaggio", message, task.room)

    async def _voice_timeouts(self, now: float) -> None:
        for tid, deadline in list(self._voice_wait.items()):
            if deadline > now:
                continue
            self._voice_wait.pop(tid, None)
            task = self.mem.task(tid)
            if task and task.status in ("aperto",):
                await self._app(task, f"Te l'avevo detto a voce ma forse non mi hai sentito: {task.text}",
                                interactive=True)

    # ------------------------------------------------------------------ risposte di Dario
    async def feedback(self, task_id: str, esito: str, nota: str = "", canale: str = "") -> dict:
        task = self.mem.task(task_id or "")
        if not task:
            return {"ok": False, "errore": f"compito {task_id} non trovato"}
        esito = (esito or "altro").strip().lower().replace(" ", "_")
        if esito not in ESITI:
            esito = "altro"
        self._voice_wait.pop(task.id, None)
        now = self.clock()
        nota = (nota or "").strip()
        if esito == "fatto":
            self.mem.update_task(task.id, status="da_verificare", next_action_at=now + 90, note=nota or None)
        elif esito == "dopo":
            self.mem.update_task(task.id, next_action_at=now + self.cfg.postpone_minutes * 60,
                                 note=nota or task.note)
        elif esito == "non_serve":
            self.mem.update_task(task.id, status="non_serve", next_action_at=now, note=nota or None)
            reason = f" ({nota})" if nota else ""
            self.mem.add_rule(task.room, f"Non serve: «{task.text}»{reason}")
        else:
            self.mem.update_task(task.id, next_action_at=now + self.cfg.reminder_minutes * 60,
                                 note=nota or task.note)
        self.mem.log("risposta", f"{esito}{': ' + nota if nota else ''} [{canale or '?'}]", task.room)
        return {"ok": True, "task_id": task.id, "esito": esito}

    async def on_feedback_event(self, event: dict) -> None:
        d = event.get("data", {})
        await self.feedback(d.get("task_id", ""), d.get("esito", ""), d.get("nota", ""), d.get("canale", "voce"))

    async def on_notification_action(self, event: dict) -> None:
        d = event.get("data", {})
        action = d.get("action", "")
        if action.startswith("HUP_FATTO_"):
            await self.feedback(action.removeprefix("HUP_FATTO_"), "fatto", canale="app")
        elif action.startswith("HUP_DOPO_"):
            await self.feedback(action.removeprefix("HUP_DOPO_"), "dopo", canale="app")
        elif action == "REPLY" and d.get("reply_text"):
            ctx = f"[Risposta alla notifica del compito {self._last_app_task}] " if self._last_app_task else ""
            answer = await self.chat(ctx + d["reply_text"], source="app")
            if self.cfg.notify_service and answer:
                domain, service = self.cfg.notify_service.split(".", 1)
                await self._fire(domain, service,
                                 {"title": self.cfg.assistant_name, "message": answer,
                                  "data": {"tag": "hup_chat", "group": "hup"}})

    async def on_state_changed(self, event: dict) -> None:
        d = event.get("data", {})
        entity = d.get("entity_id")
        new = (d.get("new_state") or {}).get("state")
        for room in self.cfg.rooms:
            if room.motion and entity == room.motion and new == "off":
                self._motion_due[room.name] = self.clock() + self.cfg.motion_cleared_minutes * 60

    # ------------------------------------------------------------------ chat (Telegram e risposte app)
    TOOLS = [
        Tool("guarda_stanza", "Guarda adesso una stanza dalla telecamera e rispondi a una domanda su cosa si vede.",
             {"type": "object", "properties": {
                 "stanza": {"type": "string"},
                 "domanda": {"type": "string", "description": "Cosa vuoi sapere dell'immagine"},
                 "invia_foto": {"type": "boolean", "description": "true per mandare la foto a Dario su Telegram"}},
              "required": ["stanza", "domanda"]}),
        Tool("controlla_stanza", "Fa subito il controllo ordine di una stanza, come quelli automatici.",
             {"type": "object", "properties": {"stanza": {"type": "string"}}, "required": ["stanza"]}),
        Tool("registra_esito", "Registra la risposta di Dario a un compito proposto.",
             {"type": "object", "properties": {
                 "task_id": {"type": "string"},
                 "esito": {"type": "string", "enum": list(ESITI)},
                 "nota": {"type": "string"}},
              "required": ["task_id", "esito"]}),
        Tool("aggiungi_regola", "Salva un'eccezione o preferenza stabile di Dario per una stanza "
                                "(es. 'lo stendino può stare vicino alla finestra').",
             {"type": "object", "properties": {"stanza": {"type": "string"}, "regola": {"type": "string"}},
              "required": ["stanza", "regola"]}),
        Tool("rimuovi_regola", "Elimina una regola salvata, dato il suo numero.",
             {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}),
        Tool("imposta_obiettivo", "Cambia l'obiettivo di Dario per una stanza.",
             {"type": "object", "properties": {"stanza": {"type": "string"}, "obiettivo": {"type": "string"}},
              "required": ["stanza", "obiettivo"]}),
        Tool("imposta_privacy", "Mette le telecamere in privacy (le spegne, es. arrivano ospiti) o le riattiva.",
             {"type": "object", "properties": {"attiva": {"type": "boolean",
                                                          "description": "true = privacy (telecamere spente)"}},
              "required": ["attiva"]}),
    ]

    async def _tool(self, name: str, args: dict, source: str):
        room = self.cfg.room(str(args.get("stanza", ""))) if "stanza" in args else None
        if "stanza" in args and not room:
            return {"errore": f"Stanza sconosciuta. Stanze disponibili: {[r.name for r in self.cfg.rooms]}"}
        if name == "imposta_privacy":
            if not self.cfg.privacy_entity:
                return {"errore": "privacy_entity non configurato"}
            service = "turn_on" if args.get("attiva") else "turn_off"
            await self.ha.call_service("input_boolean", service, {"entity_id": self.cfg.privacy_entity})
            self.mem.log("privacy", "Telecamere in privacy" if args.get("attiva") else "Telecamere riattivate")
            return {"ok": True, "privacy": bool(args.get("attiva"))}
        if name in ("guarda_stanza", "controlla_stanza") and await self.privacy_on():
            return {"errore": "Le telecamere sono in privacy: non posso guardare finché Dario non la toglie."}
        if name in ("guarda_stanza", "controlla_stanza") and not await self.camera_ok(room):
            return {"errore": f"La telecamera di {room.name} non è raggiungibile in questo momento."}
        if name == "guarda_stanza":
            img = await self.ha.camera_snapshot(room.camera)
            res = await self.llm.vision_json(
                prompts.check_system(self.cfg.assistant_name, self.cfg.personality),
                f"Stanza {room.name}. Domanda di Dario: {args.get('domanda', 'Cosa vedi?')}",
                [shrink(img)], prompts.LOOK_SCHEMA)
            if args.get("invia_foto") and source == "telegram" and self.telegram_photo:
                await self.telegram_photo(img, room.name)
            return res
        if name == "controlla_stanza":
            return await self.check_room(room, "richiesta", force=True)
        if name == "registra_esito":
            return await self.feedback(str(args.get("task_id", "")), str(args.get("esito", "")),
                                       str(args.get("nota", "")), canale=source)
        if name == "aggiungi_regola":
            return {"ok": True, "id": self.mem.add_rule(room.name, str(args["regola"]))}
        if name == "rimuovi_regola":
            return {"ok": self.mem.remove_rule(int(args["id"]))}
        if name == "imposta_obiettivo":
            self.mem.set_goal(room.name, str(args["obiettivo"]))
            return {"ok": True}
        return {"errore": f"strumento {name} sconosciuto"}

    async def chat(self, text: str, source: str = "telegram") -> str:
        async with self._chat_lock:
            _, presence = await self.availability()
            events = [f"{datetime.fromtimestamp(e['ts'], self.cfg.tz):%d/%m %H:%M} {e['room'] or ''} "
                      f"{e['kind']}: {e['text']}" for e in self.mem.events(12)]
            system = prompts.chat_system(
                self.cfg.assistant_name, self.cfg.personality, self.now_text(), presence,
                [{"name": r.name, "goal": self.goal(r)} for r in self.cfg.rooms],
                [t.as_dict() for t in self.mem.open_tasks()], self.mem.rules(), events)
            self.mem.add_chat("user", text)
            history = self.mem.chat_history()

            async def handler(name: str, args: dict):
                try:
                    return await self._tool(name, args, source)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Strumento %s fallito", name)
                    return {"errore": str(exc)}

            answer = await self.llm.chat(system, history, self.TOOLS, handler)
            answer = answer or "Ok."
            self.mem.add_chat("assistant", answer)
            return answer
