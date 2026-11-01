"""Test della logica con Home Assistant e AI simulati."""
import asyncio
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hup_locale"))

from app import config  # noqa: E402
from app.brain import Brain  # noqa: E402
from app.llm import _to_gemini_schema  # noqa: E402
from app.memory import Memory  # noqa: E402
from app import prompts  # noqa: E402

OPTIONS = {
    "llm_provider": "gemini", "gemini_api_key": "x",
    "person_entity": "person.dario", "sleep_entity": "input_boolean.hup_dormo",
    "quiet_hours_start": "23:30", "quiet_hours_end": "07:00", "timezone": "Europe/Rome",
    "rooms": [{"name": "Cucina", "camera": "camera.cucina", "motion": "binary_sensor.motion_cucina",
               "goal": "Cucina in ordine"}],
    "satellites": [{"entity": "assist_satellite.letto", "motion": ["binary_sensor.motion_letto"]}],
    "feedback_script": "script.hup_risposta", "notify_service": "notify.mobile_app_tel",
    "check_interval_minutes": 30, "min_minutes_between_checks": 10, "motion_cleared_minutes": 3,
    "reminder_minutes": 30, "postpone_minutes": 60, "max_reminders": 2,
    "voice_presence_minutes": 10, "voice_reply_timeout_minutes": 3,
}


def jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, "JPEG")
    return buf.getvalue()


class Clock:
    def __init__(self):  # domenica 1/11/2026 ore 11:00 a Roma = 10:00 UTC
        self.t = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc).timestamp()

    def __call__(self):
        return self.t

    def advance(self, minutes):
        self.t += minutes * 60


class FakeHA:
    def __init__(self, clock):
        self.clock = clock
        self.states = {"person.dario": "home", "input_boolean.hup_dormo": "off",
                       "assist_satellite.letto": "idle", "binary_sensor.motion_letto": "off",
                       "binary_sensor.motion_cucina": "off"}
        self.changed = {"binary_sensor.motion_letto": 0}
        self.calls = []
        self.image = jpeg((200, 200, 200))

    async def state(self, e):
        return self.states.get(e)

    async def get_state(self, e):
        if e not in self.states:
            return None
        ts = datetime.fromtimestamp(self.changed.get(e, 0), timezone.utc).isoformat()
        return {"state": self.states[e], "last_changed": ts}

    async def camera_snapshot(self, e):
        return self.image

    async def call_service(self, domain, service, data=None, return_response=False):
        self.calls.append((domain, service, data))
        return []

    def names(self):
        return [f"{d}.{s}" for d, s, _ in self.calls]


class FakeLLM:
    def __init__(self):
        self.vision_results = []
        self.prompts = []
        self.chat_script = []

    async def vision_json(self, system, prompt, images, schema):
        self.prompts.append(prompt)
        return self.vision_results.pop(0)

    async def chat(self, system, history, tools, handler):
        out = None
        for name, args in self.chat_script:
            out = await handler(name, args)
        return f"risposta ({out})"


def nuovo(compito="Metti la pentola nel lavello", messaggio="Ehi, metti la pentola nel lavello?"):
    return {"descrizione": "cucina", "immagine_utile": True, "verifiche": [],
            "nuovo_compito": {"presente": True, "compito": compito, "messaggio": messaggio}}


def tutto_ok(verifiche=None):
    return {"descrizione": "cucina", "immagine_utile": True, "verifiche": verifiche or [],
            "nuovo_compito": {"presente": False, "compito": "", "messaggio": ""}}


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    cfg = config.from_dict(OPTIONS, data_dir=str(tmp_path))
    ha, llm = FakeHA(clock), FakeLLM()
    mem = Memory(str(tmp_path / "hup.db"), clock=clock)
    return Brain(cfg, ha, llm, mem, clock=clock), ha, llm, mem, clock


async def settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_non_disturba_fuori_casa_dormendo_e_di_notte(env):
    brain, ha, llm, mem, clock = env
    ha.states["person.dario"] = "not_home"
    await brain.tick()
    ha.states["person.dario"] = "home"
    ha.states["input_boolean.hup_dormo"] = "on"
    await brain.tick()
    ha.states["input_boolean.hup_dormo"] = "off"
    clock.t = datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc).timestamp()  # 00:00 a Roma
    await brain.tick()
    assert llm.prompts == [] and ha.calls == []


@pytest.mark.asyncio
async def test_nuovo_compito_via_notifica_app_quando_nessun_satellite_vicino(env):
    brain, ha, llm, mem, clock = env
    llm.vision_results = [nuovo()]
    await brain.tick()
    await settle()
    assert ha.names() == ["notify.mobile_app_tel"]
    data = ha.calls[0][2]
    task = mem.open_tasks()[0]
    assert data["message"] == "Ehi, metti la pentola nel lavello?"
    assert [a["action"] for a in data["data"]["actions"]] == [f"HUP_FATTO_{task.id}", f"HUP_DOPO_{task.id}", "REPLY"]


@pytest.mark.asyncio
async def test_dialogo_vocale_quando_dario_e_vicino_al_satellite(env):
    brain, ha, llm, mem, clock = env
    ha.states["binary_sensor.motion_letto"] = "on"
    llm.vision_results = [nuovo()]
    await brain.tick()
    await settle()
    assert ha.names() == ["assist_satellite.start_conversation"]
    task = mem.open_tasks()[0]
    ctx = ha.calls[0][2]["extra_system_prompt"]
    assert task.id in ctx and "hup_risposta" in ctx


@pytest.mark.asyncio
async def test_risposta_vocale_fatto_poi_verifica_e_complimenti(env):
    brain, ha, llm, mem, clock = env
    ha.states["binary_sensor.motion_letto"] = "on"
    llm.vision_results = [nuovo()]
    await brain.tick()
    task = mem.open_tasks()[0]
    await brain.on_feedback_event({"data": {"task_id": task.id, "esito": "fatto", "nota": "fatto ora",
                                            "canale": "voce"}})
    assert mem.task(task.id).status == "da_verificare"
    clock.advance(2)
    ha.image = jpeg((20, 20, 20))
    llm.vision_results = [tutto_ok([{"task_id": task.id, "completato": True, "messaggio": "Grande!"}])]
    await brain.tick()
    await settle()
    assert mem.task(task.id).status == "fatto"
    assert ha.calls[-1][1] == "announce" and ha.calls[-1][2]["message"] == "Grande!"


@pytest.mark.asyncio
async def test_promemoria_fino_al_massimo_poi_lascia_perdere(env):
    brain, ha, llm, mem, clock = env
    llm.vision_results = [nuovo()]
    await brain.tick()
    task = mem.open_tasks()[0]
    for i in range(2):
        clock.advance(31)
        llm.vision_results = [tutto_ok([{"task_id": task.id, "completato": False, "messaggio": f"promemoria {i}"}])]
        await brain.tick()
    assert mem.task(task.id).reminders == 2
    clock.advance(31)
    llm.vision_results = [tutto_ok([{"task_id": task.id, "completato": False, "messaggio": "ancora"}])]
    await brain.tick()
    assert mem.task(task.id).status == "lasciato"
    assert [c[2]["message"] for c in ha.calls][-2:] == ["promemoria 0", "promemoria 1"]


@pytest.mark.asyncio
async def test_non_serve_diventa_regola_e_entra_nel_prompt(env):
    brain, ha, llm, mem, clock = env
    llm.vision_results = [nuovo("Sposta lo stendino")]
    await brain.tick()
    task = mem.open_tasks()[0]
    await brain.feedback(task.id, "non_serve", "lo stendino sta lì apposta", "voce")
    assert mem.task(task.id).status == "non_serve"
    clock.advance(31)
    ha.image = jpeg((10, 90, 10))
    llm.vision_results = [tutto_ok()]
    await brain.tick()
    assert "lo stendino sta lì apposta" in llm.prompts[-1]


@pytest.mark.asyncio
async def test_dopo_rimanda_il_controllo(env):
    brain, ha, llm, mem, clock = env
    llm.vision_results = [nuovo()]
    await brain.tick()
    task = mem.open_tasks()[0]
    await brain.on_notification_action({"data": {"action": f"HUP_DOPO_{task.id}"}})
    clock.advance(31)
    await brain.tick()  # nessuna verifica: è stato rimandato di 60 minuti
    assert len(llm.prompts) == 1
    clock.advance(30)
    llm.vision_results = [tutto_ok([{"task_id": task.id, "completato": True, "messaggio": "ok"}])]
    await brain.tick()
    assert mem.task(task.id).status == "fatto"


@pytest.mark.asyncio
async def test_stanza_invariata_non_chiama_ai(env):
    brain, ha, llm, mem, clock = env
    llm.vision_results = [tutto_ok()]
    await brain.tick()
    clock.advance(31)
    await brain.tick()
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_movimento_finito_anticipa_il_controllo(env):
    brain, ha, llm, mem, clock = env
    llm.vision_results = [tutto_ok()]
    await brain.tick()
    clock.advance(12)
    await brain.on_state_changed({"data": {"entity_id": "binary_sensor.motion_cucina",
                                           "new_state": {"state": "off"}}})
    ha.image = jpeg((90, 10, 10))
    clock.advance(3)
    llm.vision_results = [tutto_ok()]
    await brain.tick()
    assert len(llm.prompts) == 2


@pytest.mark.asyncio
async def test_niente_risposta_vocale_ripiega_su_notifica(env):
    brain, ha, llm, mem, clock = env
    ha.states["binary_sensor.motion_letto"] = "on"
    llm.vision_results = [nuovo()]
    await brain.tick()
    await settle()
    ha.states["binary_sensor.motion_letto"] = "off"
    clock.advance(4)
    await brain.tick()
    assert ha.names()[-1] == "notify.mobile_app_tel"
    assert "a voce" in ha.calls[-1][2]["message"]


@pytest.mark.asyncio
async def test_chat_registra_esito_e_regola(env):
    brain, ha, llm, mem, clock = env
    llm.vision_results = [nuovo()]
    await brain.tick()
    task = mem.open_tasks()[0]
    llm.chat_script = [("registra_esito", {"task_id": task.id, "esito": "fatto"}),
                       ("aggiungi_regola", {"stanza": "cucina", "regola": "Il cesto della frutta va bene"})]
    answer = await brain.chat("fatto! e il cesto della frutta lascialo stare")
    assert mem.task(task.id).status == "da_verificare"
    assert mem.rules("Cucina")[0]["text"] == "Il cesto della frutta va bene"
    assert answer.startswith("risposta")
    assert [m["role"] for m in mem.chat_history()] == ["user", "assistant"]


def test_schema_gemini():
    s = _to_gemini_schema(prompts.CHECK_SCHEMA)
    assert s["type"] == "OBJECT"
    assert s["properties"]["verifiche"]["items"]["properties"]["completato"]["type"] == "BOOLEAN"
