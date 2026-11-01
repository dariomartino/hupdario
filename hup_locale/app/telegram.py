"""Bot Telegram in long polling: la chat testuale con l'assistente."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import aiohttp

log = logging.getLogger("hup.telegram")

OnMessage = Callable[[str], Awaitable[str]]


class TelegramBot:
    def __init__(self, session: aiohttp.ClientSession, token: str, chat_id: str, on_message: OnMessage, mem):
        self.session, self.token, self.chat_id = session, token, str(chat_id or "").strip()
        self.on_message, self.mem = on_message, mem
        self.base = f"https://api.telegram.org/bot{token}"

    async def _call(self, method: str, **kwargs) -> dict:
        async with self.session.post(f"{self.base}/{method}", timeout=aiohttp.ClientTimeout(total=70),
                                     **kwargs) as r:
            data = await r.json(content_type=None)
            if not data.get("ok"):
                raise RuntimeError(f"Telegram {method}: {data.get('description')}")
            return data["result"]

    async def send_text(self, text: str, chat_id: str | None = None) -> None:
        target = chat_id or self.chat_id
        if not target:
            log.warning("Telegram: chat_id non impostato, messaggio non inviato")
            return
        for i in range(0, len(text), 4000):
            await self._call("sendMessage", json={"chat_id": target, "text": text[i:i + 4000]})

    async def send_photo(self, img: bytes, caption: str = "") -> None:
        if not self.chat_id:
            return
        form = aiohttp.FormData()
        form.add_field("chat_id", self.chat_id)
        form.add_field("caption", caption)
        form.add_field("photo", img, filename="stanza.jpg", content_type="image/jpeg")
        await self._call("sendPhoto", data=form)

    async def _typing(self) -> None:
        try:
            await self._call("sendChatAction", json={"chat_id": self.chat_id, "action": "typing"})
        except Exception:  # noqa: BLE001
            pass

    async def run(self) -> None:
        offset = int(self.mem.get("telegram_offset", 0))
        log.info("Bot Telegram avviato")
        while True:
            try:
                updates = await self._call("getUpdates", json={"offset": offset, "timeout": 50,
                                                                "allowed_updates": ["message"]})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Telegram getUpdates: %s", exc)
                await asyncio.sleep(10)
                continue
            for upd in updates:
                offset = upd["update_id"] + 1
                self.mem.set("telegram_offset", offset)
                msg = upd.get("message") or {}
                chat = str((msg.get("chat") or {}).get("id", ""))
                text = (msg.get("text") or "").strip()
                if not chat or not text:
                    continue
                if not self.chat_id:
                    await self.send_text(
                        f"Ciao! Il tuo chat id è {chat}. Inseriscilo nell'opzione telegram_chat_id "
                        "dell'app Hup locale e riavviala.", chat_id=chat)
                    continue
                if chat != self.chat_id:
                    log.warning("Messaggio Telegram ignorato da chat sconosciuta %s", chat)
                    continue
                if text == "/start":
                    text = "Ciao! Presentati in breve e dimmi come posso usarti."
                await self._typing()
                try:
                    answer = await self.on_message(text)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Errore nella chat")
                    answer = f"Ho avuto un problema: {exc}"
                await self.send_text(answer)
