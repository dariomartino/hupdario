"""Client minimale per Home Assistant: REST per stati, servizi e snapshot; WebSocket per gli eventi."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Awaitable, Callable

import aiohttp

log = logging.getLogger("hup.ha")

EventHandler = Callable[[dict], Awaitable[None]]


class HAClient:
    def __init__(self, session: aiohttp.ClientSession):
        token = os.environ.get("SUPERVISOR_TOKEN")
        if token:
            self.api = "http://supervisor/core/api"
            self.ws_url = "ws://supervisor/core/websocket"
        else:  # sviluppo fuori da Home Assistant
            base = os.environ["HA_URL"].rstrip("/")
            token = os.environ["HA_TOKEN"]
            self.api = f"{base}/api"
            self.ws_url = base.replace("http", "ws", 1) + "/api/websocket"
        self.token = token
        self.session = session
        self._handlers: dict[str, list[EventHandler]] = {}
        self._tasks: set[asyncio.Task] = set()  # riferimenti forti: altrimenti i task possono sparire

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    # ---------- REST ----------
    async def get_state(self, entity_id: str) -> dict | None:
        async with self.session.get(f"{self.api}/states/{entity_id}", headers=self._headers) as r:
            if r.status == 404:
                return None
            r.raise_for_status()
            return await r.json()

    async def state(self, entity_id: str) -> str | None:
        s = await self.get_state(entity_id)
        return s["state"] if s else None

    async def call_service(self, domain: str, service: str, data: dict | None = None,
                           return_response: bool = False):
        url = f"{self.api}/services/{domain}/{service}"
        if return_response:
            url += "?return_response"
        async with self.session.post(url, headers=self._headers, json=data or {},
                                     timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status >= 400:
                body = await r.text()
                raise RuntimeError(f"{domain}.{service} -> {r.status}: {body[:300]}")
            return await r.json()

    async def camera_snapshot(self, entity_id: str) -> bytes:
        async with self.session.get(f"{self.api}/camera_proxy/{entity_id}", headers=self._headers,
                                    timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            return await r.read()

    # ---------- WebSocket ----------
    def on(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def run_events(self) -> None:
        """Resta collegato al WebSocket e smista gli eventi; si riconnette da solo."""
        delay = 2
        while True:
            try:
                await self._ws_session()
                delay = 2
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("WebSocket HA caduto (%s), riprovo tra %ss", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    async def _ws_session(self) -> None:
        async with self.session.ws_connect(self.ws_url, heartbeat=30) as ws:
            msg = await ws.receive_json()
            if msg.get("type") == "auth_required":
                await ws.send_json({"type": "auth", "access_token": self.token})
                msg = await ws.receive_json()
            if msg.get("type") != "auth_ok":
                raise RuntimeError(f"autenticazione WebSocket fallita: {msg}")
            for i, ev in enumerate(self._handlers, start=1):
                await ws.send_json({"id": i, "type": "subscribe_events", "event_type": ev})
            log.info("Collegato agli eventi di Home Assistant: %s", ", ".join(self._handlers))
            async for raw in ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    if raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                data = json.loads(raw.data)
                if data.get("type") != "event":
                    continue
                event = data["event"]
                etype = event.get("event_type")
                if etype != "state_changed":
                    log.info("Evento ricevuto: %s %s", etype, event.get("data"))
                for h in self._handlers.get(etype, []):
                    task = asyncio.create_task(self._safe(h, event))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def _safe(handler: EventHandler, event: dict) -> None:
        try:
            await handler(event)
        except Exception:  # noqa: BLE001
            log.exception("Errore nel gestire l'evento %s", event.get("event_type"))
