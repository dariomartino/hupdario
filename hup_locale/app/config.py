"""Configurazione dell'app: legge /data/options.json (o HUP_OPTIONS per lo sviluppo)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo


@dataclass
class Room:
    name: str
    camera: str
    goal: str
    motion: str | None = None


@dataclass
class Satellite:
    entity: str
    motion: list[str] = field(default_factory=list)


@dataclass
class Config:
    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    anthropic_api_key: str
    anthropic_model: str
    assistant_name: str
    personality: str
    person_entity: str
    sleep_entity: str | None
    privacy_entity: str | None
    quiet_start: time
    quiet_end: time
    tz: ZoneInfo
    rooms: list[Room]
    satellites: list[Satellite]
    feedback_script: str
    notify_service: str | None
    telegram_bot_token: str
    telegram_chat_id: str
    check_interval_minutes: int
    min_minutes_between_checks: int
    motion_cleared_minutes: int
    reminder_minutes: int
    postpone_minutes: int
    max_reminders: int
    max_open_tasks: int
    voice_presence_minutes: int
    voice_reply_timeout_minutes: int
    data_dir: str = "/data"

    def room(self, name: str) -> Room | None:
        key = name.strip().lower()
        for r in self.rooms:
            if r.name.lower() == key:
                return r
        return None


def _parse_time(value: str) -> time:
    hh, mm = value.strip().split(":")[:2]
    return time(int(hh), int(mm))


def load(path: str | None = None) -> Config:
    path = path or os.environ.get("HUP_OPTIONS", "/data/options.json")
    with open(path, encoding="utf-8") as fh:
        o = json.load(fh)
    return from_dict(o, data_dir=os.environ.get("HUP_DATA_DIR", os.path.dirname(path) or "/data"))


def from_dict(o: dict, data_dir: str = "/data") -> Config:
    rooms = [
        Room(name=r["name"], camera=r["camera"], goal=r.get("goal", ""), motion=r.get("motion") or None)
        for r in o.get("rooms", [])
    ]
    sats = [Satellite(entity=s["entity"], motion=list(s.get("motion") or [])) for s in o.get("satellites", [])]
    return Config(
        llm_provider=o.get("llm_provider", "gemini"),
        gemini_api_key=o.get("gemini_api_key", "") or "",
        gemini_model=o.get("gemini_model", "gemini-flash-latest"),
        anthropic_api_key=o.get("anthropic_api_key", "") or "",
        anthropic_model=o.get("anthropic_model", "claude-sonnet-5-5"),
        assistant_name=o.get("assistant_name", "Hup"),
        personality=o.get("personality", "Amichevole e diretto."),
        person_entity=o.get("person_entity", "person.dario"),
        sleep_entity=o.get("sleep_entity") or None,
        privacy_entity=o.get("privacy_entity") or None,
        quiet_start=_parse_time(o.get("quiet_hours_start", "23:30")),
        quiet_end=_parse_time(o.get("quiet_hours_end", "07:00")),
        tz=ZoneInfo(o.get("timezone", "Europe/Rome")),
        rooms=rooms,
        satellites=sats,
        feedback_script=o.get("feedback_script", "script.hup_risposta"),
        notify_service=o.get("notify_service") or None,
        telegram_bot_token=o.get("telegram_bot_token", "") or "",
        telegram_chat_id=str(o.get("telegram_chat_id", "") or ""),
        check_interval_minutes=int(o.get("check_interval_minutes", 30)),
        min_minutes_between_checks=int(o.get("min_minutes_between_checks", 10)),
        motion_cleared_minutes=int(o.get("motion_cleared_minutes", 3)),
        reminder_minutes=int(o.get("reminder_minutes", 30)),
        postpone_minutes=int(o.get("postpone_minutes", 60)),
        max_reminders=int(o.get("max_reminders", 3)),
        max_open_tasks=int(o.get("max_open_tasks", 2)),
        voice_presence_minutes=int(o.get("voice_presence_minutes", 10)),
        voice_reply_timeout_minutes=int(o.get("voice_reply_timeout_minutes", 3)),
        data_dir=data_dir,
    )
