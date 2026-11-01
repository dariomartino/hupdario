"""Punto di ingresso dell'app Hup locale."""
from __future__ import annotations

import asyncio
import logging
import os

import aiohttp
from aiohttp import web

from . import config
from .brain import Brain
from .ha import HAClient
from .llm import make_llm
from .memory import Memory
from .telegram import TelegramBot
from .web import make_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hup")


async def main() -> None:
    cfg = config.load()
    async with aiohttp.ClientSession() as session:
        ha = HAClient(session)
        llm = make_llm(session, cfg)
        mem = Memory(os.path.join(cfg.data_dir, "hup.db"))
        brain = Brain(cfg, ha, llm, mem)

        ha.on("state_changed", brain.on_state_changed)
        ha.on("hup_feedback", brain.on_feedback_event)
        ha.on("mobile_app_notification_action", brain.on_notification_action)

        tasks = [asyncio.create_task(ha.run_events()), asyncio.create_task(brain.run())]

        if cfg.telegram_bot_token:
            bot = TelegramBot(session, cfg.telegram_bot_token, cfg.telegram_chat_id,
                              lambda text: brain.chat(text, source="telegram"), mem)
            brain.telegram_text = bot.send_text
            brain.telegram_photo = bot.send_photo
            tasks.append(asyncio.create_task(bot.run()))
        else:
            log.info("Telegram non configurato")

        runner = web.AppRunner(make_app(brain))
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", 8099).start()
        log.info("Hup locale avviato: stanze %s, AI %s", [r.name for r in cfg.rooms], cfg.llm_provider)
        mem.log("avvio", f"Avviato ({cfg.llm_provider})")
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
