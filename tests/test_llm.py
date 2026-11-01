"""Test degli adattatori Gemini e Claude contro un server finto."""
import sys
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hup_locale"))

from app.llm import AnthropicLLM, GeminiLLM, Tool  # noqa: E402

TOOL = Tool("registra_esito", "x", {"type": "object", "properties": {"task_id": {"type": "string"}},
                                    "required": ["task_id"]})


async def serve(handler):
    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_gemini_vision_e_chat_con_strumenti():
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        if "responseSchema" in body.get("generationConfig", {}):
            assert body["generationConfig"]["responseSchema"]["type"] == "OBJECT"
            assert "inline_data" in body["contents"][0]["parts"][1]
            return web.json_response({"candidates": [{"content": {"parts": [{"text": '{"risposta": "ok"}'}]}}]})
        if len(body["contents"]) == 1:
            return web.json_response({"candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": "registra_esito", "args": {"task_id": "abc"}},
                 "thoughtSignature": "sig"}]}}]})
        assert body["contents"][1]["parts"][0]["thoughtSignature"] == "sig"
        assert body["contents"][2]["parts"][0]["functionResponse"]["response"]["result"] == {"ok": True}
        return web.json_response({"candidates": [{"content": {"parts": [{"text": "Registrato!"}]}}]})

    runner, base = await serve(handler)
    try:
        async with aiohttp.ClientSession() as s:
            llm = GeminiLLM(s, "k", "m")
            llm.URL = base + "/{model}"
            assert await llm.vision_json("sys", "p", [b"img"], {"type": "object", "properties": {
                "risposta": {"type": "string"}}}) == {"risposta": "ok"}
            calls = []

            async def h(name, args):
                calls.append((name, args))
                return {"ok": True}

            out = await llm.chat("sys", [{"role": "user", "text": "fatto"}], [TOOL], h)
            assert out == "Registrato!" and calls == [("registra_esito", {"task_id": "abc"})]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_anthropic_vision_e_chat_con_strumenti():
    async def handler(req):
        body = await req.json()
        assert req.headers["x-api-key"] == "k"
        if body.get("tool_choice"):
            assert body["messages"][0]["content"][0]["type"] == "image"
            return web.json_response({"content": [{"type": "tool_use", "id": "1", "name": "risultato",
                                                   "input": {"risposta": "ok"}}]})
        if len(body["messages"]) == 1:
            return web.json_response({"content": [{"type": "tool_use", "id": "t1", "name": "registra_esito",
                                                   "input": {"task_id": "abc"}}]})
        assert body["messages"][2]["content"][0]["tool_use_id"] == "t1"
        return web.json_response({"content": [{"type": "text", "text": "Registrato!"}]})

    runner, base = await serve(handler)
    try:
        async with aiohttp.ClientSession() as s:
            llm = AnthropicLLM(s, "k", "m")
            llm.URL = base + "/v1/messages"
            assert await llm.vision_json("sys", "p", [b"img"], {"type": "object"}) == {"risposta": "ok"}

            async def h(name, args):
                return {"ok": True}

            assert await llm.chat("sys", [{"role": "user", "text": "fatto"}], [TOOL], h) == "Registrato!"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gemini_ripiega_se_il_modello_non_accetta_thinking_level():
    seen = []

    async def handler(req):
        body = await req.json()
        tc = body["generationConfig"].get("thinkingConfig")
        seen.append(tc)
        if tc and "thinkingLevel" in tc:
            return web.json_response({"error": {"message": "thinking_level is not supported"}}, status=400)
        return web.json_response({"candidates": [{"content": {"parts": [{"text": '{"risposta": "ok"}'}]}}]})

    runner, base = await serve(handler)
    try:
        async with aiohttp.ClientSession() as s:
            llm = GeminiLLM(s, "k", "m")
            llm.URL = base + "/{model}"
            schema = {"type": "object", "properties": {"risposta": {"type": "string"}}}
            assert await llm.vision_json("sys", "p", [b"img"], schema) == {"risposta": "ok"}
            assert await llm.vision_json("sys", "p", [b"img"], schema) == {"risposta": "ok"}
            assert seen == [{"thinkingLevel": "low"}, {"thinkingBudget": 0}, {"thinkingBudget": 0}]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_gemini_modello_visione_inesistente_ripiega_sul_principale():
    models = []

    async def handler(req):
        models.append(req.match_info["tail"])
        if "lite" in req.match_info["tail"]:
            return web.json_response({"error": {"message": "model not found"}}, status=404)
        return web.json_response({"candidates": [{"content": {"parts": [{"text": '{"risposta": "ok"}'}]}}]})

    runner, base = await serve(handler)
    try:
        async with aiohttp.ClientSession() as s:
            llm = GeminiLLM(s, "k", "main", "lite")
            llm.URL = base + "/{model}"
            schema = {"type": "object", "properties": {"risposta": {"type": "string"}}}
            assert await llm.vision_json("sys", "p", [b"img"], schema) == {"risposta": "ok"}
            assert await llm.vision_json("sys", "p", [b"img"], schema) == {"risposta": "ok"}
            assert models == ["lite", "main", "main"]
    finally:
        await runner.cleanup()
