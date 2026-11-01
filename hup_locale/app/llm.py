"""Livello AI indipendente dal fornitore: Gemini o Claude, stessa interfaccia.

Due operazioni:
- vision_json: immagini + istruzioni -> JSON con uno schema fisso (per i controlli delle stanze)
- chat: conversazione con strumenti (per Telegram e le risposte alle notifiche)
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import aiohttp

log = logging.getLogger("hup.llm")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema (tipi minuscoli)


ToolHandler = Callable[[str, dict], Awaitable[Any]]

MAX_TOOL_ROUNDS = 6


def _to_gemini_schema(schema: dict) -> dict:
    """Converte uno JSON Schema semplice nel formato OpenAPI di Gemini (tipi maiuscoli)."""
    out: dict = {}
    for key, value in schema.items():
        if key == "type":
            out["type"] = value.upper()
        elif key == "properties":
            out["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        elif key in ("required", "enum", "description", "nullable"):
            out[key] = value
    return out


class LLMError(RuntimeError):
    pass


class LLM:
    async def vision_json(self, system: str, prompt: str, images: list[bytes], schema: dict) -> dict:
        raise NotImplementedError

    async def chat(self, system: str, history: list[dict], tools: list[Tool], handler: ToolHandler) -> str:
        raise NotImplementedError


class GeminiLLM(LLM):
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Ragionamento ridotto al minimo per avere risposte rapide. I modelli Gemini 3 usano
    # thinkingLevel, i 2.5 thinkingBudget: provo nell'ordine e ricordo quello accettato.
    THINKING = [{"thinkingLevel": "low"}, {"thinkingBudget": 0}, None]

    def __init__(self, session: aiohttp.ClientSession, api_key: str, model: str, vision_model: str | None = None):
        self.session, self.key, self.model = session, api_key, model
        self.vision_model = vision_model or model
        self._thinking = 0

    async def _post(self, body: dict, model: str | None = None) -> dict:
        model = model or self.model
        while True:
            thinking = self.THINKING[self._thinking]
            payload = dict(body)
            gen = dict(payload.get("generationConfig", {}))
            if thinking:
                gen["thinkingConfig"] = thinking
            if gen:
                payload["generationConfig"] = gen
            async with self.session.post(
                self.URL.format(model=model),
                headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as r:
                data = await r.json(content_type=None)
            if r.status == 400 and thinking and "think" in json.dumps(data).lower():
                self._thinking += 1
                log.info("Gemini non accetta %s, provo l'alternativa", thinking)
                continue
            if r.status >= 400:
                raise LLMError(f"Gemini {r.status}: {json.dumps(data)[:400]}")
            return data

    @staticmethod
    def _content(data: dict) -> dict:
        try:
            return data["candidates"][0]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Risposta Gemini senza contenuto: {json.dumps(data)[:400]}") from exc

    async def vision_json(self, system, prompt, images, schema):
        parts = [{"text": prompt}] + [
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode()}} for img in images
        ]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _to_gemini_schema(schema),
                "temperature": 0.4,
            },
        }
        try:
            data = await self._post(body, self.vision_model)
        except LLMError as exc:
            if self.vision_model == self.model or " 404" not in f" {exc}":
                raise
            log.warning("Modello %s non disponibile, uso %s", self.vision_model, self.model)
            self.vision_model = self.model
            data = await self._post(body, self.model)
        content = self._content(data)
        text = "".join(p.get("text", "") for p in content.get("parts", []))
        return json.loads(text)

    async def chat(self, system, history, tools, handler):
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["text"]}]}
            for m in history
        ]
        body: dict = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents}
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": _to_gemini_schema(t.parameters)}
                for t in tools
            ]}]
        for _ in range(MAX_TOOL_ROUNDS):
            content = self._content(await self._post(body))
            parts = content.get("parts", [])
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not calls:
                return "".join(p.get("text", "") for p in parts).strip()
            # Il contenuto del modello va rimandato così com'è (contiene le thought signature).
            contents.append({"role": "model", "parts": parts})
            responses = []
            for call in calls:
                result = await handler(call["name"], call.get("args") or {})
                responses.append({"functionResponse": {"name": call["name"], "response": {"result": result}}})
            contents.append({"role": "user", "parts": responses})
        return "Scusa, mi sono incartato: riprova a chiedermelo."


class AnthropicLLM(LLM):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, session: aiohttp.ClientSession, api_key: str, model: str):
        self.session, self.key, self.model = session, api_key, model

    async def _post(self, body: dict) -> dict:
        async with self.session.post(
            self.URL,
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json=body,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as r:
            data = await r.json(content_type=None)
            if r.status >= 400:
                raise LLMError(f"Anthropic {r.status}: {json.dumps(data)[:400]}")
            return data

    async def vision_json(self, system, prompt, images, schema):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(img).decode()}}
            for img in images
        ] + [{"type": "text", "text": prompt}]
        body = {
            "model": self.model,
            "max_tokens": 1500,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "tools": [{"name": "risultato", "description": "Restituisce il risultato del controllo",
                       "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "risultato"},
        }
        data = await self._post(body)
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                return block["input"]
        raise LLMError(f"Risposta Claude senza risultato: {json.dumps(data)[:400]}")

    async def chat(self, system, history, tools, handler):
        messages = [{"role": m["role"], "content": m["text"]} for m in history]
        body: dict = {"model": self.model, "max_tokens": 1500, "system": system, "messages": messages}
        if tools:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                             for t in tools]
        for _ in range(MAX_TOOL_ROUNDS):
            data = await self._post(body)
            blocks = data.get("content", [])
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            if not uses:
                return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            messages.append({"role": "assistant", "content": blocks})
            results = []
            for use in uses:
                result = await handler(use["name"], use.get("input") or {})
                results.append({"type": "tool_result", "tool_use_id": use["id"],
                                "content": json.dumps(result, ensure_ascii=False)})
            messages.append({"role": "user", "content": results})
        return "Scusa, mi sono incartato: riprova a chiedermelo."


def make_llm(session: aiohttp.ClientSession, cfg) -> LLM:
    if cfg.llm_provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise LLMError("Manca anthropic_api_key nelle opzioni")
        return AnthropicLLM(session, cfg.anthropic_api_key, cfg.anthropic_model)
    if not cfg.gemini_api_key:
        raise LLMError("Manca gemini_api_key nelle opzioni")
    return GeminiLLM(session, cfg.gemini_api_key, cfg.gemini_model, cfg.gemini_vision_model)
