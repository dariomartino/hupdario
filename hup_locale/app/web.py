"""Pannello minimo (Ingress): compiti, regole, diario e pulsante per controllare subito."""
from __future__ import annotations

import html
from datetime import datetime

from aiohttp import web

STYLE = """
:root{--bg:#f6f5f2;--card:#fff;--text:#1f1e1c;--muted:#6b6a65;--line:#e4e2dc;--accent:#5b4fc7}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--card:#262624;--text:#ecebe6;--muted:#a3a29b;--line:#3a3936;--accent:#a99ff0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px}h1{font-size:22px;font-weight:600;margin:0 0 4px}
h2{font-size:16px;font-weight:600;margin:24px 0 8px}.sub{color:var(--muted);margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:8px}
.row{display:flex;gap:8px;align-items:baseline;justify-content:space-between;flex-wrap:wrap}
.tag{font-size:12px;color:var(--muted)}button{background:var(--accent);color:#fff;border:0;border-radius:8px;
padding:8px 12px;font:inherit;cursor:pointer}table{width:100%;border-collapse:collapse}
td{padding:6px 4px;border-bottom:1px solid var(--line);vertical-align:top}td.t{color:var(--muted);white-space:nowrap;width:1%}
"""


def make_app(brain) -> web.Application:
    app = web.Application()
    tz = brain.cfg.tz

    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz).strftime("%d/%m %H:%M")

    async def index(_req: web.Request) -> web.Response:
        mem = brain.mem
        _, presence = await brain.availability()
        e = html.escape
        rooms = "".join(
            f'<div class="card"><div class="row"><strong>{e(r.name)}</strong>'
            f'<form method="post" action="check"><input type="hidden" name="room" value="{e(r.name)}">'
            f'<button>Controlla ora</button></form></div><div class="tag">{e(brain.goal(r))}</div></div>'
            for r in brain.cfg.rooms)
        tasks = "".join(
            f'<div class="card"><div class="row"><span>{e(t.text)}</span><span class="tag">{e(t.room)} · '
            f'{e(t.status)} · {fmt(t.updated_at)}</span></div>'
            + (f'<div class="tag">{e(t.note)}</div>' if t.note else "") + "</div>"
            for t in mem.tasks_page(20)) or '<p class="sub">Nessun compito finora.</p>'
        rules = "".join(f'<div class="card">#{r["id"]} {e(r["room"])}: {e(r["text"])}</div>'
                        for r in mem.rules()) or '<p class="sub">Nessuna regola salvata.</p>'
        events = "".join(
            f'<tr><td class="t">{fmt(ev["ts"])}</td><td class="t">{e(ev["room"] or "")}</td>'
            f'<td>{e(ev["kind"])}: {e(ev["text"])}</td></tr>' for ev in mem.events(60))
        body = (f'<!doctype html><html lang="it"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Hup locale</title>'
                f'<style>{STYLE}</style></head><body><main><h1>{e(brain.cfg.assistant_name)}</h1>'
                f'<p class="sub">Dario: {e(presence)}</p><h2>Stanze</h2>{rooms}<h2>Compiti</h2>{tasks}'
                f'<h2>Regole ed eccezioni</h2>{rules}<h2>Diario</h2><table>{events}</table></main></body></html>')
        return web.Response(text=body, content_type="text/html")

    async def check(req: web.Request) -> web.Response:
        form = await req.post()
        room = brain.cfg.room(str(form.get("room", "")))
        if room:
            await brain.check_room(room, "pannello", force=True)
        raise web.HTTPSeeOther(location="./")

    app.router.add_get("/", index)
    app.router.add_post("/check", check)
    return app
