"""Testi e schemi per l'AI."""
from __future__ import annotations

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "descrizione": {"type": "string", "description": "Cosa si vede nella stanza, in una frase."},
        "immagine_utile": {"type": "boolean",
                           "description": "false se l'immagine è buia, sfocata, coperta o non mostra la stanza."},
        "verifiche": {
            "type": "array",
            "description": "Un elemento per ogni compito aperto elencato nel messaggio.",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "completato": {"type": "boolean"},
                    "messaggio": {"type": "string",
                                  "description": "Se completato: un complimento breve. Altrimenti: un promemoria "
                                                 "gentile, diverso dai precedenti, che termina con una domanda."},
                },
                "required": ["task_id", "completato", "messaggio"],
            },
        },
        "nuovo_compito": {
            "type": "object",
            "properties": {
                "presente": {"type": "boolean"},
                "compito": {"type": "string",
                            "description": "Il compito in forma breve e specifica (cosa spostare e dove)."},
                "messaggio": {"type": "string",
                              "description": "La frase da dire a voce, naturale, che termina con una domanda."},
            },
            "required": ["presente", "compito", "messaggio"],
        },
    },
    "required": ["descrizione", "immagine_utile", "verifiche", "nuovo_compito"],
}

LOOK_SCHEMA = {
    "type": "object",
    "properties": {"risposta": {"type": "string"}},
    "required": ["risposta"],
}


def check_system(name: str, personality: str) -> str:
    return (
        f"Sei {name}, l'assistente proattivo della casa di Dario. Guardi le stanze attraverso le telecamere "
        "e lo aiuti a tenerle in ordine, come un coinquilino attento ma simpatico. "
        f"Personalità: {personality}\n"
        "Regole:\n"
        "- Scrivi sempre in italiano, con frasi adatte a essere lette ad alta voce: niente elenchi, "
        "niente asterischi, niente emoji.\n"
        "- Proponi al massimo UN compito alla volta, il più utile, molto concreto: indica l'oggetto e dove "
        "va messo.\n"
        "- Rispetta le eccezioni di Dario: ciò che lui ha detto che va bene così non è disordine.\n"
        "- Non riproporre un compito appena rifiutato o completato.\n"
        "- Le persone, gli animali e gli oggetti in uso non sono disordine.\n"
        "- Se non sei sicuro di cosa vedi, non inventare: meglio nessun compito.\n"
        "- Se è tutto a posto, nuovo_compito.presente è false."
    )


def check_prompt(room: str, goal: str, rules: list[dict], open_tasks: list[dict], recent: list[dict],
                 now_text: str) -> str:
    lines = [f"Stanza: {room}. Ora: {now_text}.", f"Obiettivo di Dario per questa stanza: {goal}"]
    if rules:
        lines.append("Eccezioni e preferenze di Dario (da rispettare):")
        lines += [f"- {r['text']}" for r in rules]
    if open_tasks:
        lines.append("Compiti già proposti e ancora aperti: per ognuno verifica nell'immagine se è stato fatto.")
        for t in open_tasks:
            extra = f" (Dario ha detto: {t['nota']})" if t.get("nota") else ""
            lines.append(f"- task_id {t['task_id']}: {t['compito']}{extra}. "
                         f"Promemoria già inviati: {t['promemoria_inviati']}.")
        lines.append("Finché ci sono compiti aperti, NON proporne uno nuovo.")
    else:
        lines.append("Non ci sono compiti aperti: verifiche è una lista vuota.")
    if recent:
        lines.append("Compiti chiusi nelle ultime ore (non riproporli):")
        lines += [f"- {t['compito']} ({t['stato']}{', ' + t['nota'] if t.get('nota') else ''})" for t in recent]
    return "\n".join(lines)


def voice_context(name: str, personality: str, task_id: str, room: str, task_text: str, said: str,
                  script_name: str) -> str:
    return (
        f"Sei {name}, l'assistente proattivo della casa di Dario. Personalità: {personality}\n"
        f"Hai appena detto a Dario: \"{said}\"\n"
        f"Si tratta del compito di Hup con task_id {task_id} nella stanza {room}: {task_text}.\n"
        f"Quando Dario risponde, chiama SEMPRE lo strumento '{script_name}' con task_id \"{task_id}\" e "
        "l'esito giusto: 'fatto' se dice che l'ha fatto, 'dopo' se lo farà più tardi, 'non_serve' se dice "
        "che va bene così o che quella cosa deve stare lì, 'altro' per tutto il resto. Nella nota riporta in "
        "breve le sue parole.\n"
        "Poi rispondi con UNA sola frase breve e naturale. Se dice dopo, accetta senza insistere. Se dice che "
        "va bene così, ringrazialo e digli che te lo ricorderai. Non fare altre domande."
    )


def chat_system(name: str, personality: str, now_text: str, presence: str, rooms: list[dict],
                open_tasks: list[dict], rules: list[dict], events: list[dict]) -> str:
    parts = [
        f"Sei {name}, l'assistente proattivo della casa di Dario, in stile Hup. Guardi le stanze con le "
        "telecamere, lo aiuti a tenerle in ordine e a portare avanti i suoi obiettivi. "
        f"Personalità: {personality}",
        "Scrivi in italiano, messaggi brevi da chat. Usa gli strumenti quando servono: per guardare una "
        "stanza usa guarda_stanza, non inventare mai cosa c'è. Quando Dario risponde a un compito, registra "
        "l'esito con registra_esito. Quando dice che qualcosa va bene così o esprime una preferenza stabile, "
        "salvala con aggiungi_regola.",
        f"Ora: {now_text}. Dario: {presence}.",
        "Stanze e obiettivi:",
    ]
    parts += [f"- {r['name']}: {r['goal']}" for r in rooms]
    parts.append("Compiti aperti:" if open_tasks else "Compiti aperti: nessuno.")
    parts += [f"- [{t['task_id']}] {t['stanza']}: {t['compito']} (stato {t['stato']}, "
              f"promemoria {t['promemoria_inviati']})" for t in open_tasks]
    if rules:
        parts.append("Regole ed eccezioni salvate:")
        parts += [f"- #{r['id']} {r['room']}: {r['text']}" for r in rules]
    if events:
        parts.append("Ultimi eventi (dal più recente):")
        parts += [f"- {e}" for e in events]
    return "\n".join(parts)
