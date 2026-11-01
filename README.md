# Hup locale

Un assistente proattivo in stile [Hup](https://www.withhup.com) che gira come app di Home Assistant.
Guarda le stanze dalle telecamere, ti propone un compito alla volta per tenerle in ordine, ne parla
con te a voce (View Assist / satelliti Assist), via notifica dell'app Home Assistant o su Telegram,
e verifica da solo quando hai finito.

## Come funziona

- **Quando guarda:** a intervalli regolari, qualche minuto dopo che sei uscito dalla stanza, oppure
  quando glielo chiedi. Solo se sei in casa, non stai dormendo e non è la fascia di silenzio.
  Se la stanza non è cambiata dall'ultima volta non chiama nemmeno l'AI.
- **Cosa fa:** propone al massimo un compito per stanza. Poi ricontrolla: se l'hai fatto ti fa i
  complimenti, altrimenti ti ricorda con gentilezza (fino a `max_reminders` volte, poi lascia perdere).
- **Dove ti parla:** se c'è movimento vicino a un satellite vocale avvia un dialogo lì; altrimenti
  manda una notifica all'app con i pulsanti *Fatto*, *Dopo* e *Rispondi*. Telegram è la chat libera.
- **Memoria:** ricorda i compiti, le tue risposte e le eccezioni ("lo stendino sta lì apposta").
  Quando dici che una cosa va bene così, diventa una regola e non te la ripropone.

## Cosa serve in Home Assistant

Già creati durante la configurazione:

- `script.hup_risposta` (*Hup · Registra risposta*), esposto ad Assist: l'AI della conversazione
  vocale lo usa per passare la tua risposta all'app.
- `input_boolean.hup_dormo` (*Hup · Dormo*): quando è acceso l'assistente non parla. L'automazione
  *🤖 Hup · Stato sonno* lo spegne al primo movimento in cucina del mattino (o alle 12).
- `input_boolean.telecamere_in_privacy` (*Telecamere in privacy*) e l'automazione
  *🔒 Sicurezza · Privacy telecamere*: le telecamere restano sempre accese, tranne in privacy, che
  le spegne dalla presa. La privacy si attiva da sola con *Ospiti in casa*; puoi attivarla anche a
  voce ("attiva privacy telecamere") o scrivendolo al bot. In privacy l'assistente si mette in pausa.
- La pipeline del satellite vocale deve usare un agente di conversazione con AI (es. Google AI
  Conversation) con il controllo di Home Assistant attivo (API *Assist*).

## Installazione

1. **Impostazioni → App → Store delle app → ⋮ → Repository**, aggiungi l'URL di questo repository.
2. Installa **Hup locale**.
3. Nella scheda **Configurazione** inserisci almeno `gemini_api_key` (oppure `llm_provider: anthropic`
   e `anthropic_api_key`), poi avvia l'app.
4. Apri il pannello **Hup** nella barra laterale per vedere compiti, regole e diario.

### Telegram

1. Su Telegram scrivi a **@BotFather**, comando `/newbot`, e copia il token in `telegram_bot_token`.
2. Avvia l'app, scrivi un messaggio qualsiasi al tuo bot: ti risponderà con il tuo *chat id*.
3. Copialo in `telegram_chat_id` e riavvia l'app. Il bot risponde solo a quella chat.

## Opzioni principali

| Opzione | Significato |
|---|---|
| `rooms` | Stanze da seguire: nome, telecamera, sensore di movimento, obiettivo in linguaggio naturale |
| `satellites` | Satelliti vocali e i sensori di movimento che dicono se sei lì vicino |
| `check_interval_minutes` | Ogni quanto ricontrollare una stanza (se è cambiata) |
| `max_open_tasks` | Quanti compiti al massimo possono essere aperti in tutta la casa |
| `privacy_entity` | L'interruttore che mette in pausa l'assistente e le telecamere |
| `reminder_minutes` / `postpone_minutes` | Attesa prima di un promemoria / dopo un "dopo" |
| `quiet_hours_start` / `quiet_hours_end` | Fascia in cui l'assistente non parla mai |
| `personality` | Il carattere dell'assistente |

## Sviluppo

```bash
pip install aiohttp pillow tzdata pytest pytest-asyncio
python -m pytest tests
```
