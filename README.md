# BuddySummarizer 🤖

A friendly **Telegram bot** that creates **digest summaries** of group chats using the OpenAI API.  
Perfect for when you’ve missed long conversations with friends and don’t want to scroll through hundreds of messages.

---

## ✨ Features
- Stores group messages locally in **SQLite** (no cloud storage).
- Generates **easy-to-read digests** with OpenAI API (hierarchical summarization).
- Keeps **nicknames/mentions unchanged** (`@username` stays as-is).
- Delivers digests **privately to the user’s DM** (with a fallback to the group).
- Automatic retention (old messages are purged after `RETENTION_DAYS`).
- Noise filtering (removes tiny “ok/LOL/emoji”-style replies).

---

## 🗣️ Prompt Language (RU/EN)
The bot ships with **two prompt variants**:
- **Russian (default)** — tuned for casual Russian chats.
- **English (optional)** — ready to switch.

To switch to English, open `main.py` and in the **“PROMPTS”** section **uncomment**:
```python
# PROMPT_CHUNK = PROMPT_CHUNK_EN
# PROMPT_FINAL = PROMPT_FINAL_EN
````

…and **comment** the RU lines if desired:

```python
# PROMPT_CHUNK = PROMPT_CHUNK_RU
# PROMPT_FINAL = PROMPT_FINAL_RU
```

No other changes are needed.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourname/buddysummarizer.git
cd buddysummarizer

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (cmd):
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file:

```env
BOT_TOKEN=123456:ABC-your-telegram-bot-token
OPENAI_API_KEY=sk-your-openai-key
MODEL=gpt-5-mini

# Optional
LOG_LEVEL=INFO
RETENTION_DAYS=30
SUMMARY_PARSE_MODE=plain   # plain | markdown_v2 | html
CHUNK_MAX_CHARS=6000
INPUT_MAX_CHARS=180000
MIN_TEXT_LEN=5
```

* `BOT_TOKEN` – from [BotFather](https://t.me/BotFather).
* `OPENAI_API_KEY` – from [OpenAI](https://platform.openai.com/).
* `RETENTION_DAYS` – how long to keep messages in the DB.

**Telegram setup tips**

* Disable **Privacy Mode** in BotFather so the bot can see group messages.
* To receive DMs from the bot, start a private chat with it and press **Start**.

---

## ▶️ Run

```bash
python main.py
```

The bot runs in polling mode. Add it to your group and try:

* `/catchup` — digest since last time
* `/catchup 12h` or `/catchup 2d` — time-bounded digest
* `/catchup since YYYY-MM-DD [HH:MM]` — from a date
* `/mentions` — your recent mentions
* `/stats` — DB diagnostics

---

## 📦 Requirements

See `requirements.txt`:

```
python-telegram-bot==21.4
openai>=1.40.0
python-dotenv>=1.0.1
```

Python 3.10+ is recommended.

---

## 🛡️ Privacy

* Messages are stored **locally** in `bot.db` (SQLite).
* No external storage.
* Automatic cleanup of old messages after `RETENTION_DAYS`.

---

## 📜 License

MIT © 2025 Mike Yastrebtsov
