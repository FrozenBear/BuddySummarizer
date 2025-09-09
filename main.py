import os
import re
import time
import html
import logging
import asyncio
import datetime as dt
import sqlite3
from contextlib import closing

from dotenv import load_dotenv
from telegram import Update, Message
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
from telegram.error import BadRequest, Forbidden
from telegram.helpers import escape_markdown

from openai import OpenAI

# -------------------- ENV & LOGGING --------------------
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
log = logging.getLogger("BuddySummarizer")

BOT_TOKEN = os.environ.get("BOT_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-5-mini")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing (check .env)")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing (check .env)")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_PATH = "bot.db"
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "30"))
SUMMARY_PARSE_MODE = os.getenv("SUMMARY_PARSE_MODE", "plain").lower()  # plain|markdown_v2|html

# Tuning knobs
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "6000"))
INPUT_MAX_CHARS = int(os.getenv("INPUT_MAX_CHARS", "180000"))
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "5"))

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w{3,32}")

# -------------------- PROMPTS (RU default, EN optional) --------------------
# System instruction (language-agnostic, in English to be explicit):
SYSTEM_PROMPT = (
    "You are a helpful assistant that writes light, readable summaries of friendly group chats. "
    "Strictly preserve all nicknames and mentions (@username, names in Latin/Cyrillic) "
    "EXACTLY as in the source text: do not translate, do not transliterate, do not change case."
)

# --- Russian prompts (default) ---
PROMPT_CHUNK_RU = (
    "Это фрагмент дружеской переписки.\n"
    "Сделай короткое мини-резюме этого фрагмента (3–6 пунктов), без официоза.\n"
    "Пиши простым языком, без Markdown/HTML, только обычный текст на ru.\n"
    "Никнеймы и @упоминания оставляй без изменений.\n\n"
    "Фрагмент чата:\n{TEXT}"
)

PROMPT_FINAL_RU = (
    "Ниже несколько мини-резюме фрагментов одной дружеской переписки.\n"
    "Собери из них общий пересказ, но оформи его структурировано и читаемо:\n"
    "- используй буллеты (•) или короткие абзацы;\n"
    "- группируй по темам (например: «История/факты», «Шутки/подколы», «Споры/разногласия»);\n"
    "- если были ссылки — вынеси их отдельным блоком «Ссылки»;\n"
    "- если были конкретные предложения/договорённости — отдельным блоком «Что решили».\n\n"
    "Пиши простым языком, без Markdown/HTML, только обычный текст на ru.\n"
    "Никаких советов, извинений или предложений «сделать ещё». Никнеймы и @упоминания не менять.\n\n"
    "Мини-резюме фрагментов:\n{TEXT}"
)

# --- English prompts (optional: uncomment the two lines in the selection section below) ---
PROMPT_CHUNK_EN = (
    "This is a fragment of a friendly group chat.\n"
    "Write a short mini-summary of this fragment (3–6 bullet-style points), no corporate tone.\n"
    "Use plain English text only (no Markdown/HTML).\n"
    "Keep nicknames and @mentions exactly as in the source.\n\n"
    "Chat fragment:\n{TEXT}"
)

PROMPT_FINAL_EN = (
    "Below are several mini-summaries of fragments from one friendly chat.\n"
    "Combine them into a single readable digest with structure:\n"
    "- use bullets (•) or short paragraphs;\n"
    "- group by themes (e.g., “Facts/Context”, “Jokes”, “Debates/Disagreements”);\n"
    "- if there were links, put them in a separate “Links” section;\n"
    "- if there were proposals/agreements, add a “Decisions” section.\n\n"
    "Use plain English text only (no Markdown/HTML).\n"
    "No advice/apologies/“I can also…”. Keep nicknames and @mentions unchanged.\n\n"
    "Mini-summaries:\n{TEXT}"
)

# --- Prompt selection: RU is default. To switch to EN, uncomment the EN lines and comment the RU lines. ---
PROMPT_CHUNK = PROMPT_CHUNK_RU
PROMPT_FINAL = PROMPT_FINAL_RU

# PROMPT_CHUNK = PROMPT_CHUNK_EN
# PROMPT_FINAL = PROMPT_FINAL_EN

# -------------------- DB INIT --------------------
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            msg_id INTEGER,
            user_id INTEGER,
            username TEXT,
            text TEXT,
            ts INTEGER,
            topic_id INTEGER
        );""")
        db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_unique
        ON messages(chat_id, msg_id);
        """)
        db.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_chat_ts
        ON messages(chat_id, ts);
        """)
        db.execute("""
        CREATE TABLE IF NOT EXISTS last_seen(
            user_id INTEGER,
            chat_id INTEGER,
            last_msg_id INTEGER,
            last_ts INTEGER,
            PRIMARY KEY (user_id, chat_id)
        );""")
        db.commit()
    log.info("DB initialized (path=%s, retention_days=%d)", DB_PATH, RETENTION_DAYS)

# -------------------- HELPERS --------------------
DURATION_RE = re.compile(r"^(?:(\d+)\s*(h|d))$")  # e.g., 12h, 2d

def parse_timespec(arg: str):
    """
    Supports:
      - '12h', '2d'
      - 'since YYYY-MM-DD' or 'since YYYY-MM-DD HH:MM'
    """
    arg = arg.strip()
    m = DURATION_RE.match(arg)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        delta = dt.timedelta(hours=val) if unit == "h" else dt.timedelta(days=val)
        since = dt.datetime.now() - delta
        return int(since.timestamp()), None

    if arg.startswith("since"):
        rest = arg[len("since"):].strip()
        try:
            if " " in rest:
                since_dt = dt.datetime.strptime(rest, "%Y-%m-%d %H:%M")
            else:
                since_dt = dt.datetime.strptime(rest, "%Y-%m-%d")
        except ValueError:
            return None, "Use formats: 'since YYYY-MM-DD' or 'since YYYY-MM-DD HH:MM'"
        return int(since_dt.timestamp()), None

    return None, "Use one of: '/catchup', '/catchup 2d', '/catchup 12h', '/catchup since YYYY-MM-DD [HH:MM]'"

def fetch_messages(chat_id: int, since_ts: int = None, since_msg_id: int = None, limit: int = 5000):
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.row_factory = sqlite3.Row
        if since_msg_id is not None:
            rows = db.execute(
                "SELECT * FROM messages WHERE chat_id=? AND msg_id>? ORDER BY msg_id ASC LIMIT ?",
                (chat_id, since_msg_id, limit)
            ).fetchall()
        elif since_ts is not None:
            rows = db.execute(
                "SELECT * FROM messages WHERE chat_id=? AND ts>=? ORDER BY ts ASC LIMIT ?",
                (chat_id, since_ts, limit)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM messages WHERE chat_id=? ORDER BY ts DESC LIMIT ?",
                (chat_id, limit)
            ).fetchall()
            rows = list(reversed(rows))
        return rows

def store_last_seen(user_id: int, chat_id: int, last_msg_id: int, last_ts: int):
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute("""
        INSERT INTO last_seen(user_id, chat_id, last_msg_id, last_ts)
        VALUES(?,?,?,?)
        ON CONFLICT(user_id, chat_id) DO UPDATE SET
            last_msg_id=excluded.last_msg_id,
            last_ts=excluded.last_ts
        """, (user_id, chat_id, last_msg_id, last_ts))
        db.commit()

def get_last_seen(user_id: int, chat_id: int):
    with closing(sqlite3.connect(DB_PATH)) as db:
        row = db.execute(
            "SELECT last_msg_id, last_ts FROM last_seen WHERE user_id=? AND chat_id=?",
            (user_id, chat_id)
        ).fetchone()
        return row if row else None

def human_ts(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

# ---------- Noise filtering & scoring ----------
NOISE_TOKENS = {"ок", "ок.", "окей", "ладно", "ага", "угу", "лол", "🙂", "😀", "😂", "👍", "👌", "💪"}

def is_noise_text(text: str) -> bool:
    """Very short or trivial messages are considered noise unless they contain a link."""
    t = (text or "").strip().lower()
    if len(t) < MIN_TEXT_LEN and not URL_RE.search(t):
        return True
    if t in NOISE_TOKENS:
        return True
    return False

def score_message(row) -> int:
    """Basic importance score: links, mentions, and length."""
    text = (row["text"] or "")
    s = 0
    if URL_RE.search(text): s += 2
    if MENTION_RE.search(text): s += 1
    if len(text) >= 120: s += 1
    return s

def build_line(row) -> str:
    """Render a single line for the LLM payload: [YYYY-MM-DD HH:MM] username: text"""
    username = row["username"] or str(row["user_id"])
    ts = row["ts"]
    timestr = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    text = row["text"] or ""
    return f"[{timestr}] {username}: {text}"

def shrink_lines_for_model(lines, max_chars=INPUT_MAX_CHARS):
    """Hard limit input size by characters (quick & simple)."""
    out, total = [], 0
    for ln in lines:
        ln = ln[:2000]  # per-line cap
        if total + len(ln) > max_chars:
            break
        out.append(ln)
        total += len(ln)
    return out

# ---------- Chunking & hierarchical summarize ----------
def chunk_lines(lines, max_chars=CHUNK_MAX_CHARS):
    """Split lines into character-bounded chunks for map→reduce summarization."""
    out, cur, size = [], [], 0
    for ln in lines:
        if size + len(ln) > max_chars and cur:
            out.append("\n".join(cur)); cur, size = [], 0
        cur.append(ln); size += len(ln)
    if cur:
        out.append("\n".join(cur))
    return out

# -------------------- OpenAI --------------------
async def openai_summarize_chunk(text: str) -> str:
    """Summarize a single chunk (mini-summary)."""
    prompt = PROMPT_CHUNK.replace("{TEXT}", text)
    log.debug("Calling OpenAI (chunk): model=%s, chars=%d", MODEL, len(prompt))
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    out = resp.output_text.strip()
    log.debug("OpenAI chunk: output_chars=%d", len(out))
    return out

async def openai_summarize_final(text: str) -> str:
    """Combine mini-summaries into a final digest."""
    prompt = PROMPT_FINAL.replace("{TEXT}", text)
    log.debug("Calling OpenAI (final): model=%s, chars=%d", MODEL, len(prompt))
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    out = resp.output_text.strip()
    log.debug("OpenAI final: output_chars=%d", len(out))
    return out

async def summarize_hierarchical(lines) -> str:
    """Map→Reduce: chunk → mini-summaries → final summary."""
    chunks = chunk_lines(lines, max_chars=CHUNK_MAX_CHARS)
    log.info("Hierarchical: %d chunks", len(chunks))
    minis = []
    for i, c in enumerate(chunks, 1):
        log.debug("Summarizing chunk %d/%d, chars=%d", i, len(chunks), len(c))
        minis.append(await openai_summarize_chunk(c))
    joined = "\n\n---\n\n".join(minis)
    return await openai_summarize_final(joined)

# -------------------- SAFE SEND --------------------
def format_for_parse_mode(text: str):
    """Escape text depending on SUMMARY_PARSE_MODE."""
    mode = SUMMARY_PARSE_MODE
    if mode == "markdown_v2":
        return escape_markdown(text, version=2), ParseMode.MARKDOWN_V2
    elif mode == "html":
        return html.escape(text), ParseMode.HTML
    else:
        return text, None

async def safe_reply(message: Message, text: str):
    """Reply with chosen parse mode and fallback to plain on failure."""
    safe_text, parse_mode = format_for_parse_mode(text)
    try:
        return await message.reply_text(safe_text[:4096], parse_mode=parse_mode)
    except BadRequest as e:
        log.warning("Reply failed with parse_mode=%s: %s. Falling back to plain.", parse_mode, e)
        return await message.reply_text(text[:4096])

async def safe_send_dm(context: ContextTypes.DEFAULT_TYPE, user_id: int, text: str):
    """DM with chosen parse mode and fallback to plain; raises Forbidden if DM closed."""
    safe_text, parse_mode = format_for_parse_mode(text)
    try:
        return await context.bot.send_message(chat_id=user_id, text=safe_text[:4096], parse_mode=parse_mode)
    except BadRequest as e:
        log.warning("DM failed with parse_mode=%s: %s. Fallback to plain.", parse_mode, e)
        return await context.bot.send_message(chat_id=user_id, text=text[:4096])
    except Forbidden as e:
        log.warning("DM forbidden for user %s: %s", user_id, e)
        raise

# -------------------- RETENTION --------------------
async def retention_worker():
    """Periodic cleanup of old messages and occasional VACUUM."""
    if RETENTION_DAYS <= 0:
        log.info("Retention disabled (RETENTION_DAYS=%d)", RETENTION_DAYS)
        return
    period = 3600  # 1 hour
    log.info("Retention worker started (RETENTION_DAYS=%d)", RETENTION_DAYS)
    while True:
        try:
            cutoff = int((dt.datetime.now() - dt.timedelta(days=RETENTION_DAYS)).timestamp())
            with closing(sqlite3.connect(DB_PATH)) as db:
                cur = db.execute("SELECT COUNT(*) FROM messages WHERE ts < ?", (cutoff,))
                to_del = cur.fetchone()[0]
                if to_del > 0:
                    db.execute("DELETE FROM messages WHERE ts < ?", (cutoff,))
                    db.commit()
                    log.info("Retention: deleted %d rows older than %s", to_del, human_ts(cutoff))
                # crude periodic VACUUM ~ every 12 hours
                if int(time.time()) % (period * 12) < 5:
                    try:
                        db.execute("VACUUM;")
                        log.info("VACUUM done")
                    except Exception as e:
                        log.warning("VACUUM failed: %s", e)
        except Exception as e:
            log.error("Retention worker error: %s", e)
        await asyncio.sleep(period)

# -------------------- COMMANDS --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg is None:
        log.warning("/start: effective_message is None")
        return
    text = (
        "Привет! Я делаю дружеские дайджесты для группы.\n"
        "Команды:\n"
        "• /catchup — дайджест с последнего раза\n"
        "• /catchup 12h | 2d — за период\n"
        "• /catchup since 2025-09-01 [HH:MM] — с даты\n"
        "• /mentions — только упоминания тебя\n"
        "• /stats — диагностика по БД\n\n"
        "Чтобы получать ответы в личку, один раз напиши мне в DM и нажми Start."
    )
    await safe_reply(msg, text)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg is None:
        log.warning("/stats: effective_message is None")
        return
    chat = update.effective_chat
    with closing(sqlite3.connect(DB_PATH)) as db:
        c = db.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat.id,)).fetchone()[0]
        last = db.execute("SELECT MAX(ts) FROM messages WHERE chat_id=?", (chat.id,)).fetchone()[0]
    text = (
        "В БД этого чата пока нет сообщений. Проверь privacy и что бот запущен."
        if c == 0 or last is None
        else f"Сообщений в БД: {c}\nПоследнее: {human_ts(last)}"
    )

    user = update.effective_user
    try:
        await safe_send_dm(context, user.id, text)
        await msg.reply_text("Отправил статистику тебе в личку 👋 (если не пришло — напиши боту в DM и нажми Start).")
    except Forbidden:
        await msg.reply_text(text)

async def cmd_catchup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg is None:
        log.warning("/catchup: effective_message is None")
        return
    chat = update.effective_chat
    user = update.effective_user

    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Эта команда рассчитана на использование в группе.")
        return

    # Determine period
    since_ts = None
    since_msg_id = None

    if context.args:
        arg = " ".join(context.args)
        since_ts, err = parse_timespec(arg)
        if err and since_ts is None:
            await msg.reply_text(err)
            return
        log.info("/catchup by %s in chat %s (since_ts=%s)", user.username or user.id, chat.id, since_ts)
    else:
        ls = get_last_seen(user.id, chat.id)
        if ls:
            since_msg_id = ls[0]
            log.info("/catchup by %s since last_seen msg_id=%s", user.username or user.id, since_msg_id)
        else:
            since_ts = int((dt.datetime.now() - dt.timedelta(days=2)).timestamp())
            log.info("/catchup by %s no last_seen, default since_ts=%s", user.username or user.id, since_ts)

    rows = fetch_messages(chat.id, since_ts=since_ts, since_msg_id=since_msg_id, limit=5000)
    log.info("Fetched %d rows for summarization", len(rows))
    if not rows:
        await msg.reply_text("Новых сообщений для дайджеста нет.")
        return

    # Preprocessing: noise filtering & prioritization
    filtered = [r for r in rows if not is_noise_text(r["text"] or "")]
    dropped = len(rows) - len(filtered)
    log.info("Pre-filter: kept=%d, dropped=%d", len(filtered), dropped)

    # Sort by score (desc), then by time (asc)
    filtered.sort(key=lambda r: (-score_message(r), r["ts"]))

    # Build lines
    lines = [build_line(r) for r in filtered]
    lines = shrink_lines_for_model(lines, max_chars=INPUT_MAX_CHARS)
    payload_preview = "\n".join(lines[:5])
    log.debug("Payload preview:\n%s\n... (%d lines total)", payload_preview, len(lines))

    await msg.reply_text("Готовлю дайджест…")

    # Hierarchical summarization
    try:
        summary = await summarize_hierarchical(lines)
    except Exception:
        log.exception("Hierarchical summarization failed; falling back to single-pass")
        joined = "\n".join(lines)
        summary = await openai_summarize_final(joined)

    # Deliver via DM; fall back to group if DM closed
    try:
        await safe_send_dm(context, user.id, summary)
        await msg.reply_text("Отправил дайджест тебе в личку 👋 (если не пришло — напиши боту в DM и нажми Start).")
    except Forbidden:
        await safe_reply(msg, summary)

    # Update last_seen
    last_msg_id = max(r["msg_id"] for r in rows) if rows else 0
    last_ts = max(r["ts"] for r in rows) if rows else 0
    store_last_seen(user.id, chat.id, last_msg_id, last_ts)
    log.info("Updated last_seen for user=%s chat=%s to msg_id=%s ts=%s",
             user.id, chat.id, last_msg_id, last_ts)

async def cmd_mentions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg is None:
        log.warning("/mentions: effective_message is None")
        return
    chat = update.effective_chat
    user = update.effective_user

    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Эта команда рассчитана на использование в группе.")
        return

    target = f"@{user.username}" if user and user.username else None
    if not target:
        await msg.reply_text("У вас нет username (@user). Telegram-упоминания не найдены.")
        return

    since_ts = int((dt.datetime.now() - dt.timedelta(days=7)).timestamp())
    rows = fetch_messages(chat.id, since_ts=since_ts, limit=5000)
    hits = []
    for r in rows:
        text = r["text"] or ""
        if target in text:
            username = r["username"] or str(r["user_id"])
            timestr = dt.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
            hits.append(f"[{timestr}] {username}: {text}")

    text = "За последнюю неделю упоминаний не найдено." if not hits else "\n".join(hits[-30:])

    try:
        await safe_send_dm(context, user.id, text)
        await msg.reply_text("Отправил список упоминаний тебе в личку 👋 (если не пришло — напиши боту в DM и нажми Start).")
    except Forbidden:
        await msg.reply_text(text[:4096])

# -------------------- MESSAGE CAPTURE --------------------
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat

    if chat is None or msg is None:
        return
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return
    if not msg.text:
        return

    user = msg.from_user
    username = (user.username if user and user.username else None) or (str(user.id) if user else "unknown")
    topic_id = getattr(msg, "message_thread_id", None)
    ts = int(msg.date.timestamp())

    try:
        with closing(sqlite3.connect(DB_PATH)) as db:
            db.execute(
                "INSERT OR IGNORE INTO messages(chat_id, msg_id, user_id, username, text, ts, topic_id) "
                "VALUES(?,?,?,?,?,?,?)",
                (chat.id, msg.message_id, user.id if user else None, username, msg.text, ts, topic_id)
            )
            db.commit()
        log.debug("[stored] chat=%s msg_id=%s user=%s text=%r",
                  chat.id, msg.message_id, username, msg.text[:80])
    except Exception as e:
        log.error("Failed to store message: %s", e)

# -------------------- ERROR HANDLER --------------------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("Упс, что-то пошло не так. Я залогировал ошибку.")
    except Exception:
        pass

# -------------------- APP --------------------
def build_app():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("catchup", cmd_catchup))
    app.add_handler(CommandHandler("mentions", cmd_mentions))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_message))
    app.add_error_handler(on_error)
    return app

def main():
    init_db()
    app = build_app()
    loop = asyncio.get_event_loop()
    loop.create_task(retention_worker())
    log.info("BuddySummarizer is starting (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)

if __name__ == "__main__":
    main()
