import os
import io
import json
import time
import subprocess
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone

# ---------- Config from environment ----------
NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"].rstrip("/")
NS_ACCESS_TOKEN = os.environ["NIGHTSCOUT_ACCESS_TOKEN"]

IST = timezone(timedelta(hours=5, minutes=30))  # India Standard Time, fixed UTC+5:30

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

WHATSAPP_PHONE_NUMBER_ID = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
WHATSAPP_ACCESS_TOKEN = os.environ["WHATSAPP_ACCESS_TOKEN"]
WHATSAPP_TO = os.environ["WHATSAPP_TO"]  # e.g. 91XXXXXXXXXX, no + sign
WHATSAPP_API_VERSION = os.environ.get("WHATSAPP_API_VERSION", "v21.0")

# Template names — must exactly match what you created and got approved in WhatsApp Manager
TEMPLATE_TEXT = "digest_text"
TEMPLATE_IMAGE = "digest_image"
TEMPLATE_LANG = "en_US"


# ---------- Nightscout ----------
def get_nightscout_data(endpoint, params=None):
    url = f"{NIGHTSCOUT_URL}/api/v1/{endpoint}"
    all_params = {"token": NS_ACCESS_TOKEN}
    if params:
        all_params.update(params)
    response = requests.get(url, params=all_params, timeout=30)
    print(f"{endpoint}: HTTP {response.status_code}")
    response.raise_for_status()
    return response.json()


def get_last_24h_entries():
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    since_ms = int(since.timestamp() * 1000)
    params = {
        "find[date][$gte]": since_ms,
        "count": 2000
    }
    return get_nightscout_data("entries.json", params)


def get_last_24h_treatments():
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    since_iso = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    params = {
        "find[created_at][$gte]": since_iso,
        "count": 500
    }
    return get_nightscout_data("treatments.json", params)


# ---------- Format data for the LLM ----------
def build_summary_text(entries, treatments):
    lines = []

    sgvs = [e["sgv"] for e in entries if "sgv" in e]
    lines.append(f"Total glucose readings in last 24h: {len(entries)}")
    if sgvs:
        avg = sum(sgvs) / len(sgvs)
        in_range = sum(1 for v in sgvs if 70 <= v <= 180)
        pct_in_range = (in_range / len(sgvs)) * 100
        lines.append(f"Min: {min(sgvs)} mg/dL, Max: {max(sgvs)} mg/dL, Avg: {avg:.0f} mg/dL")
        lines.append(f"Estimated time in range (70-180 mg/dL): {pct_in_range:.0f}%")

    # Merged, time-ordered timeline of glucose readings + treatments (converted to IST),
    # so the model can see cause and effect (e.g. meal -> rise -> correction).
    timeline_events = []

    for e in entries:
        ts_ms = e.get("date")  # epoch ms, UTC, reliable regardless of device timezone
        if ts_ms:
            dt_ist = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(IST)
            timeline_events.append(
                (dt_ist, f"Glucose: {e.get('sgv', '?')} mg/dL (trend: {e.get('direction', '?')})")
            )

    for t in treatments:
        ts = t.get("created_at")  # ISO 8601 UTC
        if ts:
            dt_ist = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(IST)
            parts = [t.get("eventType", "Treatment")]
            if t.get("insulin"):
                parts.append(f"insulin={t['insulin']}u")
            if t.get("carbs"):
                parts.append(f"carbs={t['carbs']}g")
            if t.get("notes"):
                parts.append(f"notes='{t['notes']}'")
            timeline_events.append((dt_ist, " | ".join(parts)))

    timeline_events.sort(key=lambda x: x[0])

    lines.append(f"\nChronological timeline (oldest to newest, {len(timeline_events)} events, times in IST):")
    for dt_ist, desc in timeline_events:
        lines.append(f"- {dt_ist.strftime('%Y-%m-%d %I:%M %p')}: {desc}")

    return "\n".join(lines)


# ---------- Gemini analysis ----------
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]

SYSTEM_INSTRUCTION = (
    "You are analyzing 24 hours of continuous glucose monitor (CGM) data and insulin/carb "
    "treatment logs for a personal daily WhatsApp digest. The person reading this lives with "
    "diabetes day to day — they don't need a clinical recap of numbers they can already see "
    "in their app. They need help understanding what actually happened and why, in plain, "
    "everyday language.\n\n"
    "Using the chronological timeline provided (glucose readings interleaved with meals/insulin "
    "treatments), analyze cause-and-effect patterns — e.g. which meals or doses led to a spike "
    "or a low, roughly how long it took, and how it was handled. Call out anything actionable "
    "or worth noticing for tomorrow. Do not just restate summary statistics — interpret them. "
    "If nothing notable happened in a period, say so briefly instead of padding.\n\n"
    "This is not medical advice, and you should not suggest specific dose or treatment changes — "
    "just help the person understand their own day more clearly.\n\n"
    "Respond ONLY with a JSON object (no markdown fences, no preamble) matching exactly this shape:\n"
    "{\n"
    '  "overall": "one or two sentence plain-language summary of the whole day",\n'
    '  "mood": "one of: great | good | mixed | rough",\n'
    '  "events": [\n'
    "    {\n"
    '      "time": "e.g. 8:15 AM",\n'
    '      "title": "short label, e.g. Breakfast spike",\n'
    '      "description": "1-2 sentences explaining what happened and why",\n'
    '      "type": "one of: high | low | meal | stable"\n'
    "    }\n"
    "  ],\n"
    '  "watch_for_tomorrow": "one short actionable/observational note, or empty string if nothing notable"\n'
    "}\n"
    "Include 3-6 events, the most notable ones only, in chronological order."
)


def _call_gemini(model, prompt_text, system_instruction=None, user_prefix="Here is the last 24 hours of data:\n\n"):
    if system_instruction is None:
        system_instruction = SYSTEM_INSTRUCTION

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{user_prefix}{prompt_text}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json"
        }
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=90)
        except requests.exceptions.RequestException as e:
            print(f"[{model}] Request error on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                wait_seconds = 10 * attempt
                print(f"Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            return None

        print(f"[{model}] HTTP {response.status_code} (attempt {attempt}/{max_retries})")

        if response.status_code == 200:
            data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError as e:
                print(f"[{model}] Failed to parse JSON response: {e}")
                print("Raw text:", raw_text)
                return None

        if response.status_code in (429, 500, 503) and attempt < max_retries:
            wait_seconds = 10 * attempt
            print(f"Transient error, retrying in {wait_seconds}s...")
            print("Response body:", response.text)
            time.sleep(wait_seconds)
            continue

        print(f"[{model}] error response:", response.text)
        return None

    return None


def analyze_with_gemini(summary_text):
    for model in GEMINI_MODELS:
        print(f"\nTrying model: {model}")
        result = _call_gemini(model, summary_text)
        if result is not None:
            return result
        print(f"Model {model} failed, trying next fallback if available...")

    raise RuntimeError("All Gemini models failed (primary and fallback).")


# ---------- Chart generation ----------
def build_glucose_chart(entries, treatments):
    """Returns PNG image bytes of a 24h glucose trend chart with treatment markers, in IST."""
    points = []
    for e in entries:
        ts_ms = e.get("date")
        sgv = e.get("sgv")
        if ts_ms and sgv:
            dt_ist = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(IST)
            points.append((dt_ist, sgv))
    points.sort(key=lambda x: x[0])

    if not points:
        return None

    times = [p[0] for p in points]
    values = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)

    ax.axhspan(70, 180, color="#d7f0d7", alpha=0.6, zorder=0, label="Target range")
    ax.plot(times, values, color="#2c6fbb", linewidth=1.6, zorder=2)

    for t in treatments:
        created = t.get("created_at")
        if not created:
            continue
        try:
            t_time = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(IST)
        except ValueError:
            continue
        label = None
        if t.get("carbs"):
            label = f"{t.get('carbs')}g carbs"
        elif t.get("insulin"):
            label = f"{t.get('insulin')}u insulin"
        ax.axvline(t_time, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
        if label:
            ax.text(t_time, max(values) + 15, label, rotation=90, fontsize=6, color="#666666", ha="right", va="top")

    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_ylim(bottom=min(30, min(values) - 20), top=max(values) + 40)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate(rotation=45)
    ax.set_title("Last 24 Hours — Glucose Trend (IST)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------- Stats ----------
def compute_stats(entries):
    sgvs = [e["sgv"] for e in entries if "sgv" in e]
    if not sgvs:
        return None
    in_range = sum(1 for v in sgvs if 70 <= v <= 180)
    low = sum(1 for v in sgvs if v < 70)
    high = sum(1 for v in sgvs if v > 180)
    n = len(sgvs)
    return {
        "avg": round(sum(sgvs) / n),
        "min": min(sgvs),
        "max": max(sgvs),
        "pct_in_range": round((in_range / n) * 100),
        "pct_low": round((low / n) * 100),
        "pct_high": round((high / n) * 100),
        "current": sgvs[0] if entries and entries[0].get("sgv") else sgvs[-1],
    }


# ---------- WhatsApp text formatting ----------
EVENT_EMOJI = {"high": "🔺", "low": "🔻", "meal": "🍽️", "stable": "✅"}
MOOD_EMOJI = {"great": "🌟", "good": "🙂", "mixed": "⚖️", "rough": "⚠️"}


def render_whatsapp_message(analysis, stats, date_str):
    mood_emoji = MOOD_EMOJI.get(analysis.get("mood", "good"), "🙂")
    overall = analysis.get("overall", "")
    events = analysis.get("events", [])
    watch_for = analysis.get("watch_for_tomorrow", "")

    lines = ["*📊 Daily Glucose Digest*", f"_{date_str}_", ""]
    lines.append(f"{mood_emoji} {overall}")
    lines.append("")

    if stats:
        lines.append(f"*Average:* {stats['avg']} mg/dL")
        lines.append(f"*Range:* {stats['min']}–{stats['max']} mg/dL")
        lines.append(f"*Time in range:* {stats['pct_in_range']}%  (low {stats['pct_low']}% · high {stats['pct_high']}%)")
        lines.append("")

    if events:
        lines.append("*Today's timeline:*")
        for ev in events:
            emoji = EVENT_EMOJI.get(ev.get("type", "stable"), "•")
            lines.append(f"{emoji} *{ev.get('time', '')} — {ev.get('title', '')}*")
            lines.append(f"{ev.get('description', '')}")
        lines.append("")

    if watch_for:
        lines.append(f"💡 *Worth noticing:* {watch_for}")
        lines.append("")

    lines.append("_Not medical advice — a plain-language recap from your Nightscout data._")

    return "\n".join(lines)


# ---------- Meta WhatsApp Cloud API ----------
def _whatsapp_api_url(path):
    return f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{path}"


def upload_media(image_bytes, filename="chart.png"):
    """Uploads an image directly to WhatsApp's own media endpoint and returns a media ID."""
    url = _whatsapp_api_url(f"{WHATSAPP_PHONE_NUMBER_ID}/media")
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    files = {
        "file": (filename, image_bytes, "image/png"),
    }
    data = {
        "messaging_product": "whatsapp",
        "type": "image/png",
    }
    response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    print(f"WhatsApp media upload: HTTP {response.status_code}")
    if response.status_code >= 400:
        print("Response:", response.text)
    response.raise_for_status()
    media_id = response.json()["id"]
    print(f"Uploaded media, ID: {media_id}")
    return media_id


def _send_template(template_name, components):
    url = _whatsapp_api_url(f"{WHATSAPP_PHONE_NUMBER_ID}/messages")
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": WHATSAPP_TO,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": TEMPLATE_LANG},
            "components": components,
        },
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"WhatsApp send ({template_name}): HTTP {response.status_code}")
    if response.status_code >= 400:
        print("Response:", response.text)
    response.raise_for_status()


def send_whatsapp_text(text):
    """Sends the digest_text template with the full message as its named body variable."""
    components = [
        {
            "type": "body",
            "parameters": [{"type": "text", "parameter_name": "digest_text", "text": text}],
        }
    ]
    _send_template(TEMPLATE_TEXT, components)


def send_whatsapp_image(image_bytes, caption, filename="chart.png"):
    """Sends the digest_image template: an uploaded chart as the header image, with a caption."""
    media_id = upload_media(image_bytes, filename=filename)
    components = [
        {
            "type": "header",
            "parameters": [{"type": "image", "image": {"id": media_id}}],
        },
        {
            "type": "body",
            "parameters": [{"type": "text", "parameter_name": "caption_text", "text": caption}],
        },
    ]
    _send_template(TEMPLATE_IMAGE, components)


# ---------- Daily summary persistence (for the weekly digest) ----------
SUMMARY_DIR = "data/daily_summaries"
SUMMARY_RETENTION_DAYS = 7


def save_daily_summary(date_str_iso, analysis, stats):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    record = {
        "date": date_str_iso,
        "analysis": analysis,
        "stats": stats,
    }
    path = os.path.join(SUMMARY_DIR, f"{date_str_iso}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved daily summary to {path}")


def prune_old_summaries(retention_days=SUMMARY_RETENTION_DAYS):
    if not os.path.isdir(SUMMARY_DIR):
        return
    cutoff = datetime.now(IST).date() - timedelta(days=retention_days)
    removed = []
    for fname in os.listdir(SUMMARY_DIR):
        if not fname.endswith(".json"):
            continue
        date_part = fname[:-5]
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff:
            os.remove(os.path.join(SUMMARY_DIR, fname))
            removed.append(fname)
    if removed:
        print(f"Pruned {len(removed)} old summary file(s): {removed}")


def git_commit_summaries():
    try:
        subprocess.run(["git", "config", "user.name", "nightscout-bot"], check=True)
        subprocess.run(["git", "config", "user.email", "nightscout-bot@users.noreply.github.com"], check=True)
        subprocess.run(["git", "add", SUMMARY_DIR], check=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode == 0:
            print("No changes to commit for daily summaries.")
            return
        subprocess.run(["git", "commit", "-m", "Update daily glucose summary data"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("Committed and pushed daily summary changes.")
    except subprocess.CalledProcessError as e:
        print(f"Git commit/push failed (non-fatal): {e}")


# ---------- Main ----------
def main():
    print("===================================")
    print(" Nightscout Daily Digest")
    print("===================================")

    entries = get_last_24h_entries()
    treatments = get_last_24h_treatments()

    print(f"\nFetched {len(entries)} entries and {len(treatments)} treatments")

    summary_text = build_summary_text(entries, treatments)

    print("\nSending to Gemini for analysis...")
    analysis = analyze_with_gemini(summary_text)
    print("\n--- Analysis (structured) ---")
    print(json.dumps(analysis, indent=2))

    stats = compute_stats(entries)

    today = datetime.now(IST).strftime("%A, %B %d, %Y")
    subject_date = datetime.now(IST).strftime("%Y-%m-%d")

    print("\nBuilding WhatsApp message...")
    message_text = render_whatsapp_message(analysis, stats, today)

    print("\nSending WhatsApp text message...")
    send_whatsapp_text(message_text)

    print("\nGenerating chart...")
    chart_png = build_glucose_chart(entries, treatments)
    if chart_png:
        print("\nSending chart image...")
        send_whatsapp_image(chart_png, caption="📈 Glucose trend (IST)", filename=f"glucose_{subject_date}.png")

    print("\nSaving daily summary for weekly digest...")
    save_daily_summary(subject_date, analysis, stats)
    prune_old_summaries()
    git_commit_summaries()

    print("\n===================================")
    print(" Done")
    print("===================================")


if __name__ == "__main__":
    main()
