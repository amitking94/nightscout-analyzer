import os
import io
import json
import time
import subprocess
import smtplib
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta, timezone

# ---------- Config from environment ----------
NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"].rstrip("/")
NS_ACCESS_TOKEN = os.environ["NIGHTSCOUT_ACCESS_TOKEN"]

IST = timezone(timedelta(hours=5, minutes=30))  # India Standard Time, fixed UTC+5:30

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ["SMTP_USER"]              # the email address sending the digest
SMTP_PASSWORD = os.environ["SMTP_PASSWORD"]      # app password, not your regular login password
EMAIL_TO = os.environ["EMAIL_TO"]                # where the digest should be sent


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
    "treatment logs for a personal daily email digest. The person reading this lives with "
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


# ---------- HTML email rendering ----------
EVENT_STYLES = {
    "high": {"emoji": "🔺", "color": "#e0693e", "bg": "#fdf1ec"},
    "low": {"emoji": "🔻", "color": "#c94d4d", "bg": "#fdeeee"},
    "meal": {"emoji": "🍽️", "color": "#3a7ca5", "bg": "#eef5fa"},
    "stable": {"emoji": "✅", "color": "#3f8f5f", "bg": "#eef8f0"},
}
MOOD_STYLES = {
    "great": {"emoji": "🌟", "color": "#3f8f5f"},
    "good": {"emoji": "🙂", "color": "#4a9d6f"},
    "mixed": {"emoji": "⚖️", "color": "#c9902e"},
    "rough": {"emoji": "⚠️", "color": "#c94d4d"},
}


def render_html_email(analysis, stats, date_str):
    mood = MOOD_STYLES.get(analysis.get("mood", "good"), MOOD_STYLES["good"])
    overall = analysis.get("overall", "")
    events = analysis.get("events", [])
    watch_for = analysis.get("watch_for_tomorrow", "")

    stat_cards = ""
    if stats:
        stat_items = [
            ("Average", f"{stats['avg']} mg/dL", "#3a7ca5"),
            ("Range", f"{stats['min']}–{stats['max']} mg/dL", "#6a5acd"),
            ("Time in range", f"{stats['pct_in_range']}%", "#3f8f5f"),
        ]
        cells = "".join(
            f"""<td style="padding:14px 10px; text-align:center; background:#f7f9fb; border-radius:10px;">
                    <div style="font-size:12px; color:#7a8494; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;">{label}</div>
                    <div style="font-size:20px; font-weight:700; color:{color}; margin-top:4px;">{value}</div>
                </td>"""
            for label, value, color in stat_items
        )
        stat_cards = f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="8" style="margin: 20px 0;">
            <tr>{cells}</tr>
        </table>
        """

    event_cards = ""
    for ev in events:
        style = EVENT_STYLES.get(ev.get("type", "stable"), EVENT_STYLES["stable"])
        event_cards += f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
               style="margin-bottom:10px; background:{style['bg']}; border-radius:10px; border-left:4px solid {style['color']};">
            <tr>
                <td style="padding:12px 16px;">
                    <span style="font-size:15px;">{style['emoji']}</span>
                    <span style="font-weight:700; color:#2a2f36; font-size:14px;">{ev.get('title', '')}</span>
                    <span style="color:#8a93a3; font-size:12px; float:right;">{ev.get('time', '')}</span>
                    <div style="color:#4a5261; font-size:13px; margin-top:4px; line-height:1.5;">{ev.get('description', '')}</div>
                </td>
            </tr>
        </table>
        """

    watch_block = ""
    if watch_for:
        watch_block = f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
               style="margin-top:18px; background:#fff8e6; border-radius:10px; border-left:4px solid #e0a63e;">
            <tr>
                <td style="padding:12px 16px;">
                    <span style="font-weight:700; color:#8a6516; font-size:13px;">💡 WORTH NOTICING</span>
                    <div style="color:#6b5426; font-size:13px; margin-top:4px; line-height:1.5;">{watch_for}</div>
                </td>
            </tr>
        </table>
        """

    return f"""
    <html>
    <body style="margin:0; padding:0; background:#eef1f5; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef1f5; padding: 24px 0;">
            <tr>
                <td align="center">
                    <table role="presentation" width="600" cellpadding="0" cellspacing="0"
                           style="background:#ffffff; border-radius:16px; overflow:hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06);">
                        <tr>
                            <td style="background: linear-gradient(135deg, #3a7ca5, #6a5acd); padding: 24px 28px;">
                                <div style="color:#ffffff; font-size:20px; font-weight:700;">📊 Daily Glucose Digest</div>
                                <div style="color:#e0e8f5; font-size:13px; margin-top:2px;">{date_str}</div>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 24px 28px 8px 28px;">
                                <div style="font-size:15px; color:#2a2f36; line-height:1.6;">
                                    <span style="font-size:18px;">{mood['emoji']}</span>
                                    <span style="font-weight:600;">{overall}</span>
                                </div>
                                {stat_cards}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 28px 8px 28px;">
                                <div style="font-size:12px; font-weight:700; color:#8a93a3; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:10px;">Today's Timeline</div>
                                {event_cards}
                                {watch_block}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 28px 24px 28px;">
                                <img src="cid:glucose_chart" style="width:100%; border-radius:10px; margin-top:12px;">
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 16px 28px; background:#f7f9fb; border-top:1px solid #eceff3;">
                                <div style="font-size:11px; color:#a3aab6; line-height:1.5;">
                                    Not medical advice — a plain-language recap generated from your Nightscout data.
                                </div>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """


# ---------- Email sending ----------
def send_email(subject, html_body, images=None):
    """
    images: optional dict of {content_id: image_bytes}. The HTML body should
    reference each one as <img src="cid:CONTENT_ID">.
    """
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(html_body, "html"))

    if images:
        for cid, image_bytes in images.items():
            image = MIMEImage(image_bytes, name=f"{cid}.png")
            image.add_header("Content-ID", f"<{cid}>")
            image.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
            msg.attach(image)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [EMAIL_TO], msg.as_string())

    print(f"Email sent to {EMAIL_TO}")


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
    """Commit the daily_summaries folder changes back to the repo."""
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
    subject = f"📊 Nightscout Daily Digest — {subject_date}"

    print("\nGenerating chart...")
    chart_png = build_glucose_chart(entries, treatments)

    print("\nBuilding HTML email...")
    html_body = render_html_email(analysis, stats, today)

    print("\nSending email...")
    images = {"glucose_chart": chart_png} if chart_png else None
    send_email(subject, html_body, images=images)

    print("\nSaving daily summary for weekly digest...")
    save_daily_summary(subject_date, analysis, stats)
    prune_old_summaries()
    git_commit_summaries()

    print("\n===================================")
    print(" Done")
    print("===================================")


if __name__ == "__main__":
    main()
