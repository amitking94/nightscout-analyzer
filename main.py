import os
import io
import time
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
    # find query pulls entries with date >= since_ms, sorted desc by default
    params = {
        "find[date][$gte]": since_ms,
        "count": 2000  # generous cap; ~288 expected at 5-min intervals
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

    # Build one merged, time-ordered timeline of glucose readings + treatments,
    # so the model can see cause and effect (e.g. meal -> rise -> correction)
    # rather than two disconnected lists.
    timeline_events = []

    for e in entries:
        ts = e.get("dateString")
        if ts:
            timeline_events.append((ts, f"Glucose: {e.get('sgv', '?')} mg/dL (trend: {e.get('direction', '?')})"))

    for t in treatments:
        ts = t.get("created_at")
        if ts:
            parts = [t.get("eventType", "Treatment")]
            if t.get("insulin"):
                parts.append(f"insulin={t['insulin']}u")
            if t.get("carbs"):
                parts.append(f"carbs={t['carbs']}g")
            if t.get("notes"):
                parts.append(f"notes='{t['notes']}'")
            timeline_events.append((ts, " | ".join(parts)))

    timeline_events.sort(key=lambda x: x[0])

    lines.append(f"\nChronological timeline (oldest to newest, {len(timeline_events)} events):")
    for ts, desc in timeline_events:
        lines.append(f"- {ts}: {desc}")

    return "\n".join(lines)


# ---------- Gemini analysis ----------
GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.0-flash"]

SYSTEM_INSTRUCTION = (
    "You are analyzing 24 hours of continuous glucose monitor (CGM) data and insulin/carb "
    "treatment logs for a personal daily email digest. The person reading this lives with "
    "diabetes day to day — they don't need a clinical recap of numbers they can already see "
    "in their app. They need help understanding what actually happened and why, in plain, "
    "everyday language.\n\n"
    "Using the chronological timeline provided (glucose readings interleaved with meals/insulin "
    "treatments), do the following:\n"
    "1. Identify specific cause-and-effect patterns — e.g. which meals or doses led to a spike "
    "or a low, and roughly how long it took and how it was handled.\n"
    "2. Call out anything actionable or worth noticing for tomorrow — for example, a meal that "
    "consistently spikes glucose, a low that happened around a particular time or activity, or "
    "a correction that overshot or undershot.\n"
    "3. Give an overall sense of the day in plain terms (e.g. 'a fairly steady day with one sharp "
    "spike after lunch' rather than just listing min/max/avg).\n"
    "4. Keep it concise — aim for a short, scannable email, not an essay. Use short paragraphs or "
    "a few bullet points, not a wall of stats.\n"
    "5. Do not just restate the summary statistics — interpret them. If nothing notable happened, "
    "say so briefly instead of padding with generic commentary.\n\n"
    "This is not medical advice, and you should not suggest specific dose or treatment changes — "
    "just help the person understand their own day more clearly."
)


def _call_gemini(model, summary_text):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Here is the last 24 hours of data:\n\n{summary_text}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4
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
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        if response.status_code in (429, 500, 503) and attempt < max_retries:
            wait_seconds = 10 * attempt
            print(f"Transient error, retrying in {wait_seconds}s...")
            print("Response body:", response.text)
            time.sleep(wait_seconds)
            continue

        # Non-retryable error, or out of retries for this model
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
    """Returns PNG image bytes of a 24h glucose trend chart with treatment markers."""
    # Parse and sort glucose entries by time
    points = []
    for e in entries:
        ts_ms = e.get("date")  # epoch ms, provided by Nightscout
        sgv = e.get("sgv")
        if ts_ms and sgv:
            points.append((datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc), sgv))
    points.sort(key=lambda x: x[0])

    if not points:
        return None

    times = [p[0] for p in points]
    values = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)

    # Target range shading (70-180 mg/dL)
    ax.axhspan(70, 180, color="#d7f0d7", alpha=0.6, zorder=0, label="Target range")

    # Glucose line
    ax.plot(times, values, color="#2c6fbb", linewidth=1.6, zorder=2)

    # Mark treatments (meals/insulin) as vertical dashed lines
    for t in treatments:
        created = t.get("created_at")
        if not created:
            continue
        try:
            t_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            continue
        label = None
        if t.get("carbs"):
            label = f"{t.get('carbs')}g carbs"
        elif t.get("insulin"):
            label = f"{t.get('insulin')}u insulin"
        ax.axvline(t_time, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
        if label:
            ax.text(t_time, ax.get_ylim()[1] if False else max(values) + 15, label,
                     rotation=90, fontsize=6, color="#666666", ha="right", va="top")

    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_ylim(bottom=min(30, min(values) - 20), top=max(values) + 40)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate(rotation=45)
    ax.set_title("Last 24 Hours — Glucose Trend")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
def send_email(subject, body_text, chart_png=None):
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    if chart_png:
        html_body = (
            f"<html><body style='font-family: sans-serif; white-space: pre-wrap;'>"
            f"{body_text}"
            f"<br><br><img src='cid:glucose_chart' style='max-width:100%;'>"
            f"</body></html>"
        )
        msg.attach(MIMEText(html_body, "html"))
        image = MIMEImage(chart_png, name="glucose_chart.png")
        image.add_header("Content-ID", "<glucose_chart>")
        image.add_header("Content-Disposition", "inline", filename="glucose_chart.png")
        msg.attach(image)
    else:
        msg.attach(MIMEText(body_text, "plain"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [EMAIL_TO], msg.as_string())

    print(f"Email sent to {EMAIL_TO}")


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
    print("\n--- Analysis ---")
    print(analysis)

    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"Nightscout Daily Digest — {today}"
    analysis_html = analysis.replace("\n", "<br>")
    body = f"{analysis_html}<br><br>---<br>Automated daily digest from your Nightscout data collector."

    print("\nGenerating chart...")
    chart_png = build_glucose_chart(entries, treatments)

    print("\nSending email...")
    send_email(subject, body, chart_png=chart_png)

    print("\n===================================")
    print(" Done")
    print("===================================")


if __name__ == "__main__":
    main()
