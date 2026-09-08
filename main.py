import os
import smtplib
import requests
from email.mime.text import MIMEText
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
    lines.append(f"Glucose entries (last 24h): {len(entries)} readings")
    if entries:
        sgvs = [e["sgv"] for e in entries if "sgv" in e]
        if sgvs:
            lines.append(f"Min: {min(sgvs)} mg/dL, Max: {max(sgvs)} mg/dL, Avg: {sum(sgvs)/len(sgvs):.0f} mg/dL")
        lines.append("\nRecent readings (most recent first, up to 40 shown):")
        for e in entries[:40]:
            lines.append(f"- {e.get('dateString', '?')}: SGV={e.get('sgv', '?')} dir={e.get('direction', '?')}")

    lines.append(f"\nTreatments (last 24h): {len(treatments)} records")
    for t in treatments:
        lines.append(
            f"- {t.get('created_at', '?')}: {t.get('eventType', '?')} "
            f"insulin={t.get('insulin', '-')} carbs={t.get('carbs', '-')}"
        )

    return "\n".join(lines)


# ---------- Gemini analysis ----------
def analyze_with_gemini(summary_text):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
    )
    system_instruction = (
        "You are a helpful assistant summarizing 24 hours of glucose monitoring data "
        "(from Nightscout/CGM) for a personal daily digest sent over email. "
        "Keep the summary concise (under 800 characters), practical, and easy to read. "
        "Mention overall trend, any highs/lows, and time-in-range if inferable. "
        "This is not medical advice — do not diagnose or prescribe treatment changes."
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Here is the last 24 hours of data:\n\n{summary_text}\n\nPlease summarize it."}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4
        }
    }
    response = requests.post(url, json=payload, timeout=60)
    print(f"Gemini: HTTP {response.status_code}")
    if response.status_code >= 400:
        print("Gemini error response:", response.text)
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ---------- Email ----------
def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

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
    body = f"{analysis}\n\n---\nAutomated daily digest from your Nightscout data collector."

    print("\nSending email...")
    send_email(subject, body)

    print("\n===================================")
    print(" Done")
    print("===================================")


if __name__ == "__main__":
    main()
