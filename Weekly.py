"""
Weekly Nightscout Digest (WhatsApp via Meta Cloud API)
--------------------------------------------------------
Reads the last 7 daily summary JSON files (saved by main.py into data/daily_summaries/),
asks Gemini to synthesize a week-level narrative, and sends a detailed WhatsApp report:
a text message (via the digest_text template) with the analysis, a pie chart (time-in-range
breakdown), and a bar chart (daily average trend across the week) — both via the digest_image
template.

Reuses shared building blocks from main.py rather than duplicating them here.
"""

import os
import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from main import (
    IST,
    GEMINI_MODELS,
    _call_gemini,
    SUMMARY_DIR,
    send_whatsapp_text,
    send_whatsapp_image,
)

WEEKLY_SYSTEM_INSTRUCTION = (
    "You are reviewing a full week of daily glucose summaries (each day was already analyzed "
    "individually) for a person living with diabetes. You are looking across the week for "
    "patterns that don't show up in a single day — recurring trouble spots, improving or "
    "worsening trends, and the best/toughest days.\n\n"
    "Using the 7 daily summaries provided (each has that day's overall summary, notable events, "
    "and stats), write a week-level synthesis. Look for things like: a specific meal or time of "
    "day that repeatedly causes spikes or lows, whether time-in-range is trending up or down "
    "across the week, and which day stood out as best or worst and why. Be specific and "
    "practical — this should genuinely help the person see where they went right and where to "
    "focus next week, not just restate numbers.\n\n"
    "This is not medical advice — do not suggest specific dose or treatment changes, just help "
    "the person see their own week more clearly.\n\n"
    "Respond ONLY with a JSON object (no markdown fences, no preamble) matching exactly this shape:\n"
    "{\n"
    '  "overall": "two or three sentence plain-language summary of the whole week",\n'
    '  "mood": "one of: great | good | mixed | rough",\n'
    '  "best_day": {"date": "YYYY-MM-DD", "reason": "short reason this was the best day"},\n'
    '  "toughest_day": {"date": "YYYY-MM-DD", "reason": "short reason this was the toughest day"},\n'
    '  "patterns": [\n'
    "    {\"title\": \"short label, e.g. Recurring afternoon dips\", "
    "\"description\": \"1-2 sentences on the pattern, when it shows up, and likely cause\"}\n"
    "  ],\n"
    '  "focus_next_week": ["short actionable point 1", "short actionable point 2"]\n'
    "}\n"
    "Include 2-4 patterns and 2-3 focus points, the most notable/actionable only."
)


# ---------- Load and prepare data ----------
def load_recent_daily_summaries(days=7):
    if not os.path.isdir(SUMMARY_DIR):
        return []

    cutoff = datetime.now(IST).date() - timedelta(days=days)
    records = []
    for fname in sorted(os.listdir(SUMMARY_DIR)):
        if not fname.endswith(".json"):
            continue
        date_part = fname[:-5]
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date >= cutoff:
            with open(os.path.join(SUMMARY_DIR, fname)) as f:
                records.append(json.load(f))

    records.sort(key=lambda r: r["date"])
    return records


def build_weekly_prompt_text(daily_records):
    lines = [f"Here are the daily summaries for the last {len(daily_records)} days:\n"]
    for rec in daily_records:
        lines.append(f"--- {rec['date']} ---")
        analysis = rec.get("analysis", {})
        stats = rec.get("stats", {})
        lines.append(f"Overall: {analysis.get('overall', 'n/a')}")
        lines.append(f"Mood: {analysis.get('mood', 'n/a')}")
        if stats:
            lines.append(
                f"Stats: avg={stats.get('avg')} mg/dL, "
                f"range={stats.get('min')}-{stats.get('max')} mg/dL, "
                f"time_in_range={stats.get('pct_in_range')}%, "
                f"low={stats.get('pct_low', '?')}%, high={stats.get('pct_high', '?')}%"
            )
        events = analysis.get("events", [])
        if events:
            lines.append("Notable events:")
            for ev in events:
                lines.append(f"  - {ev.get('time', '')}: {ev.get('title', '')} — {ev.get('description', '')}")
        watch = analysis.get("watch_for_tomorrow", "")
        if watch:
            lines.append(f"Note from that day: {watch}")
        lines.append("")
    return "\n".join(lines)


def analyze_week_with_gemini(prompt_text):
    for model in GEMINI_MODELS:
        print(f"\nTrying model: {model}")
        result = _call_gemini(model, prompt_text, system_instruction=WEEKLY_SYSTEM_INSTRUCTION, user_prefix="")
        if result is not None:
            return result
        print(f"Model {model} failed, trying next fallback if available...")
    raise RuntimeError("All Gemini models failed for weekly analysis (primary and fallback).")


# ---------- Aggregate numeric stats across the week ----------
def compute_week_aggregate(daily_records):
    avgs, tirs, lows, highs = [], [], [], []
    daily_points = []  # (date, avg) for the bar chart
    for rec in daily_records:
        stats = rec.get("stats") or {}
        if stats.get("avg") is not None:
            avgs.append(stats["avg"])
            daily_points.append((rec["date"], stats["avg"]))
        if stats.get("pct_in_range") is not None:
            tirs.append(stats["pct_in_range"])
        if stats.get("pct_low") is not None:
            lows.append(stats["pct_low"])
        if stats.get("pct_high") is not None:
            highs.append(stats["pct_high"])

    def avg_or_none(lst):
        return round(sum(lst) / len(lst)) if lst else None

    return {
        "week_avg": avg_or_none(avgs),
        "week_tir": avg_or_none(tirs),
        "week_low": avg_or_none(lows),
        "week_high": avg_or_none(highs),
        "daily_points": daily_points,
    }


# ---------- Charts ----------
def build_pie_chart(week_low, week_tir, week_high):
    labels, sizes, colors = [], [], []
    if week_low:
        labels.append(f"Low ({week_low}%)")
        sizes.append(week_low)
        colors.append("#c94d4d")
    if week_tir:
        labels.append(f"In range ({week_tir}%)")
        sizes.append(week_tir)
        colors.append("#3f8f5f")
    if week_high:
        labels.append(f"High ({week_high}%)")
        sizes.append(week_high)
        colors.append("#e0693e")

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"fontsize": 10})
    ax.set_title("Time-in-Range Breakdown — This Week", fontsize=12)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_daily_trend_chart(daily_points):
    if not daily_points:
        return None

    dates = [d for d, _ in daily_points]
    avgs = [v for _, v in daily_points]
    short_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%a %d") for d in dates]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    bar_colors = ["#e0693e" if v > 160 or v < 100 else "#3f8f5f" for v in avgs]
    ax.bar(short_labels, avgs, color=bar_colors, width=0.6)
    ax.axhspan(70, 180, color="#d7f0d7", alpha=0.3, zorder=0)
    ax.set_ylabel("Average glucose (mg/dL)")
    ax.set_title("Daily Average Trend — This Week")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------- WhatsApp text formatting ----------
MOOD_EMOJI = {"great": "🌟", "good": "🙂", "mixed": "⚖️", "rough": "⚠️"}


def render_weekly_whatsapp_message(weekly, week_agg, week_range_str):
    mood_emoji = MOOD_EMOJI.get(weekly.get("mood", "good"), "🙂")
    best = weekly.get("best_day", {})
    tough = weekly.get("toughest_day", {})
    patterns = weekly.get("patterns", [])
    focus = weekly.get("focus_next_week", [])

    lines = ["*🗓️ Weekly Glucose Digest*", f"_{week_range_str}_", ""]
    lines.append(f"{mood_emoji} {weekly.get('overall', '')}")
    lines.append("")

    if week_agg.get("week_avg") is not None:
        lines.append(f"*Week average:* {week_agg['week_avg']} mg/dL")
    if week_agg.get("week_tir") is not None:
        lines.append(f"*Time in range:* {week_agg['week_tir']}%")
    lines.append("")

    lines.append(f"🏆 *Best day:* {best.get('date', '—')} — {best.get('reason', '')}")
    lines.append(f"🎯 *Toughest day:* {tough.get('date', '—')} — {tough.get('reason', '')}")
    lines.append("")

    if patterns:
        lines.append("*🔁 Patterns this week:*")
        for p in patterns:
            lines.append(f"• *{p.get('title', '')}* — {p.get('description', '')}")
        lines.append("")

    if focus:
        lines.append("*🎯 Focus for next week:*")
        for f in focus:
            lines.append(f"• {f}")
        lines.append("")

    lines.append("_Not medical advice — synthesized from your last 7 daily digests._")

    return "\n".join(lines)


# ---------- Main ----------
def main():
    print("===================================")
    print(" Nightscout Weekly Digest")
    print("===================================")

    daily_records = load_recent_daily_summaries(days=7)
    print(f"\nFound {len(daily_records)} saved daily summaries in the last 7 days")

    if len(daily_records) < 2:
        print("Not enough daily summaries saved yet to build a meaningful weekly digest. Skipping.")
        return

    prompt_text = build_weekly_prompt_text(daily_records)

    print("\nSending week's data to Gemini for synthesis...")
    weekly = analyze_week_with_gemini(prompt_text)
    print("\n--- Weekly Analysis (structured) ---")
    print(json.dumps(weekly, indent=2))

    week_agg = compute_week_aggregate(daily_records)

    start_date = daily_records[0]["date"]
    end_date = daily_records[-1]["date"]
    week_range_str = f"{start_date} to {end_date}"

    print("\nBuilding WhatsApp message...")
    message_text = render_weekly_whatsapp_message(weekly, week_agg, week_range_str)

    print("\nSending WhatsApp text message...")
    send_whatsapp_text(message_text)

    print("\nGenerating pie chart...")
    pie_png = build_pie_chart(week_agg.get("week_low"), week_agg.get("week_tir"), week_agg.get("week_high"))
    if pie_png:
        send_whatsapp_image(pie_png, caption="🥧 Time-in-range breakdown", filename=f"weekly_pie_{end_date}.png")

    print("\nGenerating daily trend chart...")
    trend_png = build_daily_trend_chart(week_agg.get("daily_points"))
    if trend_png:
        send_whatsapp_image(trend_png, caption="📊 Daily average trend", filename=f"weekly_trend_{end_date}.png")

    print("\n===================================")
    print(" Done")
    print("===================================")


if __name__ == "__main__":
    main()
