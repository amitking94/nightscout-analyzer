"""
Weekly Nightscout Digest (Email)
----------------------------------
Reads the last 7 daily summary JSON files (saved by main.py into data/daily_summaries/),
asks Gemini to synthesize a week-level narrative, and emails a styled report with a
pie chart (time-in-range breakdown) and a bar chart (daily average trend across the week).

Reuses shared building blocks from main.py (Gemini calling, IST timezone, email sending)
rather than duplicating them here.
"""

import os
import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from main import IST, GEMINI_MODELS, _call_gemini, SUMMARY_DIR, send_email

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


# ---------- HTML rendering ----------
MOOD_STYLES = {
    "great": {"emoji": "🌟", "color": "#3f8f5f"},
    "good": {"emoji": "🙂", "color": "#4a9d6f"},
    "mixed": {"emoji": "⚖️", "color": "#c9902e"},
    "rough": {"emoji": "⚠️", "color": "#c94d4d"},
}


def render_weekly_html_email(weekly, week_agg, week_range_str):
    mood = MOOD_STYLES.get(weekly.get("mood", "good"), MOOD_STYLES["good"])
    overall = weekly.get("overall", "")
    best_day = weekly.get("best_day", {})
    toughest_day = weekly.get("toughest_day", {})
    patterns = weekly.get("patterns", [])
    focus = weekly.get("focus_next_week", [])

    stat_items = [
        ("Week Avg", f"{week_agg.get('week_avg', '—')} mg/dL", "#3a7ca5"),
        ("Time in Range", f"{week_agg.get('week_tir', '—')}%", "#3f8f5f"),
    ]
    stat_cells = "".join(
        f"""<td style="padding:14px 10px; text-align:center; background:#f7f9fb; border-radius:10px;">
                <div style="font-size:12px; color:#7a8494; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;">{label}</div>
                <div style="font-size:20px; font-weight:700; color:{color}; margin-top:4px;">{value}</div>
            </td>"""
        for label, value, color in stat_items
    )

    day_cards = f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="8" style="margin: 16px 0;">
        <tr>
            <td style="padding:14px 16px; background:#eef8f0; border-radius:10px; border-left:4px solid #3f8f5f; width:50%;">
                <div style="font-size:12px; font-weight:700; color:#3f8f5f; text-transform:uppercase;">🏆 Best Day</div>
                <div style="font-size:13px; color:#2a2f36; font-weight:600; margin-top:4px;">{best_day.get('date', '—')}</div>
                <div style="font-size:12px; color:#4a5261; margin-top:2px;">{best_day.get('reason', '')}</div>
            </td>
            <td style="padding:14px 16px; background:#fdf1ec; border-radius:10px; border-left:4px solid #e0693e; width:50%;">
                <div style="font-size:12px; font-weight:700; color:#e0693e; text-transform:uppercase;">🎯 Toughest Day</div>
                <div style="font-size:13px; color:#2a2f36; font-weight:600; margin-top:4px;">{toughest_day.get('date', '—')}</div>
                <div style="font-size:12px; color:#4a5261; margin-top:2px;">{toughest_day.get('reason', '')}</div>
            </td>
        </tr>
    </table>
    """

    pattern_cards = ""
    for p in patterns:
        pattern_cards += f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
               style="margin-bottom:10px; background:#eef5fa; border-radius:10px; border-left:4px solid #3a7ca5;">
            <tr>
                <td style="padding:12px 16px;">
                    <span style="font-weight:700; color:#2a2f36; font-size:14px;">🔁 {p.get('title', '')}</span>
                    <div style="color:#4a5261; font-size:13px; margin-top:4px; line-height:1.5;">{p.get('description', '')}</div>
                </td>
            </tr>
        </table>
        """

    focus_block = ""
    if focus:
        focus_items = "".join(f"<li style='margin-bottom:6px;'>{f}</li>" for f in focus)
        focus_block = f"""
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
               style="margin-top:18px; background:#fff8e6; border-radius:10px; border-left:4px solid #e0a63e;">
            <tr>
                <td style="padding:12px 16px;">
                    <span style="font-weight:700; color:#8a6516; font-size:13px;">🎯 FOCUS FOR NEXT WEEK</span>
                    <ul style="color:#6b5426; font-size:13px; margin:8px 0 0 0; padding-left:18px; line-height:1.5;">
                        {focus_items}
                    </ul>
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
                            <td style="background: linear-gradient(135deg, #6a5acd, #3a7ca5); padding: 24px 28px;">
                                <div style="color:#ffffff; font-size:20px; font-weight:700;">🗓️ Weekly Glucose Digest</div>
                                <div style="color:#e0e8f5; font-size:13px; margin-top:2px;">{week_range_str}</div>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 24px 28px 8px 28px;">
                                <div style="font-size:15px; color:#2a2f36; line-height:1.6;">
                                    <span style="font-size:18px;">{mood['emoji']}</span>
                                    <span style="font-weight:600;">{overall}</span>
                                </div>
                                <table role="presentation" width="100%" cellpadding="0" cellspacing="8" style="margin: 20px 0;">
                                    <tr>{stat_cells}</tr>
                                </table>
                                {day_cards}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 28px 8px 28px;">
                                <div style="font-size:12px; font-weight:700; color:#8a93a3; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:10px;">Patterns This Week</div>
                                {pattern_cards}
                                {focus_block}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 28px 8px 28px;">
                                <img src="cid:pie_chart" style="width:100%; border-radius:10px;">
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 28px 24px 28px;">
                                <img src="cid:trend_chart" style="width:100%; border-radius:10px;">
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 16px 28px; background:#f7f9fb; border-top:1px solid #eceff3;">
                                <div style="font-size:11px; color:#a3aab6; line-height:1.5;">
                                    Not medical advice — a plain-language recap synthesized from your last 7 daily digests.
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

    subject = f"🗓️ Nightscout Weekly Digest — {start_date} to {end_date}"

    print("\nGenerating pie chart...")
    pie_png = build_pie_chart(week_agg.get("week_low"), week_agg.get("week_tir"), week_agg.get("week_high"))

    print("\nGenerating daily trend chart...")
    trend_png = build_daily_trend_chart(week_agg.get("daily_points"))

    print("\nBuilding HTML email...")
    html_body = render_weekly_html_email(weekly, week_agg, week_range_str)

    print("\nSending email...")
    images = {}
    if pie_png:
        images["pie_chart"] = pie_png
    if trend_png:
        images["trend_chart"] = trend_png
    send_email(subject, html_body, images=images if images else None)

    print("\n===================================")
    print(" Done")
    print("===================================")


if __name__ == "__main__":
    main()
