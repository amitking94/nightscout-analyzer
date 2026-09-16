# Nightscout Daily & Weekly Digest

Automated glucose data digests, powered by AI analysis (Google Gemini) and delivered by email.

- **Daily digest** — every day, pulls your last 24 hours of Nightscout data, has Gemini analyze
  patterns (meals, spikes, lows, corrections), generates a glucose trend chart, and emails you a
  styled report.
- **Weekly digest** — every Sunday, reads the last 7 days of saved daily summaries, has Gemini
  synthesize week-level patterns (best/toughest day, recurring issues, focus areas), and emails a
  report with a time-in-range pie chart and a daily-average trend chart.

Both run automatically via GitHub Actions — no server of your own required.

---

## How it works

```
your-repo/
├── main.py                          # Daily digest script
├── weekly.py                        # Weekly digest script (imports shared code from main.py)
├── data/
│   └── daily_summaries/             # Auto-created; stores each day's analysis as JSON
│                                     # (auto-pruned to the last 7 days)
└── .github/
    └── workflows/
        ├── nightscout-digest.yml    # Runs main.py daily
        └── nightscout-weekly.yml    # Runs weekly.py every Sunday
```

Each day, `main.py` runs, sends that day's analysis to your email, and also saves a small JSON
summary of the day into `data/daily_summaries/`, committing it back to the repo. On Sunday,
`weekly.py` reads the last 7 of those saved summaries and asks Gemini to find patterns across
the whole week — it does **not** re-fetch or re-analyze raw glucose data, just the daily
summaries already generated.

---

## Prerequisites

You'll need accounts/credentials from four places. None of them cost anything at the volumes
this project uses.

1. Your own running **Nightscout** site
2. A **Google AI Studio** account (for the Gemini API)
3. A **Gmail** account to send from (or any SMTP provider)
4. A **GitHub** account to host this repo and run the automation

---

## Step 1 — Get a read-only Nightscout access token

Don't use your Nightscout `API_SECRET` (that grants full admin access — read, write, delete).
Instead, create a scoped, read-only token:

1. Open your Nightscout site → **Admin Tools**.
2. Under **Subjects**, click **Add new Subject**.
3. Give it a name (e.g. `digest-bot`).
4. Under **Roles**, select `readable` — this maps to the permission `*:*:read`, which can only
   read data, never write or delete anything.
5. Save. Click the generated **Access Token** link — the token string is embedded in that URL
   (e.g. `digest-bot-a1b2c3d4e5f6...`). Copy the **entire string**, including the name prefix
   before the dash.

You'll use this as `NIGHTSCOUT_ACCESS_TOKEN` below.

---

## Step 2 — Get a free Google Gemini API key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Sign in with a Google account.
3. Click **Create API key**, select or create a Google Cloud project when prompted (no billing
   required for the free tier).
4. Copy the generated key.

You'll use this as `GEMINI_API_KEY` below. The free tier comfortably covers one digest a day
plus an occasional weekly run — no cost.

---

## Step 3 — Get a Gmail app password

Gmail blocks regular password login for scripts, so you need an **app password** instead.

1. Go to [myaccount.google.com/security](https://myaccount.google.com/security) and make sure
   **2-Step Verification** is turned on (required before app passwords are available).
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).
3. Name it something like `nightscout-digest` → **Create**.
4. Copy the 16-character password shown (remove the spaces so it's one continuous string) —
   you won't be able to see it again after leaving the page.

> **If "App passwords" isn't available on your account:** this usually means 2-Step
> Verification isn't fully enabled, the account is a managed Workspace/school account with app
> passwords disabled by an admin, or Advanced Protection is enabled on the account. In any of
> these cases, use a separate personal Gmail account for sending instead, or switch to another
> SMTP provider (e.g. Outlook, or a transactional email service like Brevo).

You'll use your Gmail address as `SMTP_USER` and this app password as `SMTP_PASSWORD` below.

---

## Step 4 — Fork this repo

1. Fork this repository to your own GitHub account.
2. Clone it or edit directly on GitHub — either works.

---

## Step 5 — Add your secrets

In your forked repo: **Settings → Secrets and variables → Actions → New repository secret**.
Add each of the following:

| Secret name | Value | Example |
|---|---|---|
| `NIGHTSCOUT_URL` | Your Nightscout site URL (no trailing slash) | `https://your-site.herokuapp.com` |
| `NIGHTSCOUT_ACCESS_TOKEN` | The read-only token from Step 1 | `digest-bot-a1b2c3d4e5f6...` |
| `GEMINI_API_KEY` | The API key from Step 2 | `AIzaSy...` |
| `SMTP_USER` | The Gmail address sending the digest | `you@gmail.com` |
| `SMTP_PASSWORD` | The 16-character app password from Step 3 | `abcdabcdabcdabcd` |
| `EMAIL_TO` | Where digests should be sent — **comma-separated** for multiple recipients | `you@gmail.com,partner@example.com` |

Optional (only needed if you're not using Gmail):

| Secret name | Default if not set |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |

---

## Step 6 — Enable Actions and test

1. Go to the **Actions** tab in your forked repo. If prompted, click to enable workflows for
   this repo (GitHub disables Actions by default on forks).
2. You should see two workflows listed: **Nightscout Daily Digest** and
   **Nightscout Weekly Digest**.
3. Click into **Nightscout Daily Digest** → **Run workflow** → confirm. This triggers it
   immediately instead of waiting for the schedule.
4. Watch the run logs. If it succeeds, check your inbox for the email.
5. The weekly digest needs **at least 2 saved daily summaries** to have anything meaningful to
   synthesize — it will skip itself with a log message if there isn't enough data yet. Run the
   daily workflow on 2 different days (or twice, a day apart), then manually trigger
   **Nightscout Weekly Digest** the same way to test it.

Once both run successfully, no further action is needed — they'll run automatically going
forward.

---

## Schedule

By default:

- **Daily digest** runs at **7:00 AM IST** (`1:30 UTC`)
- **Weekly digest** runs every **Sunday at 8:00 AM IST** (`2:30 UTC`)

To change the time, edit the `cron` line in `.github/workflows/nightscout-digest.yml` or
`nightscout-weekly.yml`. Cron schedules are always in UTC — convert your desired local time to
UTC first (e.g. via [crontab.guru](https://crontab.guru)), keeping in mind IST is UTC+5:30.

Example: for the daily job to run at 9:00 AM IST instead, that's 3:30 UTC:
```yaml
- cron: "30 3 * * *"
```

> **Note:** GitHub Actions' free scheduler doesn't guarantee exact timing — during periods of
> high load, runs can be delayed by minutes to (rarely) hours. This is a platform-level
> limitation, not a bug in this project. For most personal use this doesn't matter; if you need
> tighter timing, you can trigger the workflow externally via GitHub's REST API
> (`workflow_dispatch`) using a service like [cron-job.org](https://cron-job.org) instead of
> relying on the built-in `schedule` trigger.

---

## Customization

- **Timezone**: hardcoded to IST (`UTC+5:30`) throughout `main.py` and `weekly.py` via the
  `IST` constant. Change this value if you're not in India.
- **Target glucose range**: currently 70–180 mg/dL, used for time-in-range calculations and
  chart shading. Search for `70` and `180` in `main.py` to adjust.
- **Gemini model fallback order**: `GEMINI_MODELS` in `main.py` lists models to try in order,
  falling back automatically if one is overloaded or deprecated. Update this list if Google
  changes their model lineup.
- **Data retention**: daily summaries are kept for 7 days (`SUMMARY_RETENTION_DAYS` in
  `main.py`), auto-pruned each day. Increase this if you want the weekly digest to look further
  back, or if you plan to build a monthly digest later.

---

## Troubleshooting

- **Gemini returns 503 "high demand"**: the script already retries automatically (with backoff)
  and falls back to alternate models. If it still fails, Google's free tier is likely
  experiencing a broader outage — check back later.
- **No email arrives, but the Action shows success**: check spam/junk folders first. Then
  double-check `EMAIL_TO` is spelled correctly and `SMTP_PASSWORD` is the app password, not your
  regular Gmail login password.
- **Weekly digest logs "Not enough daily summaries saved yet"**: this is expected until the
  daily job has successfully run and saved data on at least 2 different days.
- **Workflow doesn't appear in the Actions tab at all**: confirm the `.yml` files are inside
  `.github/workflows/` exactly (not just `.github/` or the repo root), and on your repo's
  default branch.
