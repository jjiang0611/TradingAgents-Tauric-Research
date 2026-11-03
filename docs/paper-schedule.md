# Scheduling the daily paper-trading run

The command to schedule is the same on every system:

```bash
tradingagents paper run main --trading-days-only
```

- It uses **New York's date**, and exits 0 doing nothing when the US market held no session that day (weekends, holidays). Schedule it every day; it sorts out the calendar itself.
- It is **safe to run several times a day**. A ticker already decided for the date is skipped, so extra triggers only retry what failed. Exit code 1 means at least one ticker failed, which a scheduler will record, and the next trigger retries those tickers.
- Only one run per account happens at a time. A scheduled run that starts while the dashboard's Run is working (or the reverse) is skipped, not doubled.
- Every attempt appends a line to `~/.tradingagents/logs/paper/<name>/runs.log`. If a day is missing, look there first.
- It must start **in the repository folder**, because TradingAgents reads `.env` from the working directory. Every example below sets that.
- Run it **after the 4:00 pm New York close**. The examples use 17:05, 19:05 and 21:05 US Eastern; shift them if your computer is in another zone. A day the computer was off or asleep for all three triggers is a **missed day**. Past days are not backfilled: the news feeds are not archived, so a late analysis of an old date would not see what a live one did.
- Authentication stays on your computer: whichever you use (`tradingagents login chatgpt`, or API keys in `.env`) is read from there. Nothing is sent to a third-party scheduler.

Replace `/path/to/TradingAgents-Tauric-Research` with your clone, and use the full path to the `tradingagents` program (`which tradingagents`, or `.venv/bin/tradingagents`). None of the scheduler files below were tested on a real machine by whoever wrote them; check the first evening's `runs.log`.

## macOS (launchd)

launchd runs a missed job when the Mac wakes, which suits a laptop. Times are local, so convert from Eastern.

`~/Library/LaunchAgents/com.tradingagents.paper.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.tradingagents.paper</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/TradingAgents-Tauric-Research/.venv/bin/tradingagents</string>
    <string>paper</string><string>run</string><string>main</string><string>--trading-days-only</string>
  </array>
  <key>WorkingDirectory</key><string>/path/to/TradingAgents-Tauric-Research</string>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>17</integer><key>Minute</key><integer>5</integer></dict>
    <dict><key>Hour</key><integer>19</integer><key>Minute</key><integer>5</integer></dict>
    <dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>5</integer></dict>
  </array>
  <key>StandardOutPath</key><string>/tmp/tradingagents-paper.log</string>
  <key>StandardErrorPath</key><string>/tmp/tradingagents-paper.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.tradingagents.paper.plist
```

## Linux (systemd user timer)

`Persistent=true` runs a missed trigger when the machine is next on. The timezone suffix means no conversion.

`~/.config/systemd/user/tradingagents-paper.service`:

```ini
[Unit]
Description=TradingAgents paper-trading daily run

[Service]
Type=oneshot
WorkingDirectory=/path/to/TradingAgents-Tauric-Research
ExecStart=/path/to/TradingAgents-Tauric-Research/.venv/bin/tradingagents paper run main --trading-days-only
```

`~/.config/systemd/user/tradingagents-paper.timer`:

```ini
[Unit]
Description=TradingAgents paper-trading, after the US close

[Timer]
OnCalendar=*-*-* 17,19,21:05:00 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now tradingagents-paper.timer
systemctl --user list-timers tradingagents-paper.timer    # confirm the next run
journalctl --user -u tradingagents-paper.service          # output of past runs
```

For it to run while you are logged out: `loginctl enable-linger "$USER"`.

## Linux or macOS (cron)

Cron skips anything that comes due while the machine is off, and `CRON_TZ` works with cronie and most Linux crons but not with macOS's. If yours lacks it, write the times in local time.

```cron
CRON_TZ=America/New_York
5 17,19,21 * * * cd /path/to/TradingAgents-Tauric-Research && .venv/bin/tradingagents paper run main --trading-days-only >> "$HOME/tradingagents-paper.log" 2>&1
```

## Windows (Task Scheduler)

From a Command Prompt. This runs at 17:05 local time and repeats every two hours for four and a half hours; convert 17:05 Eastern to your local time first. Then, in the task's Settings tab, tick "Run task as soon as possible after a scheduled start is missed".

```bat
schtasks /Create /TN "TradingAgents Paper" /SC DAILY /ST 17:05 /RI 120 /DU 04:30 ^
  /TR "cmd /c cd /d C:\path\to\TradingAgents-Tauric-Research && .venv\Scripts\tradingagents.exe paper run main --trading-days-only >> %USERPROFILE%\tradingagents-paper.log 2>&1"
```

## Checking that it works

Run the command once by hand first, from the same folder the scheduler uses, outside market hours if you like:

```bash
cd /path/to/TradingAgents-Tauric-Research
tradingagents paper init main NVDA,AAPL,MSFT --cash 10000    # once
tradingagents paper run main --trading-days-only
```

On a weekend it should print that there was no US market session and exit 0. On a trading evening it runs the full analysis. Then open `tradingagents paper dashboard main`: on a weekday its banner names any ticker with no decision for today's date (it does not know market holidays, and uses your computer's date, not New York's).
