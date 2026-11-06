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

## Pairs accounts

```bash
tradingagents paper pairs run pairs
```

Schedule it the same way: put it in place of `paper run main --trading-days-only` in whichever scheduler file above you use, once a day after the close. It needs no `--trading-days-only`: it handles every completed session since the last run and does nothing when there is none. Because it is deterministic, **a missed day is not lost**: the next run replays it and reaches the decisions a daily run would have made. That is the opposite of the LLM account above, so a laptop that is often asleep suits this mode far better. Both modes share a `runs.log` and never overlap a run on the same account.

## Alpaca paper (optional)

If you mirror a pairs account to Alpaca paper, two more jobs, on trading days only (a closed market makes `send` refuse harmlessly):

| When (New York) | Command | Why |
|---|---|---|
| 15:30 | `tradingagents paper pairs alpaca send pairs --send` | market-on-close orders must reach Alpaca before about 3:45-3:50 pm; the decisions it sends were made at last night's close |
| 17:05 | `tradingagents paper pairs run pairs`, then `tradingagents paper pairs alpaca reconcile pairs` | the simulation fills at today's close, then the real fills are compared with it |

`send` works from the ledger in memory, so a missed evening `run` does not hide a decision. A decision whose session has already passed is skipped as stale rather than sent late. In cron:

```cron
CRON_TZ=America/New_York
30 15 * * 1-5 cd /path/to/TradingAgents-Tauric-Research && .venv/bin/tradingagents paper pairs alpaca send pairs --send >> "$HOME/tradingagents-alpaca.log" 2>&1
5 17 * * 1-5 cd /path/to/TradingAgents-Tauric-Research && .venv/bin/tradingagents paper pairs run pairs && .venv/bin/tradingagents paper pairs alpaca reconcile pairs >> "$HOME/tradingagents-alpaca.log" 2>&1
```

The same two commands go into launchd, systemd or Task Scheduler as above. Keep your laptop awake for the 15:30 job: unlike `run`, a missed `send` cannot be replayed, because the decision goes stale at the next close.

## Checking that it works

Run the command once by hand first, from the same folder the scheduler uses, outside market hours if you like:

```bash
cd /path/to/TradingAgents-Tauric-Research
tradingagents paper init main NVDA,AAPL,MSFT --cash 10000    # once
tradingagents paper run main --trading-days-only
```

On a weekend it should print that there was no US market session and exit 0. On a trading evening it runs the full analysis. Then open `tradingagents paper dashboard main`: on a weekday its banner names any ticker with no decision for today's date (it does not know market holidays, and uses your computer's date, not New York's).
