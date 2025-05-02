Below are the most frequent gotchas we see when a “scraped-at” timestamp (your observation_time) ends up earlier than the trade’s own execution_time.

⸻

1.  Time-zone or clock-source mix-ups  (> 80 % of real-world cases)

Symptom	Typical root cause	Quick test
Δ ≈ −4 h or −5 h (Eastern vs UTC)	DTCC publishes many public files in UTC (“Dissemination Timestamp is based on Coordinated Universal Time”). If your scraper logs datetime.now() in ET (EST/EDT) or vice-versa, you’ll always be off by exactly one U.S. time-zone.  ￼	Convert both fields to UTC; the delta should collapse to a few seconds.
Δ flips by +/-1 h at US DST cut-over weeks	One side is DST-aware, the other is naïve.	Plot the delta by calendar date; look for a step on 10 Mar 2024 & 3 Nov 2024 (last DST changes).
Δ follows “odd” offsets (e.g., −09:30, +05:45)	Server’s locale (e.g., IST, ACST) differs from DTCC feed.	Compare observation_time with the OS clock on the host that wrote the file (filesystem mtime).
Δ grows or shrinks gradually	Box running the scraper is not NTP-synced.	Run ntpstat / chronyc tracking on the host or simply query an accurate time API and diff.

DTCC nuance: Some feeds (e.g., Trade Archive) are explicitly EST/EDT — the PDF FAQ states “All timestamps on the Trade Archive UI will be displayed in EST / EDT”   ￼.
Others (e.g., the public GTR slice reports) ship in pure UTC. It’s easy for a black-box scraper to mishandle one of those.

⸻

2.  The scraper may not be using time.time() at all

Black-box collectors sometimes reuse whatever timestamp DTCC includes inside the file/header instead of the machine time at download:

Possible field mistakenly captured	Where it appears	Why it can pre-date execution
Last-Modified HTTP header	S3 / Akamai edge	DTCC republishes hourly snapshots; header shows build time, not your fetch time.
“Generation TimeStamp” / “Snapshot Time” inside CSV	PPD / SFTR sets	Reports are built minutes or hours before you pulled them.
File-path date (.../20250430/ )	Bulk ZIP hierarchy	Path reflects cut-off date, not trade time.

Smoke test: Compare two successive downloads of the same URL an hour apart. If observation_time is identical, you’re not logging wall-clock time at all.

⸻

3.  Parsing / datatype gremlins

Symptom	Explanation	Fix
observation_time rounded to midnight	String "2025-05-02" parsed without HH:MM:SS → defaults to 00:00:00.	Force ISO 8601 with time component.
Timestamps exactly ±24 h, ±48 h, …	Day-level integer math in pandas: int32 overflow or .dt.normalize() called accidentally.	Inspect the ETL script that “cleans” the feed.
Nanoseconds vs milliseconds confusion	execution_time stored as 13-digit epoch, observation_time as 10-digit → appears 1 000× smaller.	Convert both to pd.Timestamp(unit='s') or 'ms' consistently.



⸻

4.  Down-stream data-frame logic can scramble rows
	•	If the pipeline deduplicates by min(observation_time) inside a groupby, the earliest scrape will be copied to all subsequent rows.
	•	A merge where the left table has older observation times will propagate the stale value.
	•	Check whether every record actually needs a row-level observation_time; sometimes it belongs in a batch metadata table instead.

⸻

5.  How to diagnose when you can’t touch the code
	1.	Histogram the deltas — Δ = observation_time – execution_time in seconds.
Constant spikes at −14 400 s or −18 000 s scream “timezone offset”.
	2.	Look at DST boundaries — if the spike changes on 2025-03-09 or 2025-11-02, that nails it.
	3.	Compare to file mtime on disk / S3 object LastModified. If that matches what should be observation_time, you know the scraper is grabbing the wrong field.
	4.	Pull the same file twice a few minutes apart. If the new download still has the old observation time, the bug is in the scraper, not in DTCC’s data.
	5.	Ask for one-off raw logs — Sometimes ops can enable curl -v or proxy logging without exposing the actual scraping code.

⸻

6.  Quick mitigations (if you own the post-scrape stage)
	•	Force everything to UTC early (pd.to_datetime(..., utc=True)), then localise later.
	•	Write the wall-clock fetch time yourself in the downstream pipeline (e.g., when the cleaned CSV is written).
	•	Append a source_time_zone column ('UTC', 'EST', etc.) so analysts know which one they’re looking at.
	•	Alert if observation_time < execution_time – timedelta(minutes=1) — this simple rule-of-thumb catches most anomalies.

⸻

Next step?

If you can share one or two anonymised

 rows (execution vs observation columns


), I can run a quick delta plot and 


confirm whether it’s a pure TZ problem

 or something subtler like stale snapshot

 timestamps. Just let me know!

Daylight Saving Time (DST) — the quick rundown

DST stands for Daylight Saving Time.  It’s the period of the year when a region shifts its clocks one hour forward from its standard (“winter”) time to get more usable daylight in the evening.

Region (common pair)	Standard-time name / UTC offset	DST name / UTC offset	Typical switch-on & switch-off
U.S. East Coast	EST (UTC-5)	EDT (UTC-4)	2nd Sun Mar → 1st Sun Nov
Central Europe (your zone)	CET (UTC+1)	CEST (UTC+2)	Last Sun Mar → Last Sun Oct
U.K. & Ireland	GMT (UTC±0)	BST (UTC+1)	Last Sun Mar → Last Sun Oct

Key idea: during DST the clock reads 12:00 noon when the sun is actually at 11:00 am solar time (roughly). That pushes sunset from, say, 19:00 to 20:00 and gives people an extra hour of light after work.

⸻

Why does DST matter for data work?
	1.	Offset jumps:
	•	Paris is CET (+01:00) in winter, CEST (+02:00) in summer.
	•	New York is EST (-05:00) in winter, EDT (-04:00) in summer.
If you store a naïve local timestamp without the accompanying zone, you lose track of whether that “14:30” was stored while DST was active.
	2.	Ambiguous & missing hours:
	•	When the clock springs forward, the hour 02:00 – 02:59 doesn’t exist.
	•	When it falls back, the hour 01:00 – 01:59 happens twice.
Good time-libraries (e.g., pytz, zoneinfo) keep flags for those states; plain datetime objects do not.
	3.	Delta calculations:
Your earlier question: “observation_time <<< execution_time”.
If one timestamp is in UTC and the other is naïve EST/EDT (or CET/CEST), you’ll see a constant 1-hour or 4/5-hour negative gap that suddenly flips at the DST boundary dates.

⸻

How to stay safe

Practice	How it helps
Store in UTC, display in local time	Internal calculations never hit DST walls.
Always tag with the IANA zone (Europe/Paris, America/New_York)	Zone databases encode every past and future switch.
Watch the change-over dates (2025-03-30 and 2025-10-26 for Paris)	If your deltas step by +1 h / -1 h exactly then, it’s a DST artefact.
Use time-zone aware libraries (pd.to_datetime(..., utc=True).dt.tz_convert(...))	They disambiguate the duplicated “01:30” at fall-back.



⸻

TL;DR

DST is simply the one-hour seasonal shift that many regions adopt to push daylight into their evenings.  In data pipelines it’s the single biggest source of “impossible” time deltas—so converting everything to UTC early and being explicit about time zones saves a world of pain.


Below is a **“tool-kit” of checks you can run with nothing but the fields already in your dataset (plus the wall-clock moment when you pulled the file).  Each idea comes with a short pandas/matplotlib code sketch so you can copy-paste straight into a notebook.

⸻

1.  Get the lay of the land

cols = ['execution_time', 'observation_time']   # adjust to your column names
df[cols] = df[cols].apply(pd.to_datetime, utc=True)

print(df[cols].describe(datetime_is_numeric=True))
delta = (df['observation_time'] - df['execution_time']).dt.total_seconds()
print(delta.describe())

What you learn – ranges, obvious nulls, wild out-of-range values.

⸻

2.  Find the dominant offset (mode)

If every record is off by, say, –14 400 s (-4 h) you instantly know it’s a single-time-zone error.

offset_mode = delta.round().mode().iloc[0]
print(f"Most common Δ = {offset_mode/3600:.1f} h")



⸻

3.  Plot the delta distribution

import matplotlib.pyplot as plt
plt.hist(delta/3600, bins=200)
plt.title("observation_time − execution_time (hours)")
plt.xlabel("Δ hours"); plt.ylabel("count")
plt.show()

Reading the plot
	•	a razor-thin spike ⇒ constant offset (timezone, wrong header, etc.)
	•	a double spike at -5 h and -4 h ⇒ Eastern-time DST flip.
	•	a wide fat blob ⇒ the scraper really does run minutes after each trade.

⸻

4.  Look for Daylight-Saving jumps without external calendars

weekly = delta.groupby(df['execution_time'].dt.to_period('W')).median()/3600
weekly.plot(marker='o')
plt.title("Median Δ by ISO-week"); plt.ylabel("hours"); plt.show()

A clean ±1 h step around late-March / late-October (Europe) or early-March / early-November (US) screams DST.

⸻

5.  Sanity-check the sign of the delta

print("Percentage where observation < execution:",
      (delta < 0).mean() * 100, "%")

Rule of thumb – in real life you expect observation_time ≥ execution_time.
If > 95 % are negative you can be confident the whole column is mis-scaled or mis-zoned rather than single bad rows.

⸻

6.  Detect “batch” timestamps being reused

dup_counts = df.groupby('observation_time').size().sort_values(ascending=False)
print(dup_counts.head())

If one timestamp appears thousands of times it is not the scraper’s wall clock – it’s a build or snapshot time embedded in the file.

⸻

7.  Compare to your own retrieval time

Record the moment you hit the API:

import time, datetime as dt, requests, io
t0 = dt.datetime.utcnow()

data = requests.get(api_url).json()   # or .csv etc.

t1 = dt.datetime.utcnow()
retrieval_time = t0 + (t1 - t0)/2     # crude mid-point of request

df['delta_vs_retrieval'] = (retrieval_time - df['observation_time']).dt.total_seconds()
print(df['delta_vs_retrieval'].describe())

If observation_time is regularly hours older than retrieval_time, the scraper is probably copying a timestamp already inside the payload—not the wall clock.

⸻

8.  Epoch-length check (nanoseconds vs milliseconds)

bad_epoch = df['observation_time'].dropna().astype('int64')
print(bad_epoch.min(), bad_epoch.max())

	•	10-digit values ≈ seconds since 1970
	•	13-digit values ≈ milliseconds
	•	19-digit values ≈ nanoseconds

A mismatch (e.g., execution_time in ms, observation_time in s) produces apparent 1 000× deltas.

⸻

9.  Reconstruct a best-guess corrected timestamp

Once you know the constant offset (offset_mode above):

df['observation_time_fixed'] = df['observation_time'] + pd.Timedelta(seconds=offset_mode*-1)

You can then re-run your analytics with the repaired field.

⸻

10.  Flag the still-illogical records

After the fix, set up a quick rule so they don’t bite you again:

mask = (df['observation_time_fixed'] < df['execution_time'] - pd.Timedelta(minutes=1))
bad_rows = df.loc[mask]
print(f"{mask.mean()*100:.2f}% of rows still violate the 1-minute rule")

Save or log bad_rows for manual follow-up; often they’re genuine data-vendor glitches or time-travelling test trades.

⸻

Putting it together
	1.	Run steps 1–4 → you’ll usually spot a straight timezone or DST issue in < 5 minutes.
	2.	If not constant → use steps 5–8 to decide whether the column is in the wrong unit (seconds vs ms) or being copied from a snapshot header.
	3.	Fix or annotate the column (step 9) and quarantine anomalies (step 10).
	4.	Feed the evidence upstream – even if you can’t see the scraper code, handing ops a one-pager that says “99.6 % are stuck at UTC vs EST” or “timestamp is the HTTP Last-Modified header” lets them zero in on the bug fast.

⸻

Need a notebook template?

If you’d like, drop a tiny sample (just the two timestamp columns) and I can spin the above into a ready-to-run notebook so you only plug in the API call. Let me know!

