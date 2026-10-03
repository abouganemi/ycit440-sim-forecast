# Copied verbatim on 2026-09-28 from ml-capstone-project/.../v2/verification/03_timing_publication_lag.py.
# Archival: it reads CKAN activity pages (act*.json) and a CSV from a session scratch directory
# that no longer exists, so it does not run as-is.

"""Verifier 3 -- publication-lag reconstruction for the 'courant' SIM interventions CSV.

Method
------
1. The CKAN activity stream of the dataset
   (https://donnees.montreal.ca/api/3/action/package_activity_list?id=interventions-service-securite-incendie-montreal)
   records, for every package change, the resource's `last_modified` (UTC) and `size` (bytes).
   Pages were saved as act.json, act_100.json ... act_1200.json in SCRATCH.
2. The current CSV (downloaded 2026-09-28 00:08 UTC, md5 identical to Originals/) is not
   sorted by date, but its bytes can be summed per CREATION_DATE_TIME date. For a past
   version of size S (published in 2026, when the file already started on 2025-01-01),
   the date X whose cumulative byte total (header + all rows dated <= X) is closest to S is
   the latest date contained in that version. Small residuals reflect later edits.
3. For each preparation day P (forecast day D = P+1), the latest version uploaded before
   a given local preparation time gives the latest available date M; lag = D - M.

Run: uv run --with pandas --with numpy python 03_timing_publication_lag.py
"""
import collections
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

SCRATCH = Path("/tmp/claude-1000/-home-pliqui-Projects/d8745f98-14a3-450c-aad1-1c18e8688123/scratchpad/verification_v2/v3")
RES_ID = "71e86320-e35c-4b4c-878a-e52124294355"
TZ = ZoneInfo("America/Montreal")

# ---- upload history -------------------------------------------------------
uploads = {}
for f in sorted(SCRATCH.glob("act*.json")):
    for a in json.loads(f.read_text())["result"]:
        for r in a["data"].get("package", {}).get("resources", []):
            if r.get("id") == RES_ID and r.get("last_modified"):
                uploads[r["last_modified"]] = r["size"]

# ---- cumulative bytes by date in the current file ------------------------
data = (SCRATCH / "current_live.csv").read_bytes()
lines = data.split(b"\n")
by_day = collections.Counter()
for ln in lines[1:]:
    if ln:
        by_day[ln.split(b",")[1][:10].decode()] += len(ln) + 1
days = sorted(by_day)
cum, c = {}, len(lines[0]) + 1
for d in days:
    c += by_day[d]
    cum[d] = c
assert c == len(data)

rows = []
for lm, size in sorted(uploads.items()):
    t_utc = dt.datetime.fromisoformat(lm).replace(tzinfo=dt.timezone.utc)
    t_loc = t_utc.astimezone(TZ)
    if t_loc.date() < dt.date(2026, 1, 2):
        continue  # before 2026 the file also held 2024 rows; size matching not valid
    best = min(days, key=lambda x: abs(cum[x] - size))
    resid = size - cum[best]
    mean_day = by_day[best]
    rows.append((t_loc, size, dt.date.fromisoformat(best), resid, resid / mean_day))

print(f"uploads analysed (2026): {len(rows)}")
rs = sorted(r[4] for r in rows)
print("residual (bytes) as share of that day's bytes: max |r| =", round(max(abs(x) for x in rs), 3),
      "| min", round(rs[0], 3), "| median", round(rs[len(rs) // 2], 4), "| 95th pct |r|",
      round(sorted(abs(x) for x in rs)[int(0.95 * len(rs))], 4))
lagpub = collections.Counter((r[0].date() - r[2]).days for r in rows)
print("publication local date minus latest date contained:", dict(sorted(lagpub.items())))
hours = collections.Counter(r[0].hour for r in rows)
print("local hour of uploads:", dict(sorted(hours.items())))
first_day, last_day = rows[0][0].date(), rows[-1][0].date()


def lag_table(prep_hour):
    """Distribution of D - (latest available date) for forecasts prepared on D-1 at prep_hour local."""
    out = collections.Counter()
    worst = []
    p = first_day + dt.timedelta(days=1)
    end = dt.date(2026, 9, 27)
    while p <= end:
        t = dt.datetime.combine(p, dt.time(prep_hour), TZ)
        avail = [r[2] for r in rows if r[0] <= t]
        if avail:
            m = max(avail)
            lag = (p + dt.timedelta(days=1) - m).days
            out[lag] += 1
            if lag >= 5:
                worst.append((p.isoformat(), m.isoformat(), lag))
        p += dt.timedelta(days=1)
    return out, worst


# save (forecast day D, latest available date M) for a 09:00 preparation on D-1
recs = []
p = first_day + dt.timedelta(days=1)
while p <= dt.date(2026, 9, 27):
    t = dt.datetime.combine(p, dt.time(9), TZ)
    avail = [r[2] for r in rows if r[0] <= t]
    if avail:
        recs.append(f"{p + dt.timedelta(days=1)},{max(avail)}")
    p += dt.timedelta(days=1)
(SCRATCH / "availability_0900.csv").write_text("D,M\n" + "\n".join(recs) + "\n")

for h in (9, 17):
    tab, worst = lag_table(h)
    n = sum(tab.values())
    print(f"\nPrepared on D-1 at {h:02d}:00 local: n={n}")
    for k in sorted(tab):
        print(f"  D-{k}: {tab[k]} ({tab[k] / n:.1%})")
    print("  lag>=5 cases (prep day, latest date, lag):", worst)

# days with no upload (evening run missing)
up_dates = {r[0].date() for r in rows}
missing = []
p = first_day
while p <= dt.date(2026, 9, 26):
    if p not in up_dates:
        missing.append(p.strftime("%a %Y-%m-%d"))
    p += dt.timedelta(days=1)
print(f"\nlocal dates with no upload between {first_day} and 2026-09-26: {len(missing)}")
print(missing)
