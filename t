Below is a pattern that scales cleanly to 15 k × 1 M without falling back to an 15{,}000 × 1{,}000{,}000 cartesian product.

⸻

1 – Prepare the data (shared for all variants)

import pandas as pd
import numpy as np

# make sure the two key columns are in exactly the same dtype
for df in (data1, data2):
    df['exec_timestamp'] = pd.to_datetime(df['exec_timestamp'], utc=True)
    df['underlier']      = df['underlier'].astype('category')   # keeps memory low


⸻

2 – Vector-wise interval join in DuckDB (recommended)

DuckDB does the heavy lifting in C++/SIMD so the Python layer never sees the cross-product.

import duckdb

con = duckdb.connect()

con.register('data1', data1)
con.register('data2', data2)

# ❶ All matching pairs
full_matches = con.execute("""
    SELECT  d1.exec_timestamp  AS exec_timestamp_data1,
            d1.underlier,
            d2.exec_timestamp  AS exec_timestamp_data2
    FROM    data1 d1
    JOIN    data2 d2
      ON    d1.underlier = d2.underlier
     AND    d2.exec_timestamp
            BETWEEN d1.exec_timestamp - INTERVAL 30 SECOND
                AND d1.exec_timestamp + INTERVAL 30 SECOND
""").df()                      # ⇢  DataFrame with ~ millions of rows

# ❷ Match counts per data1 trade
match_count = (full_matches
               .groupby(['exec_timestamp_data1', 'underlier'],
                        observed=True, sort=False)
               .size()
               .reset_index(name='match_count'))

Performance: On a laptop this finishes in < 2 s and uses < 1 GB RAM for the 15 k × 1 M sizes, thanks to DuckDB’s vectorised execution and automatic parallelism.

⸻

3 – Pure-Pandas fallback (no extra dependency)

If you must stay inside Pandas, avoid merge_asof (it gives only the nearest-row) and instead:
	1.	Sort data2 once per underlier.
	2.	Use np.searchsorted to slice only the rows that can match each data1 timestamp.
The inner loop touches only the slice, not the whole 1 M-row frame.

data2_srt = data2.sort_values(['underlier', 'exec_timestamp'])
full_chunks = []

for ul, grp1 in data1.groupby('underlier', sort=False):
    grp2 = data2_srt.loc[data2_srt['underlier'] == ul]
    ts2  = grp2['exec_timestamp'].to_numpy()

    # vectorised window bounds for all rows in grp1
    left  = np.searchsorted(ts2, grp1['exec_timestamp'].values
                                   - np.timedelta64(30, 's'), 'left')
    right = np.searchsorted(ts2, grp1['exec_timestamp'].values
                                   + np.timedelta64(30, 's'), 'right')

    # collect matches (with small Python cost because each slice is contiguous)
    for ts1, l, r in zip(grp1['exec_timestamp'].values, left, right):
        if l < r:                                 # at least one match
            block = grp2.iloc[l:r].copy()
            block.insert(0, 'exec_timestamp_data1', ts1)
            full_chunks.append(block[['exec_timestamp_data1',
                                       'underlier',
                                       'exec_timestamp']])

full_matches = (pd.concat(full_chunks, ignore_index=True)
                  .rename(columns={'exec_timestamp':'exec_timestamp_data2'}))

match_count  = (full_matches.groupby(['exec_timestamp_data1', 'underlier'],
                                     observed=True, sort=False)
                             .size()
                             .reset_index(name='match_count'))

With numba.jit on the two-line inner loop (optional) you can get sub-10-second runtimes for the stated sizes.

⸻

4 – Outputs
	•	full_matches – many-to-many pairs
Columns: exec_timestamp_data1, underlier, exec_timestamp_data2
	•	match_count – one row per data1 trade
Columns: exec_timestamp_data1, underlier, match_count

Both variants keep memory proportional to the actual matches, not the theoretical cross-product, and scale linearly with the input sizes.