Below is a practical, end-to-end recipe for replacing the “band-then-score” Pandas trick with an honest-to-goodness k-d tree.
The idea is: build one tree per key/bucket, work in a two-dimensional space
(time axis, quantity axis), and query each tree for the single nearest neighbour that is also inside your ±12 h window.

⸻

0  Imports

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree          # cKDTree ≈ C-accelerated KDTree

cKDTree and KDTree share the same API; cKDTree is ~10× faster but otherwise identical  ￼.

⸻

1  Pre-processing

### 1a – normalise your key and bucket columns
for df in (data1, data2):
    df['bucket'] = pd.cut(np.log1p(df['quantity']), bins=bins, labels=False)
    df['grp']    = (
        df['str1'].astype(str) + '|' +
        df['str2'].astype(str) + '|' +
        df['bucket'].astype(str)
    )

### 1b – map timestamps to seconds since epoch
for df in (data1, data2):
    df['t_sec'] = df['exec_ts'].astype('int64') // 10**9      # ns → s

Why the epoch‐seconds column? KD-trees need numbers; timestamps in datetime64 can’t be queried directly.

⸻

2  Choose a distance scale

You care about two axes:

axis	physical tolerance	scale so that “tolerance” ≈ 1
time	±12 h	time_scale = 1 / (12*3600)
quantity	the width of one bucket (call it dq)	qty_scale  = 1 / dq

time_scale = 1 / (12*3600)       # sec → “units of 12 h”
dq         = np.diff(bins)[0]    # bucket width on the *log1p* scale
qty_scale  = 1 / dq

After scaling, a Euclidean distance of ≤ 1 means
“within ±12 h and within one bucket”.

⸻

3  Build one tree per group

trees = {}              # grp  →  (cKDTree, frame_of_points)

for grp, sub in data2.groupby('grp'):
    pts = np.vstack([
        sub['t_sec'].values * time_scale,
        np.log1p(sub['quantity'].values) * qty_scale
    ]).T
    trees[grp] = (cKDTree(pts, leafsize=32), sub.reset_index(drop=True))

Memory footprint: each point is only two floats; even ten million rows fit in a few hundred MB.

⸻

4  Query

matches   = []
inf       = np.inf        # for readability
ubound    = 1.0           # search radius in the scaled space

for idx, row in data1.iterrows():
    grp = row['grp']
    if grp not in trees:              # no candidate rows
        matches.append(None)
        continue

    tree, ref = trees[grp]

    q_point = np.array([
        row['t_sec']                 * time_scale,
        np.log1p(row['quantity'])    * qty_scale
    ])

    dist, ii = tree.query(q_point,
                          k=1,
                          distance_upper_bound=ubound)   #  [oai_citation:1‡SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html)

    if np.isinf(dist):                 # > ubound → “no neighbour in window”
        matches.append(None)
    else:
        matches.append(ref.iloc[ii])

distance_upper_bound keeps the search inside the ±12 h rectangle; if no point lies within that radius SciPy returns distance = ∞ and index n  ￼.

⸻

5  Collect the result

data1['match'] = matches
good  = data1['match'].notna()
data1.loc[good, ['match_exec_ts', 'match_quantity']] = (
    pd.DataFrame(data1.loc[good, 'match'].tolist())
        [['exec_ts', 'quantity']]
        .values
)

data1 now has the execution-timestamp-nearest and quantity-nearest row from data2.
One pass through the tree per row → O(N log M) where M = rows in data2 / groups.

⸻

6  When is KD-tree worth it?

size of data2	fastest practical method
≤ 50 000	simple cross-join + .idxmin()
50 k – 5 M	KD-tree (above) or “band-then-score”
millions & fast API	KD-tree or a vector DB / ANN index

If you need absolute speed, swap cKDTree for an approximate-NN library (FAISS, Annoy, ScaNN) and treat the tolerance as a post-filter.

⸻

Summary
	1.	Encode time and quantity as a two-column numeric vector.
	2.	Scale both axes so that your tolerance equals 1.
	3.	Build one cKDTree per (str1, str2, bucket) group.
	4.	query(point, k=1, distance_upper_bound=1) instantly yields the record that is simultaneously

	•	within ±12 h and
	•	the closest quantity value.

Because the tree stores only numeric arrays, everything stays in pure NumPy/SciPy—no Pandas shuffle after construction, and sub-second queries even for multi-million-row reference tables.