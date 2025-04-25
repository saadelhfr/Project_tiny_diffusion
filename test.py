For just 700 points, you don’t need any fancy big-data tricks—what you need is to tame those extreme values so you can actually see what’s going on around zero. Here are a few patterns you can use in Plotly:

⸻

1. Clip the axis to a percentile range

Show only the –p to +p percentile of your data so outliers don’t blow out the scale:

import numpy as np
import plotly.express as px

# pick, say, 1st and 99th percentiles
low, high = np.percentile(df['x'], [1, 99])

fig = px.histogram(
    df, x="x",
    nbins=50,
    range_x=[low, high],
    title=f"Histogram of x (clipped to 1–99th pctile = [{low:.2f}, {high:.2f}])"
)
fig.update_xaxes(title="Time difference")
fig.show()

You’ll still have all 700 points in the bins, but the axis stops at those cut-offs.

⸻

2. Symmetric log-transform on the data

Since you have both small (positive and negative) and huge values, you can define a “signed log” transform:

import numpy as np

# create a new column that’s sign(x)*log1p(|x|)
df['x_log'] = np.sign(df['x']) * np.log1p(np.abs(df['x']))

fig = px.histogram(
    df, x="x_log",
    nbins=50,
    title="Histogram of signed-log(x): sign(x)*log1p(|x|)"
)
fig.update_xaxes(title="signed-log(Time diff)")
fig.show()

Now every unit on the axis corresponds roughly to an order-of-magnitude step, yet negative values stay negative.

⸻

3. Box & violin with clipped zoom

You can use the same clipping trick in a box or violin to zoom in on the bulk:

fig = px.box(
    df.assign(x_clipped=df['x'].clip(low, high)),
    x="x_clipped",
    points="all",   # or “suspectedoutliers”
    title="Boxplot of x (clipped to 1–99th pctile)"
)
fig.update_xaxes(title="Time difference")
fig.show()

fig = px.violin(
    df.assign(x_clipped=df['x'].clip(low, high)),
    x="x_clipped",
    box=True,
    points="all",
    title="Violin of x (clipped)"
)
fig.update_xaxes(title="Time difference")
fig.show()



⸻

4. Faceted “zoom + full” view

If you want to see both the zoomed-in and full range side by side:

fig = px.histogram(
    df, x="x", nbins=50, facet_col="__zoom__", 
    category_orders={"__zoom__": ["full", "zoom"]},
    facet_col_spacing=0.05
)

# hack: create a helper column marking “zoom” vs “full”
df2 = df.copy()
df2['__zoom__'] = 'full'
zoom_slice = df['x'].between(low, high)
df2_zoom = df[zoom_slice].copy()
df2_zoom['__zoom__'] = 'zoom'
df2 = pd.concat([df2, df2_zoom])

fig = px.histogram(
    df2, x="x", nbins=50, facet_col="__zoom__",
    category_orders={"__zoom__": ["full", "zoom"]},
    facet_col_spacing=0.05,
    title="Full vs Zoomed (1–99th pctile) histogram"
)
fig.for_each_xaxis(lambda ax: ax.update(title="Time difference"))
fig.show()



⸻

When to pick which approach
	•	Clipping (range_x): fastest way to just look at the central  bulk.
	•	Signed-log: best if you want to see how values tail off over orders of magnitude (and keep negatives).
	•	Facet: if you really need both the full-range context and the zoomed-view in one figure.

Any of these will let you explore those ~700 points without one or two extreme outliers stretching your axis to infinity. Let me know if you’d like any of these fleshed out further!