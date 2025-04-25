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

Here’s how you can add a QQ-plot in Plotly that both:
	1.	spans the full negative–positive range of your data, and
	2.	emphasizes the sharp peak at zero and the long positive tail

You’ll do this in three steps:

⸻

1. Compute the theoretical vs. sample quantiles

Use SciPy’s probplot (or Statsmodels’) to get the “ordered sample quantiles” (osr) and the matching “theoretical quantiles” (osm) from a Normal distribution:

import numpy as np
import scipy.stats as stats

# drop any NaNs
data = df['x'].dropna().values

# get theoretical vs. sample quantiles
osm, osr = stats.probplot(data, dist='norm')[:2]

By default this assumes you’re comparing to a Normal—if you wanted another distribution (e.g. an exponential with a shift), you could pass dist='expon' plus fit=True, but for diagnosing heavy-tails/peakiness vs. a Normal, dist='norm' is most common.

⸻

2. (Option A) Plot raw QQ with axis limits

If you just want the raw QQ but don’t want a couple of extreme points driving your axes, compute sensible limits—e.g. clip to your 1st/99th percentiles of both sets of quantiles—and then pass them to update_xaxes/update_yaxes:

import plotly.graph_objects as go

# determine clip limits from combined quantiles
all_q = np.concatenate([osm, osr])
low, high = np.percentile(all_q, [1, 99])

fig = go.Figure()

# scatter of quantiles
fig.add_trace(go.Scatter(
    x=osm, y=osr,
    mode='markers',
    name='QQ points'
))

# 45° reference line
fig.add_trace(go.Scatter(
    x=[low, high], y=[low, high],
    mode='lines',
    name='y = x', line=dict(dash='dash')
))

fig.update_layout(
    title="QQ-Plot vs Normal",
    xaxis_title="Theoretical Quantiles",
    yaxis_title="Sample Quantiles",
)
# clip both axes so you still “see” the full bulk around zero
fig.update_xaxes(range=[low, high])
fig.update_yaxes(range=[low, high])

fig.show()

This will give you a QQ-plot where negative quantiles (below zero) and the big positive tail are both visible, but the extreme 1% on each side won’t pull your axes out of view.

⸻

2. (Option B) QQ on a signed-log scale

If you’d rather transform so that each order of magnitude is “evenly spaced” (and negatives stay negative), apply a signed log transform to both sets of quantiles before plotting:

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

x_t = signed_log(osm)
y_t = signed_log(osr)

import plotly.graph_objects as go
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_t, y=y_t,
    mode='markers',
    name='Transformed QQ'
))
# 45° line in transformed space
lims = [min(x_t.min(), y_t.min()), max(x_t.max(), y_t.max())]
fig.add_trace(go.Scatter(
    x=lims, y=lims,
    mode='lines',
    name='y = x',
    line=dict(dash='dash')
))

fig.update_layout(
    title="QQ-Plot on signed-log Scale",
    xaxis_title="Theoretical Quantiles (sign·log1p)",
    yaxis_title="Sample Quantiles (sign·log1p)",
)
fig.show()

This has two benefits:
	•	Your peak around zero stays nicely bunched (log1p stretches small values less).
	•	Your long positive tail is compressed so you can actually see the pattern of deviations from normality.

⸻

3. Putting it into Plotly Express

If you prefer the shorthand of Plotly Express, you can still do the computations yourself and then hand off to px.scatter:

import plotly.express as px

qq_df = (
    pd.DataFrame({
      'theoretical': osm,
      'sample': osr
    })
    # optional: if clipping
    .assign(
      theoretical_clipped=lambda d: d['theoretical'].clip(low, high),
      sample_clipped=lambda d: d['sample'].clip(low, high)
    )
)

# raw QQ with clipping
fig = px.scatter(
    qq_df,
    x='theoretical_clipped',
    y='sample_clipped',
    title="QQ-Plot with 1–99% clipping",
    labels={'theoretical_clipped': 'Theoretical Quantiles',
            'sample_clipped': 'Sample Quantiles'}
)
# add identity line
fig.add_shape(
    type="line",
    x0=low, y0=low, x1=high, y1=high,
    line=dict(dash='dash')
)
fig.show()



⸻

Which to pick?
	•	Option A (axis clipping) is the quickest if you just want to zoom in on those 700 points without cutting any outliers out of the data entirely.
	•	Option B (signed-log transform) is more diagnostic if you suspect power-law or exponential tails: it spaces them out so you can see curvature/deviation from the Normal reference line.

Either approach will let you “see” both


 the sharp central mass at zero,the negative side, and the really long positive tail in one QQ plot. Let me know if you want any tweaks!
Here’s how you can get perfectly symmetric 5-minute bins in Plotly, with 0 as a bin edge (not the center of a bar):

import numpy as np
import plotly.graph_objects as go

# 1) bucket size in seconds
bin_size = 5 * 60  # 300s

# 2) find max absolute delay and round up to a full bin
max_delay = df['delay_s'].abs().max()
max_edge  = int(np.ceil(max_delay / bin_size)) * bin_size

# 3) define your bins so that they start at –max_edge and step by bin_size
xbins = dict(
    start = -max_edge,
    end   =  max_edge,
    size  =  bin_size
)

# 4) build the histogram, turning off autobinning
fig = go.Figure(
    go.Histogram(
        x       = df['delay_s'],
        xbins   = xbins,
        autobinx= False
    )
)

# 5) add a zero‐line and tidy up axes
fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")

fig.update_layout(
    xaxis = dict(
        title     = "Delay (seconds)",
        tickmode  = "array",
        tickvals  = np.arange(-max_edge, max_edge + 1, bin_size),
        ticktext  = [f"{int(t/60)} min" for t in np.arange(-max_edge, max_edge + 1, bin_size)]
    ),
    yaxis = dict(title="Count"),
    bargap=0.1
)

fig.show()

Why this works
	•	xbins.start = -max_edge forces your first bin edge to be exactly -k·300, so that 0 falls exactly on a boundary.
	•	autobinx=False tells Plotly not to “refine” those bins.
	•	You get bins like [-600, -300), [-300, 0), [0, 300), [300, 600) … and zero sits cleanly on the join.

If you’d rather use Plotly Express:

import plotly.express as px

fig = px.histogram(
    df,
    x    = "delay_s",
    xbins= dict(start=-max_edge, end=max_edge, size=bin_size)
)
fig.update_traces(autobinx=False)
# …then same layout tweaks as above…

That will give you a histogram symmetric about zero, with 5-minute (300 s) buckets, and 0 as an edge rather than the center of a bar.