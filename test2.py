import numpy as np
import plotly.graph_objs as go

# — your raw data —
# 1) a vector of times (in seconds)
times = np.array([...])  # e.g. [12, 305, 1500, …]

# 2) your buckets and their cumulative percentages
buckets = {
    "5min":          0.10,   # 10% ≤ 5 min
    "10min":         0.25,   # 25% ≤ 10 min
    "15min":         0.40,
    "20min":         0.55,
    "30min":         0.75,
    "1hour":         0.90,
    ">1hour":        1.00,
}

# — map bucket labels to the numeric boundary (in seconds) —
bucket_labels    = list(buckets.keys())
bucket_seconds   = [300, 600,  900, 1200, 1800, 3600, 4600]  # last one (">1hour") mapped to your plot max
bucket_values    = [buckets[label] for label in bucket_labels]

# — compute a smooth CDF for your continuous data —
x_cdf = np.linspace(-500, 4600, 1000)
y_cdf = np.searchsorted(np.sort(times), x_cdf) / len(times)

# — compute each bucket’s start/end for horizontal steps —
# assume bucket 0 starts at 0, then [0→300], [300→600], … [3600→4600]
bucket_edges = [0] + bucket_seconds

# — build the figure —
fig = go.Figure()

# 1) bar trace (with widths matching each bucket’s duration)
widths = np.diff(bucket_edges)
# center each bar in its interval:
centers = [(bucket_edges[i] + bucket_edges[i+1]) / 2 for i in range(len(widths))]
fig.add_trace(go.Bar(
    x=centers,
    y=bucket_values,
    width=widths,
    name="Cumulative % by bucket",
    opacity=0.6,
))

# 2) smooth CDF line
fig.add_trace(go.Scatter(
    x=x_cdf,
    y=y_cdf,
    mode="lines",
    name="Empirical CDF",
    line=dict(width=2)
))

# 3) horizontal dashed “step” lines at each bucket value
for start, end, pct in zip(bucket_edges[:-1], bucket_edges[1:], bucket_values):
    fig.add_shape(type="line",
                  x0=start, x1=end, y0=pct, y1=pct,
                  line=dict(dash="dash"),
                  xref="x", yref="y")

# — finalize layout — 
fig.update_layout(
    xaxis=dict(
        title="Time (s)",
        range=[-500, 4600],
        tickmode="array",
        tickvals=bucket_seconds,
        ticktext=bucket_labels
    ),
    yaxis=dict(title="Cumulative fraction"),
    bargap=0.1,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="simple_white"
)

fig.show()