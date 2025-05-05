Great — a heatmap using Plotly is a smart, scalable way to display your quantile-delay-time information, especially if you encode delays by color (red for negative, green for positive) and label them clearly.

Here’s a step-by-step guide and code template for how you can:
	1.	Create a 7×16 heatmap (quantiles × time periods) per delay.
	2.	Color-code delay values.
	3.	Add annotated values in seconds and optionally convert to minutes/hours/days.
	4.	Build 3 separate heatmaps (for your 3 delays) from your DataFrame.

⸻

Assumptions about Your DataFrame

Suppose your dataframe (df) is structured like this:

time_period	delay_type	quantile	delay_seconds
T1	D1	0.1	-240
T1	D1	0.5	120
…	…	…	…

You may need to pivot it to get a 7x16 matrix for each delay.

⸻

Helper Function to Convert Seconds

def format_delay(seconds):
    if seconds is None:
        return ""
    abs_sec = abs(seconds)
    if abs_sec >= 86400:
        return f"{seconds/86400:.1f}d"
    elif abs_sec >= 3600:
        return f"{seconds/3600:.1f}h"
    elif abs_sec >= 60:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds:.0f}s"



⸻

Plotly Heatmap Code Example

import pandas as pd
import plotly.graph_objects as go

# Assuming df is your DataFrame and it has: time_period, delay_type, quantile, delay_seconds

# Step 1: Pivot for each delay
def create_heatmap_for_delay(df, delay_type):
    df_filtered = df[df['delay_type'] == delay_type]
    heat_df = df_filtered.pivot(index='quantile', columns='time_period', values='delay_seconds')
    quantiles = sorted(heat_df.index)  # to keep them in order

    # Create annotations
    annotations = []
    for i, q in enumerate(quantiles):
        for j, tp in enumerate(heat_df.columns):
            val = heat_df.loc[q, tp]
            label = format_delay(val)
            annotations.append(dict(
                x=tp,
                y=q,
                text=label,
                showarrow=False,
                font=dict(color="black", size=12)
            ))

    # Color scale: red to green
    colorscale = [
        [0.0, "darkred"],
        [0.5, "white"],
        [1.0, "darkgreen"]
    ]

    # Normalize color scale center at 0
    z_vals = heat_df.values
    max_abs = max(abs(z_vals.min()), abs(z_vals.max()))

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=heat_df.columns,
        y=heat_df.index,
        colorscale=colorscale,
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="Delay (s)")
    ))

    fig.update_layout(
        title=f"Delay Type: {delay_type}",
        annotations=annotations,
        xaxis_title="Time Period",
        yaxis_title="Quantile"
    )

    return fig



⸻

Plot All Three Delays

for delay in ['D1', 'D2', 'D3']:
    fig = create_heatmap_for_delay(df, delay)
    fig.show()



⸻

Tips for Polish
	•	Use plotly.subplots.make_subplots if you want to show all delays in one figure.
	•	Use hovertemplate in the Heatmap to also show raw seconds if you want richer tooltips.
	•	Add tickformat to quantile/time axis if needed for nicer formatting.

⸻

Would you like a version using Plotly Express or with interactive dropdowns for switching delays?