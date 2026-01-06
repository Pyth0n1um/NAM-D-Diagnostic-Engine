import plotly.graph_objects as go
from ttp_id import identify_ttp_dual

def visualize_ttp_radar(cvf_result, narrative_text: str):
    """
    Renders a radar (spiderweb) chart using direct TTP identification.
    - Axes: Top identified CAT TTPs from the narrative
    - Values: Confidence score (0–1) from embedding similarity
    - Title shows exact number of identified TTPs
    """
    # Get top identified TTPs with confidence
    identified_ttps = identify_ttp_dual(narrative_text, top_k_per_registry=5)

    if not identified_ttps:
        print("No TTPs identified — rendering empty radar chart.")
        fig = go.Figure()
        fig.add_annotation(
            text="No relevant CAT TTPs identified in the narrative.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font_size=16,
            align="center"
        )
        fig.update_layout(
            title="CAT TTP Identification Radar (No Matches)",
            height=600,
            plot_bgcolor="white"
        )
        fig.write_html("ttp_radar_chart.html")
        fig.show()
        return

    # Build categories and values
    categories = []
    values = []
    dominant_name = "None"
    dominant_conf = 0.0

    for ttp in identified_ttps:
        source = ttp.get("source", "CAT")
        name = ttp["name"]
        ttp_id = ttp["id"]
        conf = ttp["confidence"]

        categories.append(f"{name}<br>({ttp_id})")  # ← This is the line
        values.append(conf)

        # Track dominant
        if conf > dominant_conf:
            dominant_conf = conf
            dominant_name = name

    # Close the loop for radar
    categories += categories[:1]
    values += values[:1]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255,100,100,0.3)',
        line_color='rgba(255,50,50,1)',
        name='Cognitive Attack Profile'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickfont_size=10,
                tickformat=".2f"
            )
        ),
        showlegend=False,
        title=f"CAT TTP Identification Radar<br>"
              f"Dominant: {dominant_name} | "
              f"Identified TTPs: {len(identified_ttps)}",
        height=800,
        margin=dict(t=100)
    )

    fig.write_html("ttp_radar_chart.html")
    print(f"TTP radar chart saved with {len(identified_ttps)} identified TTPs")
    fig.show()
