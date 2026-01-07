import plotly.graph_objects as go
from cognitive_vulnerability_diagnosis import diagnose_cognitive_vulnerabilities

def visualize_cog_vuln_radar(narrative_text: str, target_audience_profile: dict):
    """
    Renders a radar (spiderweb) chart showing diagnosed cognitive vulnerabilities.
    - Axes: Top diagnosed vulnerabilities from narrative + audience profile
    - Values: Confidence score (0–1)
    - Title shows dominant vulnerability and count
    """
    # Diagnose cognitive vulnerabilities
    vulnerabilities = diagnose_cognitive_vulnerabilities(
        narrative_text=narrative_text,
        target_audience_profile=target_audience_profile,
        top_k=10,
        min_confidence=0.4
    )

    if not vulnerabilities:
        print("No cognitive vulnerabilities diagnosed — rendering empty radar chart.")
        fig = go.Figure()
        fig.add_annotation(
            text="No significant cognitive vulnerabilities detected in the narrative.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font_size=16,
            align="center"
        )
        fig.update_layout(
            title="Cognitive Vulnerability Radar (No Significant Matches)",
            height=600,
            plot_bgcolor="white"
        )
        fig.write_html("cog_vuln_radar_chart.html")
        fig.show()
        return

    # Build categories and values
    categories = []
    values = []
    dominant_name = "None"
    dominant_conf = 0.0

    for vuln in vulnerabilities:
        name = vuln["name"]
        conf = vuln["confidence"]

        categories.append(f"{name}")
        values.append(conf)

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
        fillcolor='rgba(100, 150, 255, 0.3)',  # Softer blue tone for cognitive focus
        line_color='rgba(50, 100, 255, 1)',
        name='Cognitive Vulnerability Profile'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickfont_size=11,
                tickformat=".2f"
            )
        ),
        showlegend=False,
        title=f"<b>Cognitive Vulnerability Radar</b><br>"
              f"Dominant: {dominant_name} | "
              f"Diagnosed: {len(vulnerabilities)} vulnerabilities",
        height=800,
        margin=dict(t=120, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.write_html("cog_vuln_radar_chart.html")
    print(f"Cognitive vulnerability radar saved with {len(vulnerabilities)} diagnosed vulnerabilities")
    fig.show()