import plotly.graph_objects as go
import networkx as nx
import numpy as np

def visualize_ttp_overlap_network(cvf_result, threshold=0.6):
    """
    Creates an interactive Plotly network graph of TTP overlaps.
    
    Args:
        cvf_result: The CVFResult object from cvf_model.map_to_cvf
        threshold: Minimum similarity score for drawing an edge (0.0–1.0)
    """
    ttp_cluster = cvf_result.details.get("ttp_cluster", {})
    similarities = ttp_cluster.get("similarities", {})
    activated_ttps = cvf_result.details.get("activated_ttp_ids", [])
    
    if not similarities:
        print("No TTP similarities available — skipping network graph.")
        return

    # Build NetworkX graph
    G = nx.Graph()

    # Add all known TTPs as nodes
    for ttp_id in similarities.keys():
        # Node size based on similarity (or 1 if not activated)
        sim_score = similarities.get(ttp_id, 0.0)
        size = 30 if ttp_id in activated_ttps else 15
        color = "red" if ttp_id in activated_ttps else "lightblue"
        G.add_node(ttp_id, size=size * (sim_score + 0.5), color=color, title=ttp_id)

    # Add edges for strong similarities
    for ttp1, sim_dict in similarities.items():
        for ttp2, score in sim_dict.items():
            if score >= threshold and ttp1 != ttp2:
                # Avoid duplicate edges
                if G.has_edge(ttp1, ttp2) or G.has_edge(ttp2, ttp1):
                    continue
                G.add_edge(ttp1, ttp2, weight=score)

    # Force-directed layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Extract for Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray', opacity=0.6),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        sim = similarities.get(node, 0.0)
        activated = " (ACTIVATED)" if node in activated_ttps else ""
        node_text.append(f"{node}<br>Similarity: {sim:.3f}{activated}")
        node_size.append(G.nodes[node].get('size', 15))
        node_color.append(G.nodes[node].get('color', 'lightblue'))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node.split("-")[-1] for node in G.nodes()],  # Short label
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2
        ),
        hovertext=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="TTP Overlap Network (Cognitive Attack Patterns)",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text=f"Threshold: {threshold} | Activated: {len(activated_ttps)} TTPs",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        plot_bgcolor='white'
                    ))

    # Export to interactive HTML
    fig.write_html("ttp_overlap_network.html")
    print("TTP overlap network graph saved as 'ttp_overlap_network.html'")

    # Show in browser (optional)
    fig.show()