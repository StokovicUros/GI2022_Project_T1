"""

    Used to visualize a network graph.
    
    Function draw_graph creates visualization of a graph with each node coloured according to its probability value
    and the result is stored in a html file named as requested by the third parameter of the function.
     
"""

import plotly.graph_objects as go
import networkx as nx
from src.util.path import get_project_root
import os.path


def draw_graph(graph, prob, name):
    pos = nx.spring_layout(graph)

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#444'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Hot',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Probability',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_probs = []
    node_text = []
    for node, probability in prob.items():
        node_probs.append(probability)
        node_text.append('Probability: ' + str(probability))

    node_trace.marker.color = node_probs
    node_trace.text = node_text

    # noinspection PyTypeChecker
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Node probabilities for graph with {graph.number_of_nodes()} nodes",
                        titlefont_family="Open Sans",
                        titlefont_size=25,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    folder_path = get_project_root() + "/Visualization"
    # Crete folder if it does not exist
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    # Save drawing
    fig.write_html(f'{folder_path}/drawing_{name}.html', auto_open=False)
