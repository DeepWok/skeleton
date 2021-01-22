import graphviz

def plot(file_name):
    edge_style = {
        'fontsize': 20,
        'fontname': "times",
    }
    node_style = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': 20,
        'height': 0.5,
        'width': 0.5,
        'penwidth': 2,
        'fontname': "times",
    }

    d = graphviz.Diagraph(
        format='pdf',
        edge_attr=edge_style,
        node_attr=node_style,
        engine='dot'
    )

    inputs = d.node("in", fillcolor='darkgreen2')
    g.render(file_name, view=True)
