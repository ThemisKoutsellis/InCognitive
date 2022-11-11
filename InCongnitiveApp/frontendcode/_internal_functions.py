# _internal_functions.py
import operator
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

# bokeh & holoviews imports
import holoviews as hv
hv.extension('bokeh')

from bokeh.models import (
    Div, ColumnDataSource,
    Button, Spinner,
    CheckboxGroup, Select,
    Panel, Tabs, FactorRange,
    TableColumn, DataTable,
    BoxZoomTool, PanTool,
    ResetTool, HoverTool,
    TapTool, WheelZoomTool,
    SaveTool, Circle,
    MultiLine, Range1d, Band,
)

from backendcode.fcm_layout_parameters import get_nx_graph

__all__ = []

def _display_msg(doc, div, msg, msg_type):

    def _show():
        div.text = str(msg)

        if msg_type=='error':
                div.style= {'font-size': '100%', 'color': 'red'}
        elif msg_type=='alert':
            div.style= {'font-size': '100%', 'color': 'blue'}
        else:
            div.style= {'font-size': '100%', 'color': 'green'}

    doc.add_next_tick_callback(_show)

def _display_lambda(doc, div, mc_lambda):

    def _show_lambda():
            div.text = 'Transfer function: λ = {0}'.format(mc_lambda)
            div.style= {'font-size': '100%', 'color': 'black'}

    doc.add_next_tick_callback(_show_lambda)

def _rearrange_nodes(node_data_df, input_nodes, output_nodes):
    ''' This function rearrange the
        circular layout nodes so that
        the input nodes are aranges
        to the middle-left of the graph
        and the output nodes to the
        middle-right part of the graph
    '''

    # rearrange coordinates
    xs = list(node_data_df.iloc[:,0])
    ys = list(node_data_df.iloc[:,1])
    labels = list(node_data_df.iloc[:,2])

    if xs:
        zipped_node_coord = list(zip(xs, ys))
        sorted_zip =  sorted(zipped_node_coord, key = operator.itemgetter(0))
        x_y_list = list(zip(*sorted_zip))
        xs = list(x_y_list[0])
        ys = list(x_y_list[1])

    # rearrange labels
    intermidiate_nodes = [
        node for node in labels if (node not in input_nodes) and (node not in output_nodes)]
    rearranged_labels = input_nodes + intermidiate_nodes + output_nodes

    # create the output df
    zipped = list(zip(xs, ys, rearranged_labels))
    rearranged_node_data_df = pd.DataFrame(zipped, columns=['x', 'y', 'index'])

    return rearranged_node_data_df

def _ridge(category, data, scale=20):
    return list(zip([category]*len(data), scale*data))

def _plot_results(f, _x, node_mc_values, baseline_node_values):

    source = ColumnDataSource(data=dict())

    if node_mc_values.keys():
        f.y_range.factors = list(node_mc_values.keys())
        f.y_range.range_padding = 0.1

        source = ColumnDataSource(data=dict(x=_x))

        for k in node_mc_values:
            _data = node_mc_values[k]
            if isinstance(_data, list) and len(set(_data))>1:
                kernel = gaussian_kde(_data)
                _pdf = kernel.pdf(_x)
                _max = max(_pdf)
                if _max!=0:
                    _pdf = [0.5*(_/_max) for _ in _pdf]
                    _pdf[0] = 0
                    _pdf[-1] = 0

                    _y = _ridge(k, _pdf,1)
                    source.data[k] = _y
                    f.patch('x', k, source=source, alpha=0.6, line_color=None)
                else:
                    pass

                # plot mean (or baseline scenario)
                _baseline_data = baseline_node_values[k]
                _y = _ridge(k, [0])
                f.circle(_baseline_data, _y, alpha=0.6, line_color="black", size=7, color="red")
            else:
                _y = _ridge(k, [0])
                if isinstance(_data, list):
                    _data = [_data[0]]
                f.circle(_data, _y, alpha=0.6, line_color="black", size=7, color="red")

def _update_graph_renderer(fcm_layout_dict):

    input_nodes = fcm_layout_dict["input_nodes"]
    output_nodes = fcm_layout_dict["output_nodes"]

    node_discription = fcm_layout_dict["node_discription"]
    source_nodes = fcm_layout_dict['source_nodes']
    target_nodes = fcm_layout_dict['target_nodes']
    nodes_order_list = fcm_layout_dict["nodes_order"]

    nx_graph = get_nx_graph(
        fcm_layout_dict['source_nodes'],
        fcm_layout_dict['target_nodes'],
        fcm_layout_dict['weights'],
    )

    initial_hv_graph = hv.Graph.from_networkx(nx_graph, nx.layout.circular_layout)

    # rearrenge show that the Input nodes appear first to the left
    rearranged_node_data_df = _rearrange_nodes(
        initial_hv_graph.nodes.data,
        fcm_layout_dict['input_nodes'],
        fcm_layout_dict['output_nodes'],
    )
    initial_hv_graph.nodes.data = rearranged_node_data_df


    #position of bokeh nodes
    x, y = initial_hv_graph.nodes.array([0, 1]).T
    # labels are in the order of x,y, point
    labels = list(initial_hv_graph.nodes.array([2]).T[0])

    # internal naming of nodes
    node_indices = list(range(0, len(fcm_layout_dict["nodes_order"])))

    # type of node: i) input, ii) intermediate, iii) output

    node_type = []
    for node in labels:
        if node in input_nodes:
            node_type.append('Input node')
        elif node in output_nodes:
            node_type.append('Output node')
        else:
            node_type.append('Intermediate node')

    # get the index lists of source and target nodes
    source_nodes_idx = []
    for el in source_nodes:
        index = labels.index(el)
        source_nodes_idx.append(index)

    target_nodes_idx = []
    for el in target_nodes:
        index = labels.index(el)
        target_nodes_idx.append(index)

    weights = fcm_layout_dict['weights']

    # Renderers
    hv_nodes = hv.Nodes(
        (x, y, node_indices, node_type, labels),
        vdims=['Type', 'Labels'],
    )
    hv_graph = hv.Graph(
        (
            (source_nodes_idx, target_nodes_idx, weights),
            hv_nodes,
            #edgepaths,
        ),
        vdims='Weight'
    )
    # hvgraph options
    hv_graph.opts(
        directed=True,
        node_size=30,
        arrowhead_length=0.03,
        inspection_policy='edges',  # nodes
        selection_policy='nodes',  # edges
        edge_hover_line_color='green',
        node_hover_fill_color='green',
        edge_cmap=plt.cm.Blues,
        node_cmap='Set1',
        cmap='brg',
        edge_color='Weight',
        node_color='Type',
        edge_line_width=2,
    )

    # convert hv labels to bokeh labels renderer object
    hv_labels_renderer = hv.Labels(hv_nodes, ['x', 'y'], 'Labels')
    hv_labels_renderer.opts(bgcolor='black')

    bokeh_labels_fig = hv.render(hv_labels_renderer)
    bokeh_labels_fig_renderers = bokeh_labels_fig.renderers
    bokeh_labels_renderer = bokeh_labels_fig_renderers[0]

    # convert hv to bokeh renderer object
    bokeh_graph = hv.render(hv_graph)
    bokeh_graph_renderer = bokeh_graph.renderers[0]

    return bokeh_graph_renderer, bokeh_labels_renderer

def execute_simulation():
    pass



