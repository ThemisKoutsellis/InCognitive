# main.py

# general imports

import io
import os
import bisect
import operator
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import networkx as nx
from base64 import b64decode
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

# bokeh & holoviews imports
from holoviews import opts
import holoviews as hv
hv.extension('bokeh')

from bokeh.io import curdoc
from bokeh.layouts import layout, column, row
from bokeh.plotting import figure
from bokeh.models.widgets import FileInput
from bokeh.plotting import from_networkx
from bokeh.models import (Div, ColumnDataSource, Button,
                          Spinner, CheckboxGroup, Select,
                          Panel, Tabs, FactorRange, TableColumn,
                          DataTable, BoxZoomTool, PanTool, ResetTool,
                          HoverTool, TapTool, WheelZoomTool, SaveTool,
                          Circle, MultiLine, Range1d, Band,
                          )


# package imports
from backendcode.xlparse import get_fcm_layout
from backendcode.fcmmc_simulation import monte_carlo_simulation
from backendcode.fcm_layout_parameters import get_nx_graph


PARIS_REINFORCE_COLOR = '#9CAB35'
"""str: The default RGB color of PARIS REINFORCE
"""

# ----------------------------------------------------------------------------------
# Global variables   ---------------------------------------------------------------
# ----------------------------------------------------------------------------------

current_doc = curdoc()

fcm_layout_dict = {
    'source_nodes': [],
    'target_nodes': [],
    'weights': [],
    'nodes_order': [],
    'input_nodes': [],
    'output_nodes': [],
    'node_discription': [],
}

transfer_function = 'sigmoid'

input_iterations = 1
weight_iterations = 1

variance_on_zero_weights = False

sd_inputs = 0.1
sd_weights = 0.1

lamda = None
lamda_autoslect = True

input_nodes_cds = ColumnDataSource()
intermediate_nodes_cds = ColumnDataSource()
output_nodes_cds = ColumnDataSource()

nodes_CSD = ColumnDataSource()
edges_CSD = ColumnDataSource()

output_nodes_mc_values = {}

# ----------------------------------------------------------------------------------
# Supplementary functions   --------------------------------------------------------
# ----------------------------------------------------------------------------------

def _display_msg(msg, msg_type):

    def _show():
        alert_msg_div.text = str(msg)

        if msg_type=='error':
                alert_msg_div.style= {'font-size': '100%', 'color': 'red'}
        elif msg_type=='alert':
            alert_msg_div.style= {'font-size': '100%', 'color': 'blue'}
        else:
            alert_msg_div.style= {'font-size': '100%', 'color': 'green'}

    current_doc.add_next_tick_callback(_show)

def _display_lambda(mc_lambda):

    def _show_lambda():
            lambda_div.text = 'Transfer function: λ = {0}'.format(mc_lambda)
            lambda_div.style= {'font-size': '100%', 'color': 'black'}

    current_doc.add_next_tick_callback(_show_lambda)

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

def rearrange_fcm_layout_dict_lists(fcm_layout_dict, labels):

    node_discription = fcm_layout_dict["node_discription"]
    nodes_order_list = fcm_layout_dict["nodes_order"]
    source_nodes = fcm_layout_dict['source_nodes']
    target_nodes = fcm_layout_dict['target_nodes']
    weights = fcm_layout_dict['weights']

    order_idx = []
    for node in nodes_order_list:
        index = labels.index(node)
        order_idx.append(index)

    node_discription = [node_discription[i] for i in order_idx]
    nodes_order_list = [nodes_order_list[i] for i in order_idx]
    source_nodes = [source_nodes[i] for i in order_idx]
    target_nodes = [target_nodes[i] for i in order_idx]
    weights = [weights[i] for i in order_idx]

    return nodes_order_list, node_discription, source_nodes, target_nodes, weights

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

def _is_there_variation_on_weights(active):
    global sd_weights
    global weight_iterations

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    if active:

        weight_sd_spinner.disabled = False
        weight_iterations_spinner.disabled = False

        weight_iterations = 2
        weight_iterations_spinner.value = 2
        weight_sd_spinner.value = 0.1

        weight_iterations_spinner.low = 2

        variable_zero_weights_radio_button.active = []
        variable_zero_weights_radio_button.disabled = False
        variance_on_zero_weights = False

    else:
        sd_weights = 0
        weight_iterations = 1
        weight_iterations_spinner.value = 1

        weight_sd_spinner.disabled = True
        weight_iterations_spinner.disabled = True

        weight_iterations_spinner.low = 1

        variable_zero_weights_radio_button.active = []
        variable_zero_weights_radio_button.disabled = True
        variance_on_zero_weights = False

    return None

def _lambda_autoselect(active):
    global lamda_autoslect

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    if active:
        lambda_spinner.disabled = True
        lamda_autoslect = True
        lamda = None

    else:
        lamda_autoslect = False
        lambda_spinner.disabled = False
        lambda_spinner.value = 0.5

    return None

def _set_iterations_var_inputs(attr, old, new):
    global input_iterations

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    input_iterations = new

    if new<2:
        inputs_sd_spinner.value = 0
        inputs_sd_spinner.disabled = True
    else:
        inputs_sd_spinner.disabled = False

    return None

def _set_input_sd(attr, old, new):
    global sd_inputs

    current_doc.add_next_tick_callback(
        partial(_display_msg, msg=' ', msg_type='alert'))

    sd_inputs = new

    return None

def _is_there_variation_on_input_nodes(active):
    global sd_inputs
    global input_iterations

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    if active:
        input_iterations_spinner.disabled = False
        input_iterations_spinner.value = 2
        input_iterations_spinner.low = 2
        input_iterations = 2

        inputs_sd_spinner.disabled = False
        inputs_sd_spinner.value = 0.1

    else:
        input_iterations_spinner.disabled = True
        input_iterations_spinner.value = 1
        input_iterations_spinner.low = 1
        input_iterations = 1

        inputs_sd_spinner.disabled = True
        sd_inputs = 0

    return None

def _is_zero_weights_variable(active):
    global variance_on_zero_weights

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    if active:
        variance_on_zero_weights = True
    else:
        variance_on_zero_weights = False

    return None

def _set_lambda(attr, old, new):
    global lamda

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    lamda = new

    return None

def _set_transfer_function(attr, old, new):
    global transfer_function, f1,f2, f3

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    transfer_function = new

    return None

def _clear_msg(attr, old, new):
    alert_msg_div.text = ' '

def _set_iterations_var_weights(attr, old, new):
    global weight_iterations
    global variance_on_zero_weights

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    weight_iterations = new

    if new<2:
        variable_zero_weights_radio_button.active = []
        variable_zero_weights_radio_button.disabled = True
        variance_on_zero_weights = False
        weight_sd_spinner.value = 0
        weight_sd_spinner.disabled = True
    else:
        variable_zero_weights_radio_button.disabled = False
        weight_sd_spinner.disabled = False

    return None

def _set_weights_sd(attr, old, new):
    global sd_weights

    current_doc.add_next_tick_callback(
            partial(_display_msg, msg=' ', msg_type='alert'))

    sd_weights = new

    return None


# ----------------------------------------------------------------------------------
# Fundamental functions   ----------------------------------------------------------
# ----------------------------------------------------------------------------------

def xlsx_parse(attr, old, new):
    global fcm_layout_dict
    global nodes_CSD
    global edges_CSD

    fcm_layout_dict = {}
    nodes_CSD.data = {}
    edges_CSD.data = {}

    f1.renderers = []
    f2.renderers = []
    f3.renderers = []

    raw_data = b64decode(new)
    file_io = io.BytesIO(raw_data)

    # Exceptions:
    try:
        xl_file = pd.ExcelFile(file_io)
        sheet_list = ['nodes-order', 'input-output-nodes', 'fcm-topology']
        # Exception 2: Invalied sheets
        if xl_file.sheet_names != sheet_list:
            fcm_plot.renderers = []
            excel_parse_msg_div.text = 'Wrong number or type of excel sheets'
            excel_parse_msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None
    except:
        # Exception 1: Invalied sheets
        fcm_plot.renderers = []
        excel_parse_msg_div.text = 'Error while reading the excel file'
        excel_parse_msg_div.style= {'font-size': '100%', 'color': 'red'}
        return None

    try:
        df_nodes_order = pd.read_excel(file_io, 'nodes-order', engine='openpyxl')
        df_in_out_nodes = pd.read_excel(file_io, 'input-output-nodes', engine='openpyxl')
        df_fcm_topology = pd.read_excel(file_io, 'fcm-topology', engine='openpyxl')

        sheet1cols = list(df_nodes_order.keys())
        sheet2cols = list(df_in_out_nodes.keys())
        sheet3cols = list(df_fcm_topology.keys())

        expr1 = sheet1cols!=['nodes order', 'node description', 'initial value', 'auto weights', 'auto lags']
        expr2 = sheet2cols!=['input nodes', 'output nodes']
        expr3 = sheet3cols!=['source node', 'target node', 'weight', 'lag']

        # invalid sheet1
        if expr1:
            fcm_plot.renderers = []
            excel_parse_msg_div.text = 'Error in 1st sheet named: "nodes-order"'
            excel_parse_msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None
        # invalid sheet2
        if expr2:
            fcm_plot.renderers = []
            excel_parse_msg_div.text = 'Error in 2nd sheet named: "input-output-nodes"'
            excel_parse_msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None
        # invalid sheet3
        if expr3:
            fcm_plot.renderers = []
            excel_parse_msg_div.text = 'Error in 3rd sheet named: "fcm-topology"'
            excel_parse_msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None

    except:
        # Exception 3: Error while reading the sheets
        fcm_plot.renderers = []
        excel_parse_msg_div.text = 'Error while reading the excel sheets'
        return None

    excel_parse_msg_div.text = ' '
    fcm_layout_dict = get_fcm_layout(df_nodes_order, df_fcm_topology, df_in_out_nodes)

    # node type column
    input_nodes = fcm_layout_dict['input_nodes']
    output_nodes = fcm_layout_dict['output_nodes']
    node_type = ['Input']*len(fcm_layout_dict['nodes_order'])
    for i, v in enumerate(fcm_layout_dict['nodes_order']):
        if v in input_nodes:
            pass
        elif v in output_nodes:
            node_type[i] = 'Output'
        else:
            node_type[i] = 'Intermediate'

    source_nodes_data = {'name': fcm_layout_dict['nodes_order'],
                     'desc': fcm_layout_dict['node_discription'],
                     "type": node_type,
                    }
    source_edges_data = {'source': fcm_layout_dict['source_nodes'],
                        'target': fcm_layout_dict['target_nodes'],
                        'weight': fcm_layout_dict['weights'],
                        }

    nodes_CSD.data = source_nodes_data
    edges_CSD.data = source_edges_data

    (graph_renderer, labels_renderer) = update_graph_renderer(fcm_layout_dict)
    fcm_plot.renderers = []
    fcm_plot.renderers = [graph_renderer, labels_renderer]

    return None

def update_graph_renderer(fcm_layout_dict):

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
    hv_nodes = hv.Nodes((x, y, node_indices, node_type, labels), vdims=['Type', 'Labels'])
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

def collect_global_var():

    global fcm_layout_dict
    global transfer_function
    global input_iterations
    global weight_iterations
    global variance_on_zero_weights
    global sd_inputs
    global sd_weights
    global input_nodes_cds
    global intermediate_nodes_cds
    global output_nodes_cds
    global output_nodes_mc_values
    global f1, f2, f3

    f1.renderers = []
    f2.renderers = []
    f3.renderers = []


    current_doc.add_next_tick_callback(
        partial(_display_msg, msg=' ', msg_type='alert'))

    error1 = not bool(input_xlsx_wgt.filename)
    error2 = not bool(fcm_layout_dict)
    if not error2:
        error3 = not bool(fcm_layout_dict['source_nodes'])
    else:
        error3 = True

    _expr = error1 or error2 or error3

    if _expr:
        current_doc.add_next_tick_callback(
            partial(_display_msg, msg='[Error]: There is no input excel file OR the excel file is erroneous!', msg_type='error'))
    else:
        _expr1 = input_iterations == weight_iterations
        _expr2 = (input_iterations < 2) or (weight_iterations < 2)
        if _expr1 or _expr2:
            # Monte Carlo Simulation
            (
            mc_lambda,
            input_nodes_mc_values,
            output_nodes_mc_values,
            intermediate_nodes_mc_values,
            baseline_input_nodes_values,
            baseline_output_nodes_values,
            baseline_intermediate_nodes_values,
            )=monte_carlo_simulation(
                fcm_layout_dict,
                input_iterations,
                weight_iterations,
                sd_inputs,
                sd_weights,
                variance_on_zero_weights,
                transfer_function,
                lamda,
                lamda_autoslect,
            )


            ##           Plot FIGURES              ##
            #########################################

            def _set_x_range(start, end):
                f1.x_range.start = start
                f1.x_range.end = end

                f2.x_range.start = start
                f2.x_range.end = end

                f3.x_range.start = start
                f3.x_range.end = end

            N = 600

            if transfer_function == 'sigmoid':
                _x = list(np.linspace(0, 1, N))
                bisect.insort(_x, 0.5)
                _set_x_range(0,1)

            elif transfer_function == 'hyperbolic':
                _x = list(np.linspace(-1, 1, N))
                bisect.insort(_x, 0)
                _set_x_range(-1,1)

            # ------------------------
            #   Plot Figure 1,2 & 3
            # ------------------------

            _plot_results(f1, _x, input_nodes_mc_values, baseline_input_nodes_values)
            _plot_results(f2, _x, intermediate_nodes_mc_values, baseline_intermediate_nodes_values)
            _plot_results(f3, _x, output_nodes_mc_values, baseline_output_nodes_values)

            current_doc.add_next_tick_callback(
                partial(_display_lambda, mc_lambda=mc_lambda))

            current_doc.add_next_tick_callback(
                partial(_display_msg, msg='Execution ended successfully.', msg_type='success'))
        else:
            current_doc.add_next_tick_callback(
                partial(_display_msg, msg='[ALERT]: The number of iterations (Weight & Input) must be equal!', msg_type='alert'))
            pass



    return None


# ----------------------------------------------------------------------------------
# Webpage elements   ---------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Header Div:
page_header = Div(
    text=("<figure>"
          "<img src='InCongnitiveApp/static/images/in_cognitive_logo.png' "
          "width='400' height='45' "
          "alt='FCM Simulation application'"
          "</figure>"
          ),
          width=400, height=60,
    )

# Acknowledgements Div:
acknowledgements = Div(
    text=(
        'Powered by: '
        '<a href="https://www.python.org/" target=_blank>Python 3.7.3</a>'
        ', '
        '<a href="https://holoviews.org" target=_blank> HoloViews 1.14.8</a>'
        ' and '
        '<a href="https://docs.bokeh.org/en/latest/index.html" target=_blank>Bokeh 2.3.3.</a>'
        ),
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# License Div:
license = Div(
    text='License:\
        <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target=_blank>(CC BY-NC-ND 4.0)</a>',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Github repository Div:
github_repo = Div(
    text='<a href="https://github.com/ThemisKoutsellis/InCognitive" target=_blank>GitHub repository</a>',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Separator line Div:
def separator(width=550, height=5):
    return Div(text=f'<hr width="{width}">',
               width=width, height=height)

# Simulation message Div:
alert_msg_div = Div(text='', width=300)

# Interconnection table message Div:
excel_parse_msg_div = Div(text='', width=300)

# Insert input excel button:
input_xlsx_wgt = FileInput(accept=".xlsx", multiple=False)
input_xlsx_wgt.on_change('value', xlsx_parse)

# Node table:
nodes_columns = [
    TableColumn(field="name", title="Node name"),
    TableColumn(field="desc", title="Node description"),
    TableColumn(field="type", title="Node Type [Input/Intermediate/Output]"),
]
nodes_data_table = DataTable(
    source=nodes_CSD,
    columns=nodes_columns,
    min_height=500,
    max_height=120,
    width=450,
    height = 500,
    editable=True,
    height_policy="fit",
    autosize_mode="fit_columns",
)
nodes_data_table_title = Div(
    text='FCM nodes',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Interconnections node table:
edges_columns = [
    TableColumn(field="source", title="Source node"),
    TableColumn(field="target", title="Target node"),
    TableColumn(field="weight", title="Weight"),
]
edges_data_table = DataTable(
    source=edges_CSD,
    columns=edges_columns,
    min_height=500,
    max_height=120,
    width=300,
    height = 500,
    editable=True,
    height_policy="fit",
)
edges_data_table_title = Div(
    text='FCM node interconnections',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)


# FCM topology figure:
# --------------------

# Tools of FCM figure:
taptool = TapTool()

hovertool = HoverTool()
hovertool.mode = 'mouse'
hovertool.point_policy='follow_mouse'
hovertool.line_policy = 'interp'

tools = [hovertool, taptool, ResetTool(), PanTool(),
         BoxZoomTool(), WheelZoomTool(), SaveTool(),]

fcm_plot = figure(
    height=650, width=900,
    title="FCP display",
    toolbar_location='right',
    toolbar_sticky = False,
    tools=tools,
    tooltips = 'weight: @Weight',
)
fcm_plot.toolbar.logo = None
fcm_plot.toolbar.active_drag = None
fcm_plot.toolbar.active_scroll = None
fcm_plot.toolbar.active_tap = taptool
fcm_plot.toolbar.active_inspect = [hovertool]
fcm_plot.xgrid.grid_line_color = None
fcm_plot.ygrid.grid_line_color = None

fcm_plot.xaxis.major_tick_line_color = None
fcm_plot.xaxis.minor_tick_line_color = None

fcm_plot.yaxis.major_tick_line_color = None
fcm_plot.yaxis.minor_tick_line_color = None

fcm_plot.xaxis.major_label_text_font_size = '0pt'
fcm_plot.yaxis.major_label_text_font_size = '0pt'

fcm_plot.xaxis.axis_line_width = 0
fcm_plot.xaxis.axis_line_color = None

fcm_plot.yaxis.axis_line_width = 0
fcm_plot.yaxis.axis_line_color = None

fcm_plot.on_change('renderers', _clear_msg)

#Updatate the FCM figure renderers for the 1st time:
(graph_renderer, labels_renderer) = update_graph_renderer(fcm_layout_dict)
fcm_plot.renderers = []
fcm_plot.renderers = [graph_renderer, labels_renderer]


# Widget No.2: Collect global variables
execute_btn = Button(label="Execute simulation", button_type="success", width=550)
execute_btn.on_click(collect_global_var)


# Simulation hyperparameters   ---------------------------------------------

# Widget No.3: Variable inputs: Number of iterations
input_iterations_spinner = Spinner(
    title="Number of MC iterations (variable inputs):",
    low=1, high=1000000, step=1, value=1, width=300,
    disabled = True,
)
input_iterations_spinner.on_change('value', _set_iterations_var_inputs)

# Widget No.4: Standard deviation (variable inputs):
inputs_sd_spinner = Spinner(
    title="Standard deviation (variable inputs)",
    low=0, high=1, step=0.05, value=0, width=210,
    disabled=True,
)
inputs_sd_spinner.on_change('value', _set_input_sd)

# Widget No.5: Input nodes variation:
variable_input_nodes_radio_button = CheckboxGroup(labels=["Input nodes variation"], active=[])
variable_input_nodes_radio_button.on_click(_is_there_variation_on_input_nodes)

# Widget No.6: Variable weights: Number of iterations:
weight_iterations_spinner = Spinner(
    title="Number of Monte Carlo iterations (variable weights):",
    low=1, high=1000000, step=1, value=1, width=300,
    disabled=True,
)
weight_iterations_spinner.on_change('value', _set_iterations_var_weights)

# Widget No.7: Standard deviation (variable inputs)
weight_sd_spinner = Spinner(
    title= 'Standard deviation (variable weights)',
    low=0, high=1, step=0.05, value=0.1, width=210,
    disabled=True,
)
weight_sd_spinner.on_change('value', _set_weights_sd)

# Widget No.8: Variance on zero weights
LABELS = ["Variance on zero weights?"]
variable_zero_weights_radio_button = CheckboxGroup(labels=LABELS, active=[], disabled=True)
variable_zero_weights_radio_button.on_click(_is_zero_weights_variable)

# Widget No.9: Weights variation:
# -------------------------------------------------
variable_weights_radio_button = CheckboxGroup(labels=["Weights variation"], active=[])
variable_weights_radio_button.on_click(_is_there_variation_on_weights)

# Widget No.10: Select transfer function
tr_function_select = Select(
    title="Transfer function:", value="Sigmoid",
    options=["sigmoid", "hyperbolic",], width=150)
tr_function_select.on_change("value", _set_transfer_function)

# Widget No.11: Set labbda:
lambda_spinner = Spinner(
    title= 'Set lambda value',
    low=0.001, high=20, step=0.5, value=0.1, width=150,
    disabled=True,
)
lambda_spinner.on_change('value', _set_lambda)

# Widget No.12: Set lambda:
lambda_autoselect_radio_button = CheckboxGroup(labels=["Autoselect lambda?"], active=[0])
lambda_autoselect_radio_button.on_click(_lambda_autoselect)

# -------------------------

if transfer_function == 'sigmoid':
    x_range=[0, 1]
elif transfer_function == 'hyperbolic':
    x_range=[-1, 1]

# -----------------------------------
# Figure 1.: Input nodes
# -----------------------------------
f1 = figure(
   x_range=x_range,
   y_range=FactorRange(),
   height=500,width=900,
   title="Input nodes",
   toolbar_location='right',
   toolbar_sticky = False,
)
f1.toolbar.logo = None

# -----------------------------------
# Figure 2.: Intermediate nodes
# -----------------------------------
f2 = figure(
    x_range=x_range,
    y_range=FactorRange(),
    height=500,width=900,
    title="Intermediate nodes",
    toolbar_location='right',
    toolbar_sticky = False,
)
f2.toolbar.logo = None

# -----------------------------------
# Figure 3.: Output nodes
# -----------------------------------
f3 = figure(
    x_range=x_range,
    y_range=FactorRange(),
    height=500,width=900,
    title="Output nodes",
    toolbar_location='right',
    toolbar_sticky = False,
)
f3.toolbar.logo = None

# ####################################################
tab1 = Panel(child=f1, title="Input nodes")
tab2 = Panel(child=f2, title="Intermediate nodes")
tab3 = Panel(child=f3, title="Output nodes")

tabs = Tabs(tabs=[tab1, tab2, tab3])

lambda_div = Div()

extract_btn = Button(label="Save results", button_type="success", width=550)
#execute_btn.on_click(collect_global_var)



# ----------------------------------------------------------------------------------
# Webpage layout  --- --------------------------------------------------------------
# ----------------------------------------------------------------------------------

# FCM plot and tables display:
fcm_display_layout = layout(
    row(
        fcm_plot,
        column(
            separator(width=550, height=15),
            input_xlsx_wgt,
            excel_parse_msg_div,
            separator(width=550, height=15),
            row(
                column(nodes_data_table_title, nodes_data_table),
                column(edges_data_table_title, edges_data_table),
            ),
        )
    )
)

# Simulation layout:
input_nodes_layout = layout(
    separator(width=550, height=15),
    variable_input_nodes_radio_button,
    [input_iterations_spinner, inputs_sd_spinner],
    separator(width=550, height=15)
)
weights_layout = layout(
    separator(width=550, height=15),
    variable_weights_radio_button,
    [weight_iterations_spinner, weight_sd_spinner],
    variable_zero_weights_radio_button,
    separator(width=550, height=15)
)
lambda_layout = layout(
    separator(width=550, height=15),
    lambda_autoselect_radio_button,
    [lambda_spinner, tr_function_select],
    separator(width=550, height=15)
)
simulation_parameters_layout = layout(
    input_nodes_layout,
    weights_layout,
    lambda_layout,
    separator(width=550, height=15),
    alert_msg_div,
    execute_btn,
    separator(width=550, height=15),
)
results_layout = layout(
    column(tabs, lambda_div, extract_btn))

# Parent layout:
web_page_layout = layout(
    page_header,
    separator(width=1500, height=15),
    fcm_display_layout,
    separator(width=1500, height=15),
    [simulation_parameters_layout, results_layout],
    separator(width=1500, height=15),
    [acknowledgements, license, github_repo],
)

# ----------------------------------------------------------------------------------
# Run bokeh server   ---------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Append web page layout to curent bokeh layout
current_doc.add_root(web_page_layout)

os.system('bokeh serve --show ./')
