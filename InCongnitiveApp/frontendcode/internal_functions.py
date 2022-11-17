# internal_functions.py
import bisect
import operator
import pandas as pd
import numpy as np

import networkx as nx
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

# bokeh & holoviews imports
import holoviews as hv
hv.extension('bokeh')

from bokeh.models import ColumnDataSource

# import internal modules
from backendcode.fcmmc_simulation import monte_carlo_simulation
from backendcode.fcm_layout_parameters import (
    select_lambda, get_nx_graph, get_w_matrix)


__all__ = (
    'plot_results',
    'display_msg',
    'update_graph_renderer',
    'excecute_fcmmc',
    'check_for_inconsistencies',
)

def display_msg(div, msg=' ', msg_type='alert'):

    def _show_msg():
        div.text = str(msg)
        # style of msg str
        if msg_type=='error':
            div.style= {'font-size': '100%', 'color': 'red'}
        elif msg_type=='alert':
            div.style= {'font-size': '100%', 'color': 'blue'}
        elif msg_type=='info':
            div.style= {'font-size': '100%', 'color': 'black'}
        else:
            div.style= {'font-size': '100%', 'color': 'green'}

    _show_msg()

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
        sorted_zip =  sorted(
            zipped_node_coord, key = operator.itemgetter(0))
        x_y_list = list(zip(*sorted_zip))
        xs = list(x_y_list[0])
        ys = list(x_y_list[1])

    # rearrange labels
    intermidiate_nodes = [
        node for node in labels \
            if (node not in input_nodes) and (node not in output_nodes)
    ]
    rearranged_labels = input_nodes + intermidiate_nodes + output_nodes

    # create the output df
    zipped = list(zip(xs, ys, rearranged_labels))
    rearranged_node_data_df = pd.DataFrame(
        zipped, columns=['x', 'y', 'index']
    )

    return rearranged_node_data_df

def _ridge(category, data, scale=20):
    return list(zip([category]*len(data), scale*data))

def plot_results(f, _x, node_mc_values, baseline_node_values):

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
                    f.patch(
                        'x', k,
                        source=source,
                        alpha=0.6,
                        line_color=None
                    )
                else:
                    pass

                # plot mean (or baseline scenario)
                _baseline_data = baseline_node_values[k]
                _y = _ridge(k, [0])
                f.circle(
                    _baseline_data,
                    _y, alpha=0.6,
                    line_color="black",
                    size=7, color="red"
                )
            else:
                _y = _ridge(k, [0])
                if isinstance(_data, list):
                    _data = [_data[0]]
                f.circle(
                    _data, _y,
                    alpha=0.6,
                    line_color="black",
                    size=7, color="red"
                )

def update_graph_renderer(fcm_layout_dict):

    input_nodes = fcm_layout_dict["input_nodes"]
    output_nodes = fcm_layout_dict["output_nodes"]

    nodes_discription = fcm_layout_dict["nodes_discription"]
    source_nodes = fcm_layout_dict['source_nodes']
    target_nodes = fcm_layout_dict['target_nodes']
    nodes_order_list = fcm_layout_dict["nodes_order"]

    nx_graph = get_nx_graph(
        fcm_layout_dict['source_nodes'],
        fcm_layout_dict['target_nodes'],
        fcm_layout_dict['weights'],
    )

    initial_hv_graph = hv.Graph.from_networkx(
        nx_graph, nx.layout.circular_layout
    )

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

def excecute_fcmmc(doc):

    # Execute FCM-MC Sim after passing the test of values inconsistency
    ###################################################################

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
        doc.fcm_layout_dict,
        doc.iter_on_input_nodes,
        doc.iter_on_weights,
        doc.input_nodes_sd,
        doc.weights_sd,
        doc.zero_weights_are_rand_var,
        doc.trans_func,
        doc.lamda,
        doc.autoslect_lambda,
    )

    # Display results
    # -----------------------------------------------------------------
    f1 = doc.get_model_by_name('f1')
    f2 = doc.get_model_by_name('f2')
    f3 = doc.get_model_by_name('f3')

    lambda_div = doc.get_model_by_name('lambda_div')
    alert_msg_div = doc.get_model_by_name('alert_msg_div')

    # Initialize Figures 1,2 & 3
    def _set_x_range(start, end):
        f1.x_range.start = start
        f1.x_range.end = end

        f2.x_range.start = start
        f2.x_range.end = end

        f3.x_range.start = start
        f3.x_range.end = end
    N = 600
    if doc.trans_func == 'sigmoid':
        _x = list(np.linspace(0, 1, N))
        bisect.insort(_x, 0.5)
        _set_x_range(0,1)
    elif doc.trans_func == 'hyperbolic':
        _x = list(np.linspace(-1, 1, N))
        bisect.insort(_x, 0)
        _set_x_range(-1,1)

    # Plot Figures 1,2 & 3
    plot_results(
        f1, _x,
        input_nodes_mc_values,
        baseline_input_nodes_values
    )
    plot_results(
        f2, _x,
        intermediate_nodes_mc_values,
        baseline_intermediate_nodes_values
    )
    plot_results(
        f3, _x,
        output_nodes_mc_values,
        baseline_output_nodes_values
    )

    # Final display msgs
    _error_str = (
        'Execution ended successfully!'
        ' Transfer function: λ = {0}'.format(mc_lambda)
    )
    display_msg(lambda_div, msg=_error_str, msg_type='success')
    display_msg(alert_msg_div, msg='', msg_type='success')
    return

def _display_last_exec_msg(doc, _error_str, _lambda_div_str):

    lambda_div = doc.get_model_by_name('lambda_div')
    alert_msg_div = doc.get_model_by_name('alert_msg_div')

    display_msg(alert_msg_div, msg=_error_str, msg_type='alert')
    display_msg(lambda_div, msg=_error_str, msg_type='alert')
    return

def _check_lambdas(doc):
    nodes_order = doc.fcm_layout_dict['nodes_order']
    input_nodes = doc.fcm_layout_dict['input_nodes']
    auto_w = doc.fcm_layout_dict['auto_weights']
    activation_function = doc.trans_func
    zero_weights_are_rand_var = doc.zero_weights_are_rand_var
    lamda_value = doc.lamda
    lamda_autoslect = doc.autoslect_lambda
    if doc.iter_on_weights > 1:
        weights_are_rand_var = True
    else:
        weights_are_rand_var = False
    nx_graph = get_nx_graph(
        doc.fcm_layout_dict['source_nodes'],
        doc.fcm_layout_dict['target_nodes'],
        doc.fcm_layout_dict['weights'],
    )
    w_matrix = get_w_matrix(nx_graph, nodes_order, input_nodes, auto_w)

    # If I have FCM-MC is not allowed to have
    # lambda greater than the autoselect value
    # FCM-MC case:
    # 1. Input nodes are random variables, weight not.
    _expr1 = (doc.iter_on_input_nodes>1) and (doc.iter_on_weights<2)
    # 2. Input nodes are constants. Weight are random variables.
    _expr2 = (doc.iter_on_input_nodes<2) and (doc.iter_on_weights>1)
    # 3. Input nodes and weights are randoms variables.
    _expr3 = (doc.iter_on_input_nodes>1) and (doc.iter_on_weights>1)\
        and (doc.iter_on_input_nodes == doc.iter_on_weights)
    _FCMMC_expr = _expr1 or _expr2 or _expr3

    # if we have FCMMC and the autselect_lambda is active
    # (equivalently, doc.lamda==None) there is no need to
    # chack the lambda parameter for consistency.
    if _FCMMC_expr and doc.lamda:
        max_accepted_lambda, lamda_autoslect = select_lambda(
            w_matrix,
            nodes_order,
            input_nodes,
            activation_function,
            lamda_value,
            True,
            zero_weights_are_rand_var,
            weights_are_rand_var,
        )
        print('>> Accepted_lambda {0}'.format(max_accepted_lambda))
        if max_accepted_lambda < doc.lamda:
            lambda_is_OK = False
        else:
            lambda_is_OK = True
    else:
        lambda_is_OK = True
        max_accepted_lambda = np.inf

    return lambda_is_OK, max_accepted_lambda


def _check_initial_input_node_values(doc):

    _initial_values = doc.fcm_layout_dict['initial_values']
    initial_input_node_values_are_within_accepted_range = False

    try:
        _initial_values = [float(x) for x in _initial_values]
    except:
        initial_input_node_values_are_float = False
        initial_input_node_values_are_within_accepted_range = False
        range_str = None
    else:
        initial_input_node_values_are_float = True

        max_init_v = max(_initial_values)
        min_init_v = min(_initial_values)
        if doc.trans_func == 'sigmoid':
            range_str = '[0,1]'
            if (min_init_v>=0) and (max_init_v<=1):
                initial_input_node_values_are_within_accepted_range = True
        elif doc.trans_func == 'hyperbolic':
            range_str = '[-1,1]'
            if (min_init_v >=- 1) and (max_init_v <= 1):
                initial_input_node_values_are_within_accepted_range = True

    return (range_str,
            initial_input_node_values_are_within_accepted_range,
            initial_input_node_values_are_float)

def _check_initial_weight_values(doc):

    _initial_weight_values = doc.fcm_layout_dict['weights']
    print(_initial_weight_values)
    initial_weight_values_are_floats = False

    try:
        _initial_weight_values = [
            float(x) for x in _initial_weight_values]
    except:
        initial_weight_values_are_floats = False
        initial_weight_values_are_within_accepted_range = False
    else:
        initial_weight_values_are_floats = True

        _max_value = max(_initial_weight_values)
        _min_value = min(_initial_weight_values)
        if (_min_value>=-1) and (_max_value<=1):
            initial_weight_values_are_within_accepted_range = True
        else:
            initial_weight_values_are_within_accepted_range = True

    return (initial_weight_values_are_within_accepted_range,
            initial_weight_values_are_floats)

def check_for_inconsistencies(doc):
    # get necessary widgets
    lambda_div = doc.get_model_by_name('lambda_div')
    alert_msg_div = doc.get_model_by_name('alert_msg_div')
    upload_xlsx_wgt = doc.get_model_by_name('upload_xlsx_wgt')

    # initialize parameters
    _error_str = ' '
    proceed = False

    # check of input values inconsistencies
    # -------------------------------------
    # 1. Estimate the check-expressions
    error1 = not bool(upload_xlsx_wgt.filename)
    error2 = not bool(doc.fcm_layout_dict)
    if not error2:
        error3 = not bool(doc.fcm_layout_dict['source_nodes'])
    else:
        error3 = True
    there_is_no_input_excel = error1 or error2 or error3
    _expr2 = doc.iter_on_input_nodes == doc.iter_on_weights
    _expr3 = (doc.iter_on_input_nodes < 2) or (doc.iter_on_weights < 2)
    mc_iterations_dont_much = not (_expr2 or _expr3)
    (
    initial_weight_values_are_within_accepted_range,
    initial_weight_values_are_floats,
    ) = _check_initial_weight_values(doc)
    (
    range_str,
    initial_input_node_values_are_within_accepted_range,
    initial_input_node_values_are_float,
    ) = _check_initial_input_node_values(doc)

    data_tables_are_OK = initial_weight_values_are_within_accepted_range and \
        initial_weight_values_are_floats and \
        initial_input_node_values_are_within_accepted_range and \
        initial_input_node_values_are_float
    if data_tables_are_OK:
        lambda_is_OK, max_accepted_lambda = _check_lambdas(doc)
    else:
        pass

    # 2. Validate inputs
    # If-statements in that order. Do not change!
    if there_is_no_input_excel:
        _error_str = ('[ERROR]: There is no input excel'
                    ' file OR the excel file is erroneous!')
    elif not initial_weight_values_are_floats:
        _error_str = ('[ERROR]: Some, or all, of the '
            'weight values are NaN.')
    elif not initial_weight_values_are_within_accepted_range:
        _error_str = ('Some, or all, of the initial weight'
            ' values are out of range.')
    elif not initial_input_node_values_are_float:
        _error_str = (
            '[ERROR]: Some, or all, of the '
            'initial input-node-values are NaN.'
        )
    elif not initial_input_node_values_are_within_accepted_range:
        print(doc.trans_func)
        _error_str = ('[ERROR] The transfer function is: {0}. '
            'Some, or all, of the initial node values are out of'
            ' range, {1}.'.format(doc.trans_func, range_str)
        )
    elif mc_iterations_dont_much:
        _error_str = ('[ERROR]: The number of iterations'
                    ' (Weight & Input) must be equal!')
    elif not lambda_is_OK:
        _error_str = ('[ERROR] Max-accepted-λ* (={0})'
            ' is less than the given parameter λ (={1}).\n'
            '* is the parameter λ which quearntees'
            ' 100% FCM convergence for all iterations.'.format(
                max_accepted_lambda, doc.lamda)
        )
    else:
        _error_str = 'Please wait ...'
        proceed = True

    display_msg(lambda_div, msg=_error_str, msg_type='alert')

    # call the FCM-MC function on next tick
    if proceed:
        fcmmc_cb = partial(excecute_fcmmc, doc)
        doc.add_next_tick_callback(fcmmc_cb)
    else:
        display_last_exec_msg_cb = partial(
            _display_last_exec_msg, doc, _error_str, _error_str)
        doc.add_next_tick_callback(display_last_exec_msg_cb)
