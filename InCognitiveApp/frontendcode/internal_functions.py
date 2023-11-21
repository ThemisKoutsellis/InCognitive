# internal_functions.py

import bisect
import operator
import pandas as pd
import numpy as np
from time import sleep

import networkx as nx
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

# bokeh & holoviews imports
import holoviews as hv
hv.extension('bokeh')
from holoviews.core.dimension import Dimension

from bokeh.models import ColumnDataSource

# import internal modules
from backendcode.fcmmc_simulation import monte_carlo_simulation
from backendcode.fcm_layout_parameters import (
    select_lambda, get_nx_graph, get_w_matrix)

__all__ = (
    '_plot_results',
    '_update_graph_renderer',
    '_excecute_fcmmc',
    '_check_for_inconsistencies',
    '_update_fcm_dict',
)

#######################################################################
def _ridge(category, data, scale=20):
    return list(zip([category]*len(data), scale*data))

#######################################################################
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

    return
#######################################################################
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

#######################################################################
def _update_graph_renderer(fcm_layout_dict):

    nodes_order_list = fcm_layout_dict["nodes_order"]
    nodes_discription = fcm_layout_dict["nodes_discription"]

    input_nodes = fcm_layout_dict["input_nodes"]
    output_nodes = fcm_layout_dict["output_nodes"]
    source_nodes = fcm_layout_dict['source_nodes']
    target_nodes = fcm_layout_dict['target_nodes']
    weights = fcm_layout_dict['weights']
    nx_graph = get_nx_graph(
        nodes_order_list,
        source_nodes,
        target_nodes,
        weights
    )
    initial_hv_graph = hv.Graph.from_networkx(
        nx_graph, nx.layout.circular_layout
    )
    # rearrenge so that the input-nodes appear firsts, to the left.
    rearranged_node_data_df = _rearrange_nodes(
        initial_hv_graph.nodes.data,
        input_nodes,
        output_nodes,
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
    # -----------------------------------------------------------------
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
        vdims=hv.Dimension('Weight', range=(-1, 1))
    )

   # hvgraph options
    hv_graph.opts(
        directed=True,
        node_size=20,
        arrowhead_length=0.017,
        inspection_policy='edges',
        selection_policy='nodes',
        edge_hover_line_color='green',
        node_hover_fill_color='green',
        edge_cmap=plt.cm.RdYlBu,
        node_cmap='Set1',
        cmap='brg',
        edge_color='Weight',
        node_color='Type',
        edge_line_width=1.4,
    )

    # convert hv labels to bokeh labels renderer object
    hv_labels_renderer = hv.Labels(hv_nodes, ['x', 'y'], 'Labels')

    hv_labels_renderer.opts(
        bgcolor='black',
        text_font_size='10pt',
        xoffset=0.07,
        yoffset=0.05,
)

    bokeh_labels_fig = hv.render(hv_labels_renderer)
    bokeh_labels_fig_renderers = bokeh_labels_fig.renderers
    bokeh_labels_renderer = bokeh_labels_fig_renderers[0]

    # convert hv to bokeh renderer object
    bokeh_graph = hv.render(hv_graph)
    bokeh_graph_renderer = bokeh_graph.renderers[0]

    return bokeh_graph_renderer, bokeh_labels_renderer

#######################################################################
def _excecute_fcmmc(doc):
    sleep(0.5)

    # Execute FCM-MC after passing the tests of values inconsistency
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
        doc.FCMMC,
    )
    # Display results
    # -----------------------------------------------------------------
    f1 = doc.get_model_by_name('f1')
    f2 = doc.get_model_by_name('f2')
    f3 = doc.get_model_by_name('f3')

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
    _plot_results(
        f1, _x,
        input_nodes_mc_values,
        baseline_input_nodes_values
    )
    _plot_results(
        f2, _x,
        intermediate_nodes_mc_values,
        baseline_intermediate_nodes_values
    )
    _plot_results(
        f3, _x,
        output_nodes_mc_values,
        baseline_output_nodes_values
    )

    # Final display msgs
    msg_div = doc.get_model_by_name('msg_div')
    _msg_str = (
        'Execution ended successfully!'
        ' Transfer function: λ = {0}'.format(mc_lambda)
    )
    msg_div.text = _msg_str
    msg_div.style= {'font-size': '100%', 'color': 'green'}

    return

#######################################################################
def _display_last_exec_msg(doc, msg_text):

    sleep(0.5)
    msg_div = doc.get_model_by_name('msg_div')
    msg_div.text = msg_text
    msg_div.style= {'font-size': '100%', 'color': 'red'}

    return

#######################################################################
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
        doc.fcm_layout_dict['nodes_order'],
        doc.fcm_layout_dict['source_nodes'],
        doc.fcm_layout_dict['target_nodes'],
        doc.fcm_layout_dict['weights'],
    )
    w_matrix = get_w_matrix(nx_graph, nodes_order, input_nodes, auto_w)

    # If we have FCM-MC, it is not allowed to have
    # lambda greater than the autoselect value.
    # FCM-MC case:
    # 1. Input nodes are random variables, weight not.
    _expr1 = (doc.iter_on_input_nodes>1) and (doc.iter_on_weights<2)
    # 2. Input nodes are constants. Weight are random variables.
    _expr2 = (doc.iter_on_input_nodes<2) and (doc.iter_on_weights>1)
    # 3. Input nodes and weights are randoms variables.
    _expr3 = (doc.iter_on_input_nodes>1) and (doc.iter_on_weights>1)\
        and (doc.iter_on_input_nodes == doc.iter_on_weights)
    _FCMMC_expr = _expr1 or _expr2 or _expr3

    if _FCMMC_expr:
        # This variable is necessary to the backend code.
        # If False, no normalization procedure. Otherwise,
        # perform normalization.
        doc.FCMMC = True
    else:
        doc.FCMMC = False

    # If we have FCMMC and the autselect_lambda is active
    # (equivalently, doc.lamda==None), there is no need to
    # check the lambda parameter for inconsistency.
    if _FCMMC_expr and doc.lamda:
        (
            _max_accepted_lambda,
            lamda_autoslect
        ) = select_lambda(
            w_matrix,
            nodes_order,
            input_nodes,
            activation_function,
            lamda_value,
            True,
            zero_weights_are_rand_var,
            weights_are_rand_var,
        )
        if _max_accepted_lambda < doc.lamda:
            _check__no_valid_lambda = True
        else:
            _check__no_valid_lambda = False
    else:
        _check__no_valid_lambda = False
        _max_accepted_lambda = np.inf

    return _check__no_valid_lambda, _max_accepted_lambda

#######################################################################
def _check_initial_input_node_values(doc):

    _initial_values = doc.fcm_layout_dict['initial_values']

    if doc.trans_func == 'sigmoid':
        range_str = '[0,1]'
    elif doc.trans_func == 'hyperbolic':
        range_str = '[-1,1]'

    if _initial_values:
        if np.isnan(_initial_values).any():
            _check__no_initial_node_values = True
            _check__no_valid_initial_node_values = True
        else:
            _check__no_initial_node_values = False

            _max_init_v = max(_initial_values)
            _min_init_v = min(_initial_values)
            if doc.trans_func == 'sigmoid':
                if (_min_init_v>=0) and (_max_init_v<=1):
                    _check__no_valid_initial_node_values = False
                else:
                    _check__no_valid_initial_node_values = True
            elif doc.trans_func == 'hyperbolic':
                if (_min_init_v >=- 1) and (_max_init_v <= 1):
                    _check__no_valid_initial_node_values = False
                else:
                    _check__no_valid_initial_node_values = True

    else:
        _check__no_initial_node_values = True
        _check__no_valid_initial_node_values = True

    return (
        range_str,
        _check__no_valid_initial_node_values,
        _check__no_initial_node_values,
    )

#######################################################################
def _check_weight_values(doc):

    _weight_values = doc.fcm_layout_dict['weights']

    if _weight_values:
        if np.isnan(_weight_values).any():
            _check__no_weight_values = True
            _check__no_valid_weight_values = True
        else:
            _check__no_weight_values = False

            _max_value = max(_weight_values)
            _min_value = min(_weight_values)
            if (_min_value>=-1) and (_max_value<=1):
                _check__no_valid_weight_values = False
            else:
                _check__no_valid_weight_values = True
    else:
        _check__no_weight_values = True
        _check__no_valid_weight_values = True

    return (
        _check__no_valid_weight_values,
        _check__no_weight_values
    )

#######################################################################
def _check_auto_weight_values(doc):

    _auto_weight_values = doc.fcm_layout_dict['auto_weights']

    if _auto_weight_values:
        if np.isnan(_auto_weight_values).any():
            _check__no_auto_weight_values = True
            _check__no_valid_auto_weight_values = True
        else:
            _check__no_auto_weight_values = False

            _max_value = max(_auto_weight_values)
            _min_value = min(_auto_weight_values)
            if (_min_value>=-1) and (_max_value<=1):
                _check__no_valid_auto_weight_values = False
            else:
                _check__no_valid_auto_weight_values = True
    else:
        _check__no_auto_weight_values = True
        _check__no_valid_auto_weight_values = True

    return (
        _check__no_valid_auto_weight_values,
        _check__no_auto_weight_values
    )

#######################################################################
def _check_for_source_nodes(doc):
    _dict = doc.fcm_layout_dict
    _source_nodes = _dict['source_nodes']

    if _source_nodes:
        if 'NaN' in set(_source_nodes):
            _check__no_source_nodes = True
        else:
            _check__no_source_nodes = False
    else:
        _check__no_source_nodes = True

    return _check__no_source_nodes

#######################################################################
def _check_for_inconsistencies(doc):
    # get necessary widgets
    msg_div = doc.get_model_by_name('msg_div')

    # initialize parameters
    _msg_str = ' '
    proceed = False

    _check__no_fcm_dict_layout = True
    _check__no_source_nodes = True
    _check__no_weight_values = True
    _check__no_valid_weight_values = True
    _check__no_initial_node_values = True
    _check__no_valid_initial_node_values = True
    _check__no_auto_weight_values = True
    _check__no_valid_auto_weight_values = True
    _check__no_valid_mc_iterations = True

    # check for input data inconsistencies
    # -------------------------------------
    # 1. Estimate the _check_ Booleans
    _check__no_fcm_dict_layout = not bool(doc.fcm_layout_dict)
    if not _check__no_fcm_dict_layout:
        _check__no_source_nodes = _check_for_source_nodes(doc)
        (
            _check__no_valid_weight_values,
            _check__no_weight_values,
        ) = _check_weight_values(doc)
        (
            range_str,
            _check__no_valid_initial_node_values,
            _check__no_initial_node_values,
        ) = _check_initial_input_node_values(doc)
        (
            _check__no_valid_auto_weight_values,
            _check__no_auto_weight_values
        )=_check_auto_weight_values(doc)

        _check__no_matched_iterations = \
            not (doc.iter_on_input_nodes == doc.iter_on_weights)
        _check__no_input_nodes_iterations = (doc.iter_on_input_nodes < 2)
        _check__no_weights_iterations = (doc.iter_on_weights < 2)
        _check__no_valid_mc_iterations = \
            (_check__no_matched_iterations) and not \
            (
                _check__no_weights_iterations or
                _check__no_input_nodes_iterations
            )
    _rest_of_input_data_are_OK = not (
        _check__no_fcm_dict_layout or \
        _check__no_source_nodes or \
        _check__no_weight_values or \
        _check__no_valid_weight_values or \
        _check__no_initial_node_values or\
        _check__no_valid_initial_node_values or \
        _check__no_auto_weight_values or \
        _check__no_valid_auto_weight_values or \
        _check__no_valid_mc_iterations
    )

    if _rest_of_input_data_are_OK:
        (
            _check__no_valid_lambda,
            _max_accepted_lambda
        ) = _check_lambdas(doc)

    # 2. Validate inputs
    # If-statements in that order. Do not change!
    proceed = False
    if _check__no_fcm_dict_layout:
        _msg_str = '[ERROR]: FCM layout not given!'
    elif _check__no_source_nodes:
        _msg_str = '[ERROR]: Some source-node values are "NaN"!'
    elif _check__no_weight_values:
        _msg_str = '[ERROR]:  Some weight values are "NaN"!'
    elif _check__no_valid_weight_values:
        _msg_str = '[ERROR]:  Some weight values are out of range!'
    elif _check__no_initial_node_values:
        _msg_str = '[ERROR]: Some initial node values are "NaN"!'
    elif _check__no_valid_initial_node_values:
        _msg_str = ('[ERROR] The transfer function is: {0}. '
            'Some of the initial node values are out of'
            ' range, {1}.'.format(doc.trans_func, range_str)
        )
    elif _check__no_auto_weight_values:
        _msg_str = '[ERROR]:  Some auto-weight values are "NaN"!'
    elif _check__no_valid_auto_weight_values:
        _msg_str = '[ERROR]:  Some auto-weight values are out of range!'
    elif _check__no_valid_mc_iterations:
        _msg_str = ('[ERROR]: The number of MC iterations'
                    ' (Weight & Input) must be equal!')
    elif _check__no_valid_lambda:
        _msg_str = ('[ERROR] Max-accepted-λ* (={0})'
            ' is smaller than the given parameter λ (={1}).\n'
            '* is the parameter λ which guarantees'
            ' 100% FCM convergence for all(!) iterations.'.format(
                _max_accepted_lambda, doc.lamda)
        )
    else:
        _msg_str = 'Please wait ...'
        proceed = True

    msg_div.text = _msg_str
    msg_div.style= {'font-size': '100%', 'color': 'blue'}

    # call the FCM-MC function on next tick
    if proceed:
        fcmmc_cb = partial(_excecute_fcmmc, doc=doc)
        doc.add_next_tick_callback(fcmmc_cb)
    else:
        display_last_exec_msg_cb = partial(
            _display_last_exec_msg, doc=doc, msg_text=_msg_str)
        doc.add_next_tick_callback(display_last_exec_msg_cb)

    return

#######################################################################
def _check_for_missing_nodes(doc):
    if doc.deleting_rows_from_nodes_DataTable:
        _missing_nodes = []
    else:
        # get the necessery lists to compare
        _nodes_df = doc.nodes_CDS.to_df()
        _nodes_df = _nodes_df.copy()
        _edges_df = doc.edges_CDS.to_df()
        _edges_df = _edges_df.copy()

        if not _edges_df.empty:
            _nodes_in_edges_DataTable = [
                *list(_edges_df['source']),
                *list(_edges_df['target']),
            ]
            _nodes_in_edges_DataTable = [
                x for x in _nodes_in_edges_DataTable \
                if isinstance(x,str) and x!='NaN'
            ]
            _nodes_in_edges_DataTable = list(set(_nodes_in_edges_DataTable))

            if not _nodes_df.empty:
                _nodes_in_nodes_DataTable = list(_nodes_df['name'])
                _nodes_in_nodes_DataTable = [
                    x for x in _nodes_in_nodes_DataTable \
                    if isinstance(x,str) and x!='NaN'
                ]
            else:
                _nodes_in_nodes_DataTable = []

            # check for missing nodes
            _missing_nodes = [
                x for x in _nodes_in_edges_DataTable \
                if x not in _nodes_in_nodes_DataTable
            ]
        else:
            _missing_nodes = []

    return _missing_nodes

#######################################################################
def _del_some_of_nan_rows(doc):
    _nodes_with_nan_df = doc.nodes_CDS.to_df()
    _nodes_with_nan_df = _nodes_with_nan_df.copy()
    _edges_with_nan_df = doc.edges_CDS.to_df()
    _edges_with_nan_df = _edges_with_nan_df.copy()
    # delete the NaN values (not all)
    if not _nodes_with_nan_df.empty:
        _nodes_with_nan_df = _nodes_with_nan_df[
            _nodes_with_nan_df.name != 'NaN']
    if not _edges_with_nan_df.empty:
        indexNames = _edges_with_nan_df[
            (_edges_with_nan_df['source'] == 'NaN') & \
            (_edges_with_nan_df['target'] == 'NaN')
        ].index
        _edges_with_nan_df.drop(indexNames , inplace=True)
        indexNames = _edges_with_nan_df[
            (_edges_with_nan_df['source'] == 'NaN') | \
            (_edges_with_nan_df['target'] == 'NaN')
        ].index
        _edges_with_nan_df.drop(indexNames , inplace=True)

    return _nodes_with_nan_df, _edges_with_nan_df

#######################################################################
def _add_missing_nodes(doc, _missing_nodes):
    _nodes_df = doc.nodes_CDS.to_df()
    _nodes_df = _nodes_df.copy()

    # add missing nodes in Nodes-DataTable.
    _nodes_data = {
        'name': [],
        'desc': [],
        'type': [],
        'initial value': [],
        'auto-weight': [],
    }

    for _node in _missing_nodes:
        _nodes_data['name'].append(_node)
        _nodes_data['desc'].append('NaN')
        _nodes_data['type'].append('NaN')
        _nodes_data['initial value'].append(np.nan)
        _nodes_data['auto-weight'].append(np.nan)

    _new_nodes_df = pd.DataFrame(_nodes_data)
    _nodes_df = pd.concat(
        [_nodes_df, _new_nodes_df],
        ignore_index = True
    )
    #Change the doc.CDS
    doc.nodes_CDS.data = doc.nodes_CDS.from_df(_nodes_df)
    # Delete the 'index' column because it causes
    # errors in other parts of the code.
    if 'index' in doc.nodes_CDS.data or doc.edges_CDS.data:
        doc.dont_update_fcm_layout_dict = True
        if 'index' in doc.nodes_CDS.data:
            del doc.nodes_CDS.data['index']
        elif  'index' in doc.edges_CDS.data:
            del doc.edges_CDS.data['index']
        doc.dont_update_fcm_layout_dict = False

    # Uncheck the rest of DataTable rows
    doc.nodes_CDS.selected.indices = []
    doc.edges_CDS.selected.indices = []

    return None

#######################################################################
def _fill_the_fcm_layout_dict(
    _dict,
    nodes_with_nan_df,
    edges_with_nan_df,
):
    _nodes_df = nodes_with_nan_df
    _edges_df = edges_with_nan_df

    _nodes_order = list(_nodes_df['name'])
    _nodes_discription = list(_nodes_df['desc'])
    _initial_values = list(_nodes_df['initial value'])
    _auto_weights = list(_nodes_df['auto-weight'])
    _auto_lags = [1]*len(_nodes_order)
    _nodes_type = list(_nodes_df['type'])
    # correct input values of node type
    _nodes_type = [
        'Intermediate' if item == 'intermediate' \
            else item for item in _nodes_type
    ]
    _nodes_type = [
        'Output' if item == 'output' \
        else item for item in _nodes_type
    ]
    _nodes_type = [
        'Input' if item == 'input' \
        else item for item in _nodes_type
    ]
    _nodes_type = [
        'Intermediate' if item == np.nan \
        else item for item in _nodes_type
    ]
    _valid_types = [
        'Intermediate',
        'intermediate',
        'Output',
        'output',
        'Input',
        'input',
    ]
    _nodes_type = [
        'Intermediate' if item not in _valid_types \
        else item for item in _nodes_type
    ]
    _input_nodes = [
        v for i, v in enumerate(_nodes_order) \
        if _nodes_type[i]=='Input'
    ]
    _output_nodes = [
        v for i, v in enumerate(_nodes_order) \
        if _nodes_type[i]=='Output'
    ]

    if not _edges_df.empty:
        _source_nodes = list(_edges_df['source'])
        _target_nodes = list(_edges_df['target'])
        _weights = list(_edges_df['weight'])
        _lags = [1]*len(_weights)
    else:
        _source_nodes = []
        _target_nodes = []
        _weights = []
        _lags = []

    # Check for NaN values in all DataTable columns
    no_NaN = True
    if 'nan' in np.nan_to_num(_nodes_order):
        no_NaN = False
    if 'nan' in np.nan_to_num(_nodes_discription):
        no_NaN = False
    if 'nan' in np.nan_to_num(_initial_values):
        no_NaN = False

    if not _edges_df.empty:
        if 'nan' in np.nan_to_num(_source_nodes):
            no_NaN = False
        if 'nan' in np.nan_to_num(_target_nodes):
            no_NaN = False
        if 'nan' in np.nan_to_num(_weights):
            no_NaN = False
    # -------------------------------------------------------
    if no_NaN:
        _dict['nodes_order'] = _nodes_order
        _dict['nodes_discription'] = _nodes_discription
        _dict['auto_weights'] = _auto_weights
        _dict['auto_lags'] = _auto_lags
        _dict['initial_values'] = _initial_values

        _dict['input_nodes'] = _input_nodes
        _dict['output_nodes'] = _output_nodes

        _dict['source_nodes'] = _source_nodes
        _dict['target_nodes'] = _target_nodes
        _dict['weights'] = _weights
        _dict['lags'] = _lags
    else:
        pass

    return _dict

#######################################################################
def _update_fcm_dict(doc, _dict):

    # Check if all nodes of Edges-DataTable exist in Nodes-DataTable
    # as well. If not, then add the corresponding rows to Nodes-DataTable.
    # ----------------------------------
    _missing_nodes = _check_for_missing_nodes(doc)

    if _missing_nodes:
        _add_missing_nodes(doc, _missing_nodes)
        # After the above command the _update_fcm_dict
        # will be called again because the CDS changed!
        # In the second invokation, the following 'else'
        # statement will be executed.
    else:
        (_nodes_with_nan_df,
         _edges_with_nan_df,
        ) = _del_some_of_nan_rows(doc)

        fcm_plot = doc.get_model_by_name('fcm_plot')

        if _nodes_with_nan_df.empty:
            _dict = {}
            fcm_plot.renderers = []
        else:
            _dict = _fill_the_fcm_layout_dict(
                _dict,
                _nodes_with_nan_df,
                _edges_with_nan_df,
            )
            (graph_renderer,
             labels_renderer
            ) = _update_graph_renderer(_dict)
            fcm_plot.renderers = [graph_renderer, labels_renderer]

    return _dict
