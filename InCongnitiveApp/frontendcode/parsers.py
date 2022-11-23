# parsers.py

import pandas as pd

from backendcode.xlparse import get_fcm_layout
from frontendcode.internal_functions import _update_graph_renderer


all = ('parse_input_xlsx')

def parse_input_xlsx(
    file_io,
    nodes_CDS,
    edges_CDS,
    fcm_plot,
    msg_div
):

    # Exceptions:
    try:
        xl_file = pd.ExcelFile(file_io)
        sheet_list = ['nodes-order', 'input-output-nodes', 'fcm-topology']
        # Exception 2: Invalid sheets
        if xl_file.sheet_names != sheet_list:
            fcm_plot.renderers = []
            msg_div.text = 'Wrong number or type of excel sheets'
            msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None
    except:
        # Exception 1: Invalid sheets
        fcm_plot.renderers = []
        msg_div.text = 'Error while reading the excel file'
        msg_div.style= {'font-size': '100%', 'color': 'red'}
        return None

    try:
        df_nodes_order = pd.read_excel(
            file_io,
            'nodes-order',
            engine='openpyxl'
        )
        df_in_out_nodes = pd.read_excel(
            file_io,
            'input-output-nodes',
            engine='openpyxl'
        )
        df_fcm_topology = pd.read_excel(
            file_io,
            'fcm-topology',
            engine='openpyxl'
        )

        sheet1cols = list(df_nodes_order.keys())
        sheet2cols = list(df_in_out_nodes.keys())
        sheet3cols = list(df_fcm_topology.keys())

        expr1 = sheet1cols!=[
            'nodes order',
            'node description',
            'initial value',
            'auto weights',
            'auto lags',
        ]
        expr2 = sheet2cols!=['input nodes', 'output nodes']
        expr3 = sheet3cols!=['source node', 'target node', 'weight', 'lag']

        # invalid sheet1
        if expr1:
            fcm_plot.renderers = []
            msg_div.text = 'Error in 1st sheet named: "nodes-order"'
            msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None
        # invalid sheet2
        if expr2:
            fcm_plot.renderers = []
            msg_div.text = 'Error in 2nd sheet named: "input-output-nodes"'
            msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None
        # invalid sheet3
        if expr3:
            fcm_plot.renderers = []
            msg_div.text = 'Error in 3rd sheet named: "fcm-topology"'
            msg_div.style= {'font-size': '100%', 'color': 'red'}
            return None

    except:
        # Exception 3: Error while reading the sheets
        fcm_plot.renderers = []
        msg_div.text = 'Error while reading the excel sheets'
        return None

    msg_div.text = ' '
    fcm_layout_dict = get_fcm_layout(
        df_nodes_order,
        df_fcm_topology,
        df_in_out_nodes,
    )

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

    source_nodes_data = {
        'name': fcm_layout_dict['nodes_order'],
        'desc': fcm_layout_dict['nodes_discription'],
        "type": node_type,
        "initial value": fcm_layout_dict['initial_values'],
        "auto-weight": fcm_layout_dict['auto_weights'],

    }
    source_edges_data = {
        'source': fcm_layout_dict['source_nodes'],
        'target': fcm_layout_dict['target_nodes'],
        'weight': fcm_layout_dict['weights'],
    }

    nodes_CDS.data = source_nodes_data
    edges_CDS.data = source_edges_data

    (
        graph_renderer,
        labels_renderer
    ) = _update_graph_renderer(fcm_layout_dict)

    fcm_plot.renderers = []
    fcm_plot.renderers = [graph_renderer, labels_renderer]

    return nodes_CDS, edges_CDS, fcm_layout_dict
