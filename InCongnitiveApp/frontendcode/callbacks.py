# callbacks.py

import io
import numpy as np
import pandas as pd
from base64 import b64decode
from functools import partial
from time import sleep

from bokeh.models import ColumnDataSource

# import internal modules
from frontendcode.parsers import parse_input_xlsx
from frontendcode.internal_functions import (
    _display_msg,
    _check_for_inconsistencies,
    _update_fcm_dict,
)

__all__ = (
    'set_iter_when_weights_vary',
    'set_iter_when_inputs_vary',
    'get_xlsx',
    'set_lambda',
    'autoslect_lambda',
    'variation_on_weights',
    'variation_on_input_nodes',
    'set_trans_func',
    'set_input_sd',
    'set_weights_sd',
    'are_zero_weights_rand_var',
    'clear_allert_msg_div',
    'collect_global_var',
    'del_edges_cds_rows',
    'del_nodes_cds_rows',
    'add_node_cds_row',
    'add_edge_cds_row',
    'update_fcm_layout_dict',
)

#######################################################################
def get_xlsx(attr, old, new, doc):
    doc.dont_update_fcm_layout_dict = True

    doc.fcm_layout_dict = {}
    doc.nodes_CDS.data = {}
    doc.edges_CDS.data = {}

    div = doc.get_model_by_name('xlsx_msg_div')
    fcm_plot = doc.get_model_by_name('fcm_plot')
    f1 = doc.get_model_by_name('f1')
    f2 = doc.get_model_by_name('f2')
    f3 = doc.get_model_by_name('f3')

    f1.renderers = []
    f2.renderers = []
    f3.renderers = []

    raw_data = b64decode(new)
    file_io = io.BytesIO(raw_data)

    (doc.nodes_CDS,
     doc.edges_CDS,
     doc.fcm_layout_dict
    )  = parse_input_xlsx(
        file_io,
        doc.nodes_CDS,
        doc.edges_CDS,
        fcm_plot,
        div,
    )
    doc.dont_update_fcm_layout_dict = False

#######################################################################
def set_iter_when_weights_vary(attr, old, new, doc):

    _display_msg(doc.get_model_by_name('alert_msg_div'))
    doc.iter_on_weights = new

    _zero_w_rb = doc.get_model_by_name('variable_zero_weights_rb')
    _w_sd_spinner = doc.get_model_by_name('weight_sd_spinner')

    if new<2:
        _zero_w_rb.active = []
        _zero_w_rb.disabled = True
        doc.zero_weights_are_rand_var = False
        _w_sd_spinner.value = 0
        _w_sd_spinner.disabled = True
    else:
        _zero_w_rb.disabled = False
        _w_sd_spinner.disabled = False

#######################################################################
def set_iter_when_inputs_vary(attr, old, new, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    _in_sd_spinner = doc.get_model_by_name('input_nodes_sd_spinner')

    doc.iter_on_input_nodes = new

    if new<2:
        _in_sd_spinner.value = 0
        _in_sd_spinner.disabled = True
    else:
        _in_sd_spinner.disabled = False

#######################################################################
def set_lambda(attr, old, new, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.lamda = new

#######################################################################
def autoslect_lambda(active, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    _lambda_spinner = doc.get_model_by_name('lambda_spinner')

    if active:
        _lambda_spinner.disabled = True
        doc.autoslect_lambda = True
        doc.lamda = None
    else:
        doc.autoslect_lambda = False
        _lambda_spinner.disabled = False
        _lambda_spinner.value = 0.5
        doc.lamda = 0.5

#######################################################################
def variation_on_weights(active, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    _itr_w_spinner = doc.get_model_by_name('iter_on_weights_spinner')
    _w_sd_spinner = doc.get_model_by_name('weight_sd_spinner')
    _var_zero_w_rb= doc.get_model_by_name('variable_zero_weights_rb')

    if active:
        _w_sd_spinner.disabled = False
        _itr_w_spinner.disabled = False

        doc.iter_on_weights = 2
        _itr_w_spinner.value = 2
        _w_sd_spinner.value = 0.1

        _itr_w_spinner.low = 2

        _var_zero_w_rb.active = []
        _var_zero_w_rb.disabled = False
        doc.zero_weights_are_rand_var = False
    else:
        doc.weights_sd = 0
        doc.iter_on_weights = 1

        _itr_w_spinner.value = 1
        _itr_w_spinner.disabled = True
        _itr_w_spinner.low = 1

        _w_sd_spinner.disabled = True

        _var_zero_w_rb.active = []
        _var_zero_w_rb.disabled = True
        doc.zero_weights_are_rand_var = False

#######################################################################
def variation_on_input_nodes(active, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    _itr_spinner = doc.get_model_by_name('iter_on_input_nodes_spinner')
    _in_sd_spinner = doc.get_model_by_name('input_nodes_sd_spinner')

    if active:
        _itr_spinner.disabled = False
        _itr_spinner.value = 2
        _itr_spinner.low = 2
        doc.iter_on_input_nodes = 2

        _in_sd_spinner.disabled = False
        _in_sd_spinner.value = 0.1

    else:
        _itr_spinner.disabled = True
        _itr_spinner.value = 1
        _itr_spinner.low = 1
        doc.iter_on_input_nodes = 1

        _itr_spinner.disabled = True
        doc.sd_inputs = 0

#######################################################################
def set_trans_func(attr, old, new, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.trans_func = new
    print(doc.trans_func)

#######################################################################
def set_input_sd(attr, old, new, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.input_nodes_sd = new

#######################################################################
def set_weights_sd(attr, old, new, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.weights_sd = new

#######################################################################
def are_zero_weights_rand_var(active, doc):
    _display_msg(doc.get_model_by_name('alert_msg_div'))

    if active:
        doc.zero_weights_are_rand_var = True
    else:
        doc.zero_weights_are_rand_var = False

#######################################################################
def clear_allert_msg_div(attr, old, new, doc):
    lambda_div = doc.get_model_by_name('lambda_div')
    alert_msg_div = doc.get_model_by_name('alert_msg_div')
    if lambda_div and alert_msg_div:
        _display_msg(alert_msg_div)
        _display_msg(lambda_div)

#######################################################################
def collect_global_var(doc):

    #print(doc.fcm_layout_dict)
    f1 = doc.get_model_by_name('f1')
    f2 = doc.get_model_by_name('f2')
    f3 = doc.get_model_by_name('f3')

    # initialize figures
    f1.renderers = []
    f2.renderers = []
    f3.renderers = []

    lambda_div = doc.get_model_by_name('lambda_div')
    alert_msg_div = doc.get_model_by_name('alert_msg_div')
    _display_msg(lambda_div)
    _display_msg(alert_msg_div)

    # check for inconsistencies
    check_incons_cb = partial(_check_for_inconsistencies, doc)
    doc.add_next_tick_callback(check_incons_cb)

#######################################################################
def add_node_cds_row(doc):
    _nodes_df = doc.nodes_CDS.to_df()
    _nodes_data = {
        'name': ['NaN'],
        'desc': ['NaN'],
        'type': ['NaN'],
        'initial value': [np.nan],
        'auto-weight': [np.nan],
    }
    _nodes_empty_df = pd.DataFrame(_nodes_data)
    _nodes_df = pd.concat(
        [_nodes_df, _nodes_empty_df],
        ignore_index = True
    )
    #Change the doc.CDS
    doc.nodes_CDS.data = doc.nodes_CDS.from_df(_nodes_df)

    # delete the 'index' column
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

#######################################################################
def add_edge_cds_row(doc):
    _edges_df = doc.edges_CDS.to_df()

    _edges_data = {
        'source': ['NaN'],
        'target': ['NaN'],
        'weight': [np.nan],
    }
    _edges_empty_df = pd.DataFrame(_edges_data)
    _edges_df = pd.concat(
        [_edges_df, _edges_empty_df],
        ignore_index = True
    )

    #Change the doc.CDS
    doc.edges_CDS.data = doc.edges_CDS.from_df(_edges_df)

    # delete the 'index' column
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

#######################################################################
def del_edges_cds_rows(doc):
    doc.dont_update_fcm_layout_dict = False
    if 'index' in doc.edges_CDS.data:
        doc.dont_update_fcm_layout_dict = True
        del doc.edges_CDS.data['index']
        doc.dont_update_fcm_layout_dict = False

    _rows_to_del = doc.edges_CDS.selected.indices
    # deleting rows from Edges DataTables
    _edges_df = doc.edges_CDS.to_df()
    _edges_df.drop(_rows_to_del, inplace=True)

    # Change edges CDS:
    doc.dont_update_fcm_layout_dict = False
    doc.edges_CDS.data = doc.edges_CDS.from_df(_edges_df)
    if 'index' in doc.edges_CDS.data:
        doc.dont_update_fcm_layout_dict = True
        del doc.edges_CDS.data['index']
        doc.dont_update_fcm_layout_dict = False

    # Uncheck the rest of DataTable rows
    doc.edges_CDS.selected.indices = []

#######################################################################
def del_nodes_cds_rows(doc):
    doc.deleting_rows_from_nodes_DataTable = True
    if 'index' in doc.nodes_CDS.data or doc.edges_CDS.data:
        doc.dont_update_fcm_layout_dict = True
        if 'index' in doc.nodes_CDS.data:
            del doc.nodes_CDS.data['index']
        elif  'index' in doc.edges_CDS.data:
            del doc.edges_CDS.data['index']
        doc.dont_update_fcm_layout_dict = False

    _nodes_df = doc.nodes_CDS.to_df()
    _edges_df = doc.edges_CDS.to_df()

    # Rows to delete in both DataTables
    # ----------------------------------
    # 1. Node DataTable
    _node_rows_to_del = doc.nodes_CDS.selected.indices
    _nodes_to_delete = [_nodes_df.iloc[i,0] for i in _node_rows_to_del]
    # 2. Edges DataTable
    if doc.edges_CDS.data:
        _edges_to_delete_idx = []
        for _node in _nodes_to_delete:
            _idx_list1 = _edges_df.index[
                _edges_df['source'] == _node].tolist()
            _edges_to_delete_idx = [*_edges_to_delete_idx, *_idx_list1]
            _idx_list2 = _edges_df.index[
                _edges_df['target'] == _node].tolist()
            _edges_to_delete_idx = [*_edges_to_delete_idx, *_idx_list2]
        _edges_to_delete_idx = list(set(_edges_to_delete_idx))
    else:
        _edges_to_delete_idx = []

    # Deleting rows from both DataTables
    # ----------------------------------
    _nodes_df.drop(_node_rows_to_del, inplace=True)
    _edges_df.drop(_edges_to_delete_idx, inplace=True)

    #Change the doc.CDS:
    if _edges_to_delete_idx:
        doc.dont_update_fcm_layout_dict = True
        doc.nodes_CDS.data = doc.nodes_CDS.from_df(_nodes_df)
        doc.dont_update_fcm_layout_dict = False
        doc.edges_CDS.data = doc.edges_CDS.from_df(_edges_df)
    else:
        doc.dont_update_fcm_layout_dict = False
        doc.nodes_CDS.data = doc.nodes_CDS.from_df(_nodes_df)
        doc.edges_CDS.data = doc.edges_CDS.from_df(_edges_df)

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

    doc.deleting_rows_from_nodes_DataTable = False

#######################################################################
def update_fcm_layout_dict(attr, old, new, doc, who):
    _dict = doc.fcm_layout_dict

    if who == 'nodesCDS':
        if not doc.dont_update_fcm_layout_dict:
            if 'index' in doc.nodes_CDS.data or doc.edges_CDS.data:
                doc.dont_update_fcm_layout_dict = True
                if 'index' in doc.nodes_CDS.data:
                    del doc.nodes_CDS.data['index']
                elif  'index' in doc.edges_CDS.data:
                    del doc.edges_CDS.data['index']
                doc.dont_update_fcm_layout_dict = False
            doc.fcm_layout_dict = _update_fcm_dict(doc, _dict)
    elif who == 'edgesCDS':
        if not doc.dont_update_fcm_layout_dict:
            if 'index' in doc.nodes_CDS.data or doc.edges_CDS.data:
                doc.dont_update_fcm_layout_dict = True
                if 'index' in doc.nodes_CDS.data:
                    del doc.nodes_CDS.data['index']
                elif  'index' in doc.edges_CDS.data:
                    del doc.edges_CDS.data['index']
                doc.dont_update_fcm_layout_dict = False
            doc.fcm_layout_dict = _update_fcm_dict(doc, _dict)
