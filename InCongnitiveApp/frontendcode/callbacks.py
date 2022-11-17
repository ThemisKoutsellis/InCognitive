# callbacks.py

import io
import numpy as np
from base64 import b64decode
from functools import partial

from bokeh.models import ColumnDataSource

# import internal modules
from frontendcode.parsers import parse_input_xlsx
from frontendcode.internal_functions import (
    display_msg, excecute_fcmmc, check_for_inconsistencies)

__all__ = (
    '_set_iter_when_weights_vary',
    '_set_iter_when_inputs_vary',
    '_get_xlsx',
    '_set_lambda',
    '_autoslect_lambda',
    '_variation_on_weights',
    '_variation_on_input_nodes',
    '_set_trans_func',
    '_set_input_sd',
    '_set_weights_sd',
    '_are_zero_weights_rand_var',
    '_clear_allert_msg_div',
    '_collect_global_var',
    '_del_edges_cds_rows',
    '_del_nodes_cds_rows',
)

#######################################################################
def _get_xlsx(attr, old, new, doc):

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

#######################################################################
def _set_iter_when_weights_vary(attr, old, new, doc):

    display_msg(doc.get_model_by_name('alert_msg_div'))
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
def _set_iter_when_inputs_vary(attr, old, new, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

    _in_sd_spinner = doc.get_model_by_name('input_nodes_sd_spinner')

    doc.iter_on_input_nodes = new

    if new<2:
        _in_sd_spinner.value = 0
        _in_sd_spinner.disabled = True
    else:
        _in_sd_spinner.disabled = False

#######################################################################
def _set_lambda(attr, old, new, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.lamda = new

#######################################################################
def _autoslect_lambda(active, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

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
def _variation_on_weights(active, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

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
def _variation_on_input_nodes(active, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

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
def _set_trans_func(attr, old, new, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.trans_func = new
    print(doc.trans_func)

#######################################################################
def _set_input_sd(attr, old, new, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.input_nodes_sd = new

#######################################################################
def _set_weights_sd(attr, old, new, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

    doc.weights_sd = new

#######################################################################
def _are_zero_weights_rand_var(active, doc):
    display_msg(doc.get_model_by_name('alert_msg_div'))

    if active:
        doc.zero_weights_are_rand_var = True
    else:
        doc.zero_weights_are_rand_var = False

#######################################################################
def _clear_allert_msg_div(attr, old, new, doc):
    lambda_div = doc.get_model_by_name('lambda_div')
    alert_msg_div = doc.get_model_by_name('alert_msg_div')
    if lambda_div and alert_msg_div:
        display_msg(alert_msg_div)
        display_msg(lambda_div)

#######################################################################
def _del_edges_cds_rows(doc):
    if 'index' in doc.edges_CDS.data:
        del doc.edges_CDS.data['index']
    _rows_to_del = doc.edges_CDS.selected.indices

    # deleting rows from Edges DataTables
    _edges_df = doc.edges_CDS.to_df()
    _edges_df.drop(_rows_to_del, inplace=True)

    doc.edges_CDS.data = doc.edges_CDS.from_df(_edges_df)
    if 'index' in doc.edges_CDS.data:
            del doc.edges_CDS.data['index']

    # Uncheck the rest of DataTable rows
    doc.edges_CDS.selected.indices = []

#######################################################################
def _del_nodes_cds_rows(doc):

    if 'index' in doc.nodes_CDS.data:
        del doc.nodes_CDS.data['index']
    if 'index' in doc.edges_CDS.data:
        del doc.edges_CDS.data['index']
    #print('doc.nodes_CDS=', doc.nodes_CDS.data)

    _nodes_df = doc.nodes_CDS.to_df()
    _edges_df = doc.edges_CDS.to_df()

    # Rows to delete in both DataTables
    # ----------------------------------
    # 1. Node DataTable
    _node_rows_to_del = doc.nodes_CDS.selected.indices
    _nodes_to_delete = [_nodes_df.iloc[i,0] for i in _node_rows_to_del]
    #print('_rows_to_del= ', _node_rows_to_del)
    # 2. Edges DataTable
    _edges_to_delete_idx = []
    for _node in _nodes_to_delete:
        pass












    # Deleting rows from both DataTables
    # ----------------------------------
    print('_nodes_df= \n', _nodes_df.head(2))
    _nodes_df.drop(_node_rows_to_del, inplace=True)
    print()
    print('_nodes_df= \n', _nodes_df.head(2))
    #Change the doc.CDS
    doc.nodes_CDS.data = doc.nodes_CDS.from_df(_nodes_df)
    if 'index' in doc.nodes_CDS.data:
            del doc.nodes_CDS.data['index']
    print()
    print('doc.nodes_CDS=', doc.nodes_CDS.data)
    print('========================================')
    # Uncheck the rest of DataTable rows
    doc.nodes_CDS.selected.indices = []
    doc.edges_CDS.selected.indices = []


    #_edges_cds  = doc.edges_CDS
    #if 'index' in _edges_cds.data:
    #    del _edges_cds.data['index']
    #_edges_df = _edges_cds.to_df()
    #_nodes_to_delete = [_nodes_df.iloc[i,0] for i in _rows_to_del]
    #print('_nodes_to_delete=', _nodes_to_delete)
    #_edges_to_delete_idx = []
    #for _node in _nodes_to_delete:
    #    _idx_list1 = _edges_df.index[_edges_df['source'] == _node].tolist()
    #    _edges_to_delete_idx = [*_edges_to_delete_idx, *_idx_list1]
    #    _idx_list2 = _edges_df.index[_edges_df['target'] == _node].tolist()
    #    _edges_to_delete_idx = [*_edges_to_delete_idx, *_idx_list2]
    #    _edges_to_delete_idx = list(set(_edges_to_delete_idx))
    #print('_edges_to_delete_idx=', _edges_to_delete_idx)

    # deleting rows from both DataTables
    #_nodes_df.drop(_rows_to_del, inplace=True)
    #print(_nodes_df)
    #_nodes_cds = ColumnDataSource(_nodes_df)

    #_edges_df.drop(_edges_to_delete_idx, inplace=True)
    #doc.edges_CDS = ColumnDataSource(_nodes_df)

    #if 'index' in _edges_cds.data:
    #    del doc.edges_CDS.data['index']
    #if 'index' in _nodes_cds.data:
    #    print(_nodes_cds.data)
    #    del _nodes_cds.data['index']

    #print('AFTER deleting')
    #print(doc.nodes_CDS.data)
    #print(doc.edges_CDS.data)
    #print('======================================')
    #print()
    #print()


    #_edges_cds.selected.indices = []

#######################################################################
def _collect_global_var(doc):

    print(doc.fcm_layout_dict)
    f1 = doc.get_model_by_name('f1')
    f2 = doc.get_model_by_name('f2')
    f3 = doc.get_model_by_name('f3')

    # initialize figures
    f1.renderers = []
    f2.renderers = []
    f3.renderers = []

    lambda_div = doc.get_model_by_name('lambda_div')
    alert_msg_div = doc.get_model_by_name('alert_msg_div')
    display_msg(lambda_div)
    display_msg(alert_msg_div)

    # check for inconsistencies
    check_incons_cb = partial(check_for_inconsistencies, doc)
    doc.add_next_tick_callback(check_incons_cb)
