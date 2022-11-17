# callbacks.py

import io
import numpy as np
from base64 import b64decode
from functools import partial

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
    '_collect_global_var'
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

