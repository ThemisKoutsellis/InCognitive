# _callbacks.py

import io

from base64 import b64decode
from functools import partial

from frontendcode._parsers import parse_input_xlsx
from frontendcode._internal_functions import _display_msg

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
)

# clear msg div
def _clear_msg_div(doc, div):
    msg_cb = partial(
        _display_msg,
        doc=doc,
        div=div,
        msg=' ',
        msg_type='alert'
    )
    doc.add_next_tick_callback(msg_cb)

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
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

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
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

    _in_sd_spinner = doc.get_model_by_name('input_nodes_sd_spinner')

    doc.iter_on_input_nodes = new

    if new<2:
        _in_sd_spinner.value = 0
        _in_sd_spinner.disabled = True
    else:
        _in_sd_spinner.disabled = False

#######################################################################
def _set_lambda(attr, old, new, doc):
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

    doclamda = new

#######################################################################
def _autoslect_lambda(active, doc):
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

    _lambda_spinner = doc.get_model_by_name('lambda_spinner')

    if active:
        _lambda_spinner.disabled = True
        doc.autoslect_lambda = True
        doc.lamda = None
    else:
        doc.autoslect_lambda = False
        _lambda_spinner.disabled = False
        _lambda_spinner.value = 0.5

#######################################################################
def _variation_on_weights(active, doc):
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

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
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

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
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

    doc.trans_func = new

#######################################################################
def _set_input_sd(attr, old, new, doc):
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

    doc.input_nodes_sd = new

#######################################################################
def _set_weights_sd(attr, old, new, doc):
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

    doc.weights_sd = new

#######################################################################
def _are_zero_weights_rand_var(active, doc):
    _clear_msg_div(doc, doc.get_model_by_name('alert_msg_div'))

    if active:
        doc.zero_weights_are_rand_var = True
    else:
        doc.zero_weights_are_rand_var = False

#######################################################################
def _clear_allert_msg_div(attr, old, new, doc):
    doc.set_select({'name': 'alert_msg_div'}, {'text': ' '})

#######################################################################
def collect_global_var():

    global f1, f2, f3

    f1.renderers = []
    f2.renderers = []
    f3.renderers = []

    current_doc.add_next_tick_callback(
        partial(
            _display_msg,
            doc=current_doc,
            div=alert_msg_div,
            msg=' ',
            msg_type='alert',
        )
    )

    error1 = not bool(upload_xlsx_wgt.filename)
    error2 = not bool(current_doc.fcm_layout_dict)
    if not error2:
        error3 = not bool(current_doc.fcm_layout_dict['source_nodes'])
    else:
        error3 = True

    _expr = error1 or error2 or error3
    if _expr:

        _error_str = '[Error]: There is no input excel file OR the excel file is erroneous!'
        current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=_error_str,
                msg_type='error'
            )
        )
    else:
        _expr1 = current_doc.iter_on_input_nodes == current_doc.iter_on_weights
        _expr2 = (current_doc.iter_on_input_nodes < 2) or (current_doc.iter_on_weights < 2)

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
                current_doc.fcm_layout_dict,
                current_doc.iter_on_input_nodes,
                current_doc.iter_on_weights,
                current_doc.input_nodes_sd,
                current_doc.weights_sd,
                current_doc.zero_weights_are_rand_var,
                current_doc.trans_func,
                current_doc.lamda,
                current_doc.autoslect_lambda,
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

            if current_doc.trans_func == 'sigmoid':
                _x = list(np.linspace(0, 1, N))
                bisect.insort(_x, 0.5)
                _set_x_range(0,1)

            elif current_doc.trans_func == 'hyperbolic':
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
                partial(
                    _display_lambda,
                    doc=current_doc,
                    div=lambda_div,
                    mc_lambda=mc_lambda,
                )
            )

            current_doc.add_next_tick_callback(
                partial(
                    _display_msg,
                    doc=current_doc,
                    div=alert_msg_div,
                    msg='Execution ended successfully.',
                    msg_type='success'
                )
            )
        else:
            #### TODO FIX THIS STRANGE CODE WITH fcm_layout_dict = partial(

            current_doc.add_next_tick_callback(
                current_doc.nodes_CDS,
                current_doc.edges_CDS,
                fcm_layout_dict = partial(
                    _display_msg,
                    doc=current_doc,
                    div=alert_msg_div,
                    msg='[ALERT]: The number of iterations (Weight & Input) must be equal!',
                    msg_type='alert'
                )
            )
            pass

    return None