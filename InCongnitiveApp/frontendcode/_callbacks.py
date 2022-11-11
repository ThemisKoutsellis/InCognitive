# _callbacks.py

import io

from base64 import b64decode
from functools import partial

from frontendcode._internal_functions import _display_msg
from frontendcode._parsers import parse_input_xlsx


__all__ = [
    '_set_iter_when_weights_vary',
    '_set_iterations_when_inputs_vary',
    '_get_xlsx',
]

#################################
def _get_xlsx(
    attr, old, new,
    f1, f2, f3,



):
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


    (
        nodes_CSD,
        edges_CSD,
        fcm_layout_dict
    ) = parse_input_xlsx(
        file_io,
        nodes_CSD,
        edges_CSD,
        fcm_plot,
        excel_parse_msg_div
    )

    return







################################################
def _set_iter_when_weights_vary(
    attr, old, new,
    doc, div,
    var_zero_weights_rd_btn,
    weight_sd_spinner
):

    global weight_iterations
    global variance_on_zero_weights

    doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=doc,
                div=div,
                msg=' ',
                msg_type='alert'
            )
    )

    weight_iterations = new

    if new<2:
        var_zero_weights_rd_btn.active = []
        var_zero_weights_rd_btn.disabled = True
        variance_on_zero_weights = False
        weight_sd_spinner.value = 0
        weight_sd_spinner.disabled = True
    else:
        var_zero_weights_rd_btn.disabled = False
        weight_sd_spinner.disabled = False

    return None

################################################
def _set_iterations_when_inputs_vary(attr, old, new, doc, div, inputs_sd_spinner):


    global input_iterations

    doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=doc,
                div=div,
                msg=' ',
                msg_type='alert'
            )
    )

    input_iterations = new

    if new<2:
        inputs_sd_spinner.value = 0
        inputs_sd_spinner.disabled = True
    else:
        inputs_sd_spinner.disabled = False

    return None