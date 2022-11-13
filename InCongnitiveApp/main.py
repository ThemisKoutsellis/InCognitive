# main.py

# general imports
import os
import bisect
import numpy as np
from functools import partial

# bokeh & holoviews imports
from holoviews import opts
import holoviews as hv
hv.extension('bokeh')

from bokeh.io import curdoc
from bokeh.layouts import layout, column, row
from bokeh.plotting import figure
from bokeh.models.widgets import FileInput
from bokeh.models import (
    Div, ColumnDataSource, Button,
    Spinner, CheckboxGroup, Select,
    Panel, Tabs, FactorRange, TableColumn,
    DataTable, BoxZoomTool, PanTool, ResetTool,
    HoverTool, TapTool, WheelZoomTool, SaveTool,
)

# import internal modules
from backendcode.fcmmc_simulation import monte_carlo_simulation

from frontendcode._widgets import *
from frontendcode._callbacks import *
from frontendcode._internal_functions import (
    _plot_results, _update_graph_renderer,
    _display_msg, _display_lambda
)


# ---------------------------------------------------------------------
# Apps variables   ----------------------------------------------------
# ---------------------------------------------------------------------

current_doc = curdoc()

current_doc.fcm_layout_dict = {
    'source_nodes': [],
    'target_nodes': [],
    'weights': [],
    'nodes_order': [],
    'input_nodes': [],
    'output_nodes': [],
    'nodes_discription': [],
}

current_doc.trans_func = 'sigmoid'

current_doc.iter_on_input_nodes = 1
current_doc.iter_on_weights = 1

current_doc.zero_weights_are_rand_var = False

current_doc.input_nodes_sd = 0.1
current_doc.weights_sd = 0.1

current_doc.lamda = None
current_doc.autoslect_lambda = True

current_doc.nodes_CDS = ColumnDataSource()
current_doc.edges_CDS = ColumnDataSource()

current_doc.input_nodes_CDS = ColumnDataSource()
current_doc.interm_nodes_CDS = ColumnDataSource()
current_doc.output_nodes_CDS = ColumnDataSource()

current_doc.output_nodes_mc_values = {}



# ---------------------------------------------------------------------
# Webpage layout  --- -------------------------------------------------
# ---------------------------------------------------------------------

# FCM plot and tables display:
fcm_display_layout = layout(
    row(
        fcm_plot,
        column(
            separator(width=550, height=15),
            upload_xlsx_wgt,
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
    variable_input_nodes_rb,
    [iter_on_input_nodes_spinner, input_nodes_sd_spinner],
    separator(width=550, height=15)
)
weights_layout = layout(
    separator(width=550, height=15),
    variable_weights_rb,
    [iter_on_weights_spinner, weight_sd_spinner],
    variable_zero_weights_rb,
    separator(width=550, height=15)
)
lambda_layout = layout(
    separator(width=550, height=15),
    lambda_autoselect_rb,
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

# ---------------------------------------------------------------------
# Assign callbacks on widgets    --------------------------------------
# ---------------------------------------------------------------------

upload_xlsx_cb = partial(_get_xlsx, doc=current_doc)
upload_xlsx_wgt.on_change('value', upload_xlsx_cb)

iter_on_weights_cb = partial(_set_iter_when_weights_vary, doc=current_doc)
iter_on_weights_spinner.on_change('value', iter_on_weights_cb)

iter_on_input_nodes_cb = partial(_set_iter_when_inputs_vary, doc=current_doc)
iter_on_input_nodes_spinner.on_change('value', iter_on_input_nodes_cb)

set_lambda_cb = partial(_set_lambda, doc=current_doc)
lambda_spinner.on_change('value', set_lambda_cb)

lambda_autoselect_cb = partial(_autoslect_lambda,doc=current_doc)
lambda_autoselect_rb.on_click(lambda_autoselect_cb)

variation_on_weights_cb = partial(_variation_on_weights, doc=current_doc)
variable_weights_rb.on_click(variation_on_weights_cb)

variation_on_input_nodes_cb = partial(_variation_on_input_nodes, doc=current_doc)
variable_input_nodes_rb.on_click(variation_on_input_nodes_cb)

set_input_sd_cb = partial(_set_input_sd, doc=current_doc)
input_nodes_sd_spinner.on_change('value', set_input_sd_cb)

set_trans_func_cb = partial(_set_trans_func, doc=current_doc)
tr_function_select.on_change("value", set_trans_func_cb)

set_weights_sd_cb = partial(_set_weights_sd, doc=current_doc)
weight_sd_spinner.on_change('value', set_weights_sd_cb)

is_zero_weights_variable_cb = partial(_are_zero_weights_rand_var, doc=current_doc)
variable_zero_weights_rb.on_click(_are_zero_weights_rand_var)

clear_allert_msg_div_cb = partial(_clear_allert_msg_div, doc=current_doc)
fcm_plot.on_change('renderers', clear_allert_msg_div_cb)

execute_btn.on_click(collect_global_var)

# ---------------------------------------------------------------------
# Run bokeh server   --------------------------------------------------
# ---------------------------------------------------------------------

# Append web page layout to curent bokeh layout
current_doc.add_root(web_page_layout)
# run bokeh server
os.system('bokeh serve --show ./')
