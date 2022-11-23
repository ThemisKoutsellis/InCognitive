# wpage_layouts.py

# general imports
from bokeh.layouts import layout, column, row
from bokeh.models import Spacer

# import internal modules
from frontendcode.widgets import *

__all__ = (
    'uploadxlsx_layout',
    'input_nodes_layout',
    'weights_layout',
    'lambda_layout',
    'results_layout',
    'simulation_parameters_layout',
    'fcmmc_layout',
    'footer_layout',
    'edges_buttons',
    'node_buttons',
)

# ---------------------------------------------------------------------
# Webpage layouts  ----------------------------------------------------
# ---------------------------------------------------------------------

uploadxlsx_layout = column(upload_xlsx_wgt, excel_parse_msg_div)

spacer = Spacer(width=20, height=20)
node_buttons = row(add_node_row, spacer, del_node_row)
edges_buttons = row(add_edge_row, spacer, del_edge_row)

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

results_layout = layout(column(tabs, lambda_div))

fcmmc_layout = row(
    simulation_parameters_layout,
    spacer,
    results_layout)

footer_layout = row(acknowledgements, license, github_repo)

