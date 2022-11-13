# _web_page_layout.py

# general imports
from bokeh.layouts import layout, column, row

# import internal modules
from frontendcode._widgets import *


__all__ = ('web_page_layout')

# ---------------------------------------------------------------------
# Webpage layout  --- -------------------------------------------------
# ---------------------------------------------------------------------

# FCM plot and tables display:
_fcm_display_layout = layout(
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
_input_nodes_layout = layout(
    separator(width=550, height=15),
    variable_input_nodes_rb,
    [iter_on_input_nodes_spinner, input_nodes_sd_spinner],
    separator(width=550, height=15)
)

_weights_layout = layout(
    separator(width=550, height=15),
    variable_weights_rb,
    [iter_on_weights_spinner, weight_sd_spinner],
    variable_zero_weights_rb,
    separator(width=550, height=15)
)

_lambda_layout = layout(
    separator(width=550, height=15),
    lambda_autoselect_rb,
    [lambda_spinner, tr_function_select],
    separator(width=550, height=15)
)

_simulation_parameters_layout = layout(
    _input_nodes_layout,
    _weights_layout,
    _lambda_layout,
    separator(width=550, height=15),
    alert_msg_div,
    execute_btn,
    separator(width=550, height=15),
)

_results_layout = layout(
    column(tabs, lambda_div, extract_btn))

# Parent layout:
web_page_layout = layout(
    page_header,
    separator(width=1500, height=15),
    _fcm_display_layout,
    separator(width=1500, height=15),
    [_simulation_parameters_layout, _results_layout],
    separator(width=1500, height=15),
    [acknowledgements, license, github_repo],
)