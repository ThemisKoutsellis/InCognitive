# main.py

import os
from functools import partial

# bokeh & holoviews imports
from holoviews import opts
import holoviews as hv
hv.extension('bokeh')

from bokeh.io import curdoc
from bokeh.models import Spacer
from bokeh.models import DataTable
from bokeh.models import ColumnDataSource
from bokeh.layouts import layout, column, row
#From bokeh.models.layouts import Spacer

# import internal modules
from frontendcode.widgets import *
from frontendcode.callbacks import *
from frontendcode.wpage_layouts import *
from frontendcode.internal_functions import _update_graph_renderer

# ---------------------------------------------------------------------
# Sessions variables   ------------------------------------------------
# ---------------------------------------------------------------------

# Assign the seesion vars to the current Doc
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

current_doc.output_nodes_mc_values = {}

# This variable is needed for the backend code.
# If False, no normalization procedure. Otherwise,
# perform normalization.
current_doc.FCMMC = True
# variables to manage the events that happen simultaneously.
current_doc.dont_update_fcm_layout_dict = False
current_doc.deleting_rows_from_nodes_DataTable = False

# ---------------------------------------------------------------------
# Widgets and layouts that depends on doc vars   ----------------------
# ---------------------------------------------------------------------

nodes_data_table = DataTable(
    source=current_doc.nodes_CDS,
    columns=nodes_columns,
    min_height=500,
    max_height=120,
    width=450,
    height = 500,
    editable=True,
    height_policy="fit",
    autosize_mode="fit_columns",
    header_row=True,
    sortable=True,
    auto_edit=True,
    selectable='checkbox',
    scroll_to_selection=True,
    )

edges_data_table = DataTable(
    source=current_doc.edges_CDS,
    columns=edges_columns,
    min_height=500,
    max_height=120,
    width=300,
    height = 500,
    editable=True,
    height_policy="fit",
    auto_edit=True,
    selectable='checkbox',
    scroll_to_selection=True,
)

# FCM plot and tables display:
datatables_layout = row(
    column(
        nodes_data_table_title,
        nodes_data_table,
        node_buttons
    ),
    Spacer(width=20, height=20),
    column(
        edges_data_table_title,
        edges_data_table,
        edges_buttons
    ),
)
fcm_data_manager_layout = column(
    separator(width=550, height=15),
    uploadxlsx_layout,
    separator(width=550, height=15),
    datatables_layout,
    separator(width=550, height=15),
    save_bn,
)

fcm_display_layout = layout(
    row(fcm_plot, Spacer(width=20, height=20), fcm_data_manager_layout)
)

# Root (web page) layout:
web_page_layout = column(
    web_page_header,
    separator(width=1500, height=15),
    fcm_display_layout,
    separator(width=1500, height=15),
    fcmmc_layout,
    separator(width=1500, height=15),
    footer_layout,
)

# ---------------------------------------------------------------------
# Attach callbacks on widgets    --------------------------------------
# ---------------------------------------------------------------------

upload_xlsx_cb = partial(get_xlsx, doc=current_doc)
upload_xlsx_wgt.on_change('value', upload_xlsx_cb)

iter_on_weights_cb = partial(set_iter_when_weights_vary, doc=current_doc)
iter_on_weights_spinner.on_change('value', iter_on_weights_cb)

iter_on_input_nodes_cb = partial(set_iter_when_inputs_vary, doc=current_doc)
iter_on_input_nodes_spinner.on_change('value', iter_on_input_nodes_cb)

set_lambda_cb = partial(set_lambda, doc=current_doc)
lambda_spinner.on_change('value', set_lambda_cb)

lambda_autoselect_cb = partial(autoslect_lambda, doc=current_doc)
lambda_autoselect_rb.on_click(lambda_autoselect_cb)

variation_on_weights_cb = partial(variation_on_weights, doc=current_doc)
variable_weights_rb.on_click(variation_on_weights_cb)

variation_on_input_nodes_cb = partial(variation_on_input_nodes, doc=current_doc)
variable_input_nodes_rb.on_click(variation_on_input_nodes_cb)

set_input_sd_cb = partial(set_input_sd, doc=current_doc)
input_nodes_sd_spinner.on_change('value', set_input_sd_cb)

set_trans_func_cb = partial(set_trans_func, doc=current_doc)
tr_function_select.on_change("value", set_trans_func_cb)

set_weights_sd_cb = partial(set_weights_sd, doc=current_doc)
weight_sd_spinner.on_change('value', set_weights_sd_cb)

are_zero_weights_variable_cb = partial(are_zero_weights_rand_var, doc=current_doc)
variable_zero_weights_rb.on_click(are_zero_weights_variable_cb)

clear_allert_msg_div_cb = partial(clear_allert_msg_div, doc=current_doc)
fcm_plot.on_change('renderers', clear_allert_msg_div_cb)

clear_allert_msg_div_cb = partial(collect_global_var, doc=current_doc)
execute_btn.on_click(clear_allert_msg_div_cb)

clear_allert_msg_div_cb = partial(collect_global_var, doc=current_doc)
execute_btn.on_click(clear_allert_msg_div_cb)

add_edge_cds_row_cb = partial(add_edge_cds_row, doc=current_doc)
add_edge_row.on_click(add_edge_cds_row_cb)

del_edges_cb = partial(del_edges_cds_rows, doc=current_doc)
del_edge_row.on_click(del_edges_cb)

add_node_cds_row_cb = partial(add_node_cds_row, doc=current_doc)
add_node_row.on_click(add_node_cds_row_cb)

del_nodes_cb = partial(del_nodes_cds_rows, doc=current_doc)
del_node_row.on_click(del_nodes_cb)

nodes_CDS_changed_cb = partial(
    update_fcm_layout_dict, doc=current_doc, who='nodesCDS')
current_doc.nodes_CDS.on_change('data',nodes_CDS_changed_cb)

edges_CDS_changed_cb = partial(
    update_fcm_layout_dict, doc=current_doc, who='edgesCDS')
current_doc.edges_CDS.on_change('data',edges_CDS_changed_cb)

# ---------------------------------------------------------------------
# Initialize doc   ----------------------------------------------------
# ---------------------------------------------------------------------

#Updatate the FCM figure renderers for the 1st time:
(
    graph_renderer,
    labels_renderer
) = _update_graph_renderer(
    current_doc.fcm_layout_dict
)
fcm_plot.renderers = []
fcm_plot.renderers = [graph_renderer, labels_renderer]

# ---------------------------------------------------------------------
# Run bokeh server   --------------------------------------------------
# ---------------------------------------------------------------------

# Append web page layout to curent bokeh layout
current_doc.add_root(web_page_layout)
# run bokeh server
os.system('bokeh serve --show ./')
