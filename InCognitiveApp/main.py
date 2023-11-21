# main.py

import os
from functools import partial

# bokeh & holoviews imports
import holoviews as hv
hv.extension('bokeh')

from bokeh.io import curdoc
from bokeh.models import (Spacer, DataTable, ColumnDataSource, 
    Div, Button, Spinner, CheckboxGroup, Select, Panel, Tabs,
    FactorRange, TableColumn, BoxZoomTool, PanTool, ResetTool,
    HoverTool, TapTool, WheelZoomTool, SaveTool,
)
from bokeh.models.widgets import FileInput, NumberEditor, TextEditor
from bokeh.layouts import layout, column, row
from bokeh.plotting import figure

# import internal modules
from frontendcode.callbacks import *
from frontendcode.internal_functions import _update_graph_renderer


PARIS_REINFORCE_COLOR = '#9CAB35'

# ---------------------------------------------------------------------
# Webpage elements   --------------------------------------------------
# ---------------------------------------------------------------------

# Header Div:
web_page_header = Div(
    text=("<figure>"
          "<img src='InCongnitiveApp/static/images/in_cognitive_logo.png' "
          "width='400' height='45' "
          "alt='FCM Simulation application'"
          "</figure>"
          ),
          width=400, height=60,
    )

# Acknowledgements Div:
acknowledgements = Div(
    text=(
        'Powered by: '
        '<a href="https://www.python.org/" target=_blank>Python 3.7.3</a>'
        ', '
        '<a href="https://holoviews.org" target=_blank> HoloViews 1.14.8</a>'
        ' and '
        '<a href="https://docs.bokeh.org/en/latest/index.html"'
        ' target=_blank>Bokeh 2.3.3.</a>'
        ),
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# License Div:
license = Div(
    text=('License:'
          '<a href="https://creativecommons.org/licenses/by-nc-nd/4.0/"'
          ' target=_blank>(CC BY-NC-ND 4.0)</a>'),
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Github repository Div:
github_repo = Div(
    text=('<a href="https://github.com/ThemisKoutsellis/InCognitive"'
          ' target=_blank>GitHub repository</a>'),
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Separator line Div:
def separator(width=550, height=5):
    return Div(text=f'<hr width="{width}">',
               width=width, height=height)

# Interconnection table message Div:
excel_parse_msg_div = Div(text='', width=300, name='xlsx_msg_div')

# Insert input excel button:
upload_xlsx_wgt = FileInput(
    accept=".xlsx",
    multiple=False,
    name='upload_xlsx_wgt'
)

# Nodes DataTable:
nodes_columns = [
    TableColumn(field="name",
                title="Node name",
                editor=TextEditor()),
    TableColumn(field="desc",
                title="Node description",
                editor=TextEditor()),
    TableColumn(field="type",
                title="Node type [Input/Intermediate/Output]",
                editor=TextEditor()),
    TableColumn(field="initial value",
                title="Initial value",
                editor=NumberEditor()),
    TableColumn(field="auto-weight",
                title="Auto-weight",
                editor=NumberEditor()),
]

nodes_data_table_title = Div(
    text='FCM nodes',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

add_node_row = Button(
    label="Add new row",
    button_type="success", width=80,
)

del_node_row = Button(
    label="Delete selected row(s)",
    button_type="success", width=80,
)

# Node interconnections DataTable:
edges_columns = [
    TableColumn(field="source",
                title="Source node",
                editor=TextEditor()),
    TableColumn(field="target",
                title="Target node",
                editor=TextEditor()),
    TableColumn(field="weight",
                title="Weight",
                editor=NumberEditor()),
]

edges_data_table_title = Div(
    text='FCM node interconnections',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

add_edge_row = Button(
    label="Add new row",
    button_type="success", width=80,
)

del_edge_row = Button(
    label="Delete selected row(s)",
    button_type="success", width=80,
)

#######################################################################
# FCM topology figure:
#######################################################################

# Tools of FCM figure:
taptool = TapTool()

hovertool = HoverTool()
hovertool.mode = 'mouse'
hovertool.point_policy='follow_mouse'
hovertool.line_policy = 'interp'

tools = [hovertool, taptool, ResetTool(), PanTool(),
         BoxZoomTool(), WheelZoomTool(), SaveTool(),]

fcm_plot = figure(
    height=650, width=900,
    title="FCP display",
    toolbar_location='right',
    toolbar_sticky = False,
    tools=tools,
    tooltips = 'weight: @Weight',
    name='fcm_plot',
)

fcm_plot.toolbar.logo = None
fcm_plot.toolbar.active_drag = None
fcm_plot.toolbar.active_scroll = None
fcm_plot.toolbar.active_tap = taptool
fcm_plot.toolbar.active_inspect = [hovertool]
fcm_plot.xgrid.grid_line_color = None
fcm_plot.ygrid.grid_line_color = None

fcm_plot.xaxis.major_tick_line_color = None
fcm_plot.xaxis.minor_tick_line_color = None

fcm_plot.yaxis.major_tick_line_color = None
fcm_plot.yaxis.minor_tick_line_color = None

fcm_plot.xaxis.major_label_text_font_size = '0pt'
fcm_plot.yaxis.major_label_text_font_size = '0pt'

fcm_plot.xaxis.axis_line_width = 0
fcm_plot.xaxis.axis_line_color = None

fcm_plot.yaxis.axis_line_width = 0
fcm_plot.yaxis.axis_line_color = None


# Widget No.2: Collect global variables
execute_btn = Button(
    label="Execute simulation",
    button_type="success",
    width=550
)

# Simulation hyperparameters   ----------------------------------------

# # Widget No.3: Variable inputs: Number of iterations
iter_on_input_nodes_spinner = Spinner(
    title="Number of MC iterations (variable inputs):",
    low=1, high=1000000, step=1, value=1, width=300,
    disabled = True,
    name='iter_on_input_nodes_spinner',
)

# Widget No.4: Standard deviation (variable inputs):
input_nodes_sd_spinner = Spinner(
    title="Standard deviation (variable inputs)",
    low=0, high=1, step=0.05, value=0, width=210,
    disabled=True,
    name='input_nodes_sd_spinner',
)

# Widget No.5: Input nodes variation:
variable_input_nodes_rb = CheckboxGroup(
    labels=["Input nodes variation"],
    active=[]
)

# Widget No.6: Standard deviation (variable inputs)
weight_sd_spinner = Spinner(
    title= 'Standard deviation (variable weights)',
    low=0, high=1, step=0.05, value=0.1, width=210,
    disabled=True,
    name='weight_sd_spinner',
)

# Widget No.7: Variance on zero weights
LABELS = ["Variance on zero weights?"]
variable_zero_weights_rb = CheckboxGroup(
    labels=LABELS,
    active=[],
    disabled=True,
    name='variable_zero_weights_rb',
)

# Widget No.8: Variable weights: Number of iterations:
iter_on_weights_spinner = Spinner(
    title="Number of Monte Carlo iterations (variable weights):",
    low=1, high=1000000, step=1, value=1, width=300,
    disabled=True,
    name='iter_on_weights_spinner',
)

# Widget No.9: Weights variation:
# -------------------------------------------------
variable_weights_rb = CheckboxGroup(
    labels=["Weights variation"],
    active=[]
)

# Widget No.10: Select transfer function
tr_function_select = Select(
    title="Transfer function:",
    value="Sigmoid",
    options=["sigmoid","hyperbolic"],
    width=150
)

# Widget No.11: Set labbda:
lambda_spinner = Spinner(
    title= 'Set lambda value',
    low=0.001, high=100, step=0.5, value=0.1, width=150,
    disabled=True,
    name='lambda_spinner',
)

# Widget No.12: Set lambda:
lambda_autoselect_rb = CheckboxGroup(
    labels=["Autoselect lambda?"],
    active=[0]
)

# -----------------------------------
# Figure 1.: Input nodes
# -----------------------------------
f1 = figure(
   x_range=[0, 1],
   y_range=FactorRange(),
   height=500,width=900,
   title="Input nodes",
   toolbar_location='right',
   toolbar_sticky = False,
   name='f1'
)
f1.toolbar.logo = None

# -----------------------------------
# Figure 2.: Intermediate nodes
# -----------------------------------
f2 = figure(
    x_range=[0, 1],
    y_range=FactorRange(),
    height=500,width=900,
    title="Intermediate nodes",
    toolbar_location='right',
    toolbar_sticky = False,
    name='f2'
)
f2.toolbar.logo = None

# -----------------------------------
# Figure 3.: Output nodes
# -----------------------------------
f3 = figure(
    x_range=[0, 1],
    y_range=FactorRange(),
    height=500,width=900,
    title="Output nodes",
    toolbar_location='right',
    toolbar_sticky = False,
    name='f3'
)
f3.toolbar.logo = None

# ####################################################
_tab1 = Panel(child=f1, title="Input nodes")
_tab2 = Panel(child=f2, title="Intermediate nodes")
_tab3 = Panel(child=f3, title="Output nodes")

tabs = Tabs(tabs=[_tab1, _tab2, _tab3])

msg_div = Div(name='msg_div')

save_bn = Button(
    label="Save layout to xlsx format",
    button_type="success", width=550,
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
    variable_input_nodes_rb,
    [iter_on_input_nodes_spinner, input_nodes_sd_spinner],
    separator(width=550, height=15)
)

weights_layout = layout(
    variable_weights_rb,
    [iter_on_weights_spinner, weight_sd_spinner],
    variable_zero_weights_rb,
    separator(width=550, height=15)
)

lambda_layout = layout(
    lambda_autoselect_rb,
    [lambda_spinner, tr_function_select],
    separator(width=550, height=15)
)

simulation_parameters_layout = layout(
    input_nodes_layout,
    weights_layout,
    lambda_layout,
    separator(width=550, height=15),
    execute_btn,
)

results_layout = layout(column(tabs, msg_div))

fcmmc_layout = row(
    simulation_parameters_layout,
    spacer,
    results_layout)

footer_layout = row(acknowledgements, license, github_repo)


# ---------------------------------------------------------------------
# Session's variables   ------------------------------------------------
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
    'auto_weights': [],
    'auto_lags': [],
    'initial_values': [],
    'lags': [],
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

#clear_allert_msg_div_cb = partial(clear_allert_msg_div, doc=current_doc)
#fcm_plot.on_change('renderers', clear_allert_msg_div_cb)

collect_global_var_cb = partial(collect_global_var, doc=current_doc)
execute_btn.on_click(collect_global_var_cb)

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
fcm_plot.renderers = [graph_renderer, labels_renderer]

# ---------------------------------------------------------------------
# Run bokeh server   --------------------------------------------------
# ---------------------------------------------------------------------

# Append web page layout to curent bokeh layout
current_doc.add_root(web_page_layout)

