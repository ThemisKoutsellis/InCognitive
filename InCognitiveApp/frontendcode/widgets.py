# widgets.py

from bokeh.plotting import figure
from bokeh.models.widgets import FileInput, NumberEditor, TextEditor
from bokeh.models import (
    Div, Button, Spinner, CheckboxGroup, Select, Panel, Tabs,
    FactorRange, TableColumn, BoxZoomTool, PanTool, ResetTool,
    HoverTool, TapTool, WheelZoomTool, SaveTool, PreText
)

__all__ = (
    #'web_page_header',
    'separator',
    'excel_parse_msg_div',
    'upload_xlsx_wgt',
    'nodes_columns',
    'nodes_data_table_title',
    'edges_columns',
    'edges_data_table_title',
    'taptool',
    'hovertool',
    'tools',
    'fcm_plot',
    'execute_btn',
    'iter_on_input_nodes_spinner',
    'input_nodes_sd_spinner',
    'variable_input_nodes_rb',
    'weight_sd_spinner',
    'variable_zero_weights_rb',
    'iter_on_weights_spinner',
    'variable_weights_rb',
    'tr_function_select',
    'lambda_spinner',
    'lambda_autoselect_rb',
    'f1',
    'f2',
    'f3',
    'tabs',
    'msg_div',
    'save_bn',
    'add_edge_row',
    'del_edge_row',
    'del_node_row',
    'add_node_row',
    'callback_holder'
)


#PARIS_REINFORCE_COLOR = '#9CAB35'
PARIS_REINFORCE_COLOR = '#2f2f2f'
"""str: The default RGB color of PARIS REINFORCE
"""

# ---------------------------------------------------------------------
# Webpage elements   --------------------------------------------------
# ---------------------------------------------------------------------

# Header Div:
# web_page_header = Div(
#     text=("<figure>"
#           "<img src='InCognitiveApp/static/images/InCognitive2.png' "
#           "width='650' height='100' "
#           "alt='FCM Simulation application'"
#           "</figure>"
#           ),
#           width=650, height=100
#     )

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
    style={'font-size': '120%', 'color': PARIS_REINFORCE_COLOR}
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
    style={'font-size': '120%', 'color': PARIS_REINFORCE_COLOR}
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
    #title="FCP display",
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


fcm_plot.background_fill_color = "#3f3f3f"
fcm_plot.background_fill_alpha = 0.5
fcm_plot.border_fill_color = "whitesmoke"

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
   frame_height=500,frame_width=900,
   height=500,width=900,
   title="Input nodes",
   toolbar_location='right',
   toolbar_sticky = False,
   name='f1'
)
f1.toolbar.logo = None
f1.background_fill_color = "#3f3f3f"
f1.background_fill_alpha = 0.5

# -----------------------------------
# Figure 2.: Intermediate nodes
# -----------------------------------
f2 = figure(
    x_range=[0, 1],
    y_range=FactorRange(),
    frame_height=500,frame_width=900,
    height=500,width=900,
    title="Intermediate nodes",
    toolbar_location='right',
    toolbar_sticky = False,
    name='f2'
)
f2.toolbar.logo = None
f2.background_fill_color = "#3f3f3f"
f2.background_fill_alpha = 0.5

# -----------------------------------
# Figure 3.: Output nodes
# -----------------------------------
f3 = figure(
    x_range=[0, 1],
    y_range=FactorRange(),
    frame_height=500,frame_width=900,
    height=500,width=900,
    title="Output nodes",
    toolbar_location='right',
    toolbar_sticky = False,
    name='f3'
)
f3.toolbar.logo = None
f3.background_fill_color = "#3f3f3f"
f3.background_fill_alpha = 0.5

# ####################################################
_tab1 = Panel(child=f1, title="Input nodes")
_tab2 = Panel(child=f2, title="Intermediate nodes")
_tab3 = Panel(child=f3, title="Output nodes")

tabs = Tabs(tabs=[_tab1, _tab2, _tab3])

msg_div = Div(name='msg_div')

save_bn = Button(
    label="Save layout",
    button_type="success", width=200,
)

callback_holder = PreText(
    name='cb_holder',
    text = '',
)