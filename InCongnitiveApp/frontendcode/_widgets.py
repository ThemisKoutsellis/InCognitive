# _widgets.py

from bokeh.plotting import figure
from bokeh.models.widgets import FileInput
from bokeh.models import (
    Div, ColumnDataSource, Button,
    Spinner, CheckboxGroup, Select,
    Panel, Tabs, FactorRange, TableColumn,
    DataTable, BoxZoomTool, PanTool, ResetTool,
    HoverTool, TapTool, WheelZoomTool, SaveTool,
)

from frontendcode._internal_functions import (
    _update_graph_renderer
)

all = ()


PARIS_REINFORCE_COLOR = '#9CAB35'
"""str: The default RGB color of PARIS REINFORCE
"""


# ---------------------------------------------------------------------
# Webpage elements   --------------------------------------------------
# ---------------------------------------------------------------------

# Header Div:
page_header = Div(
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
        '<a href="https://docs.bokeh.org/en/latest/index.html" target=_blank>Bokeh 2.3.3.</a>'
        ),
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# License Div:
license = Div(
    text='License:\
        <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target=_blank>(CC BY-NC-ND 4.0)</a>',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Github repository Div:
github_repo = Div(
    text='<a href="https://github.com/ThemisKoutsellis/InCognitive" target=_blank>GitHub repository</a>',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Separator line Div:
def separator(width=550, height=5):
    return Div(text=f'<hr width="{width}">',
               width=width, height=height)

# Simulation message Div:
alert_msg_div = Div(text='', width=300, name='alert_msg_div')

# Interconnection table message Div:
excel_parse_msg_div = Div(text='', width=300, name='xlsx_msg_div')

# Insert input excel button:
upload_xlsx_wgt = FileInput(accept=".xlsx", multiple=False)

# Node table:
nodes_columns = [
    TableColumn(field="name", title="Node name"),
    TableColumn(field="desc", title="Node description"),
    TableColumn(field="type", title="Node Type [Input/Intermediate/Output]"),
]
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
)
nodes_data_table_title = Div(
    text='FCM nodes',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
)

# Interconnections node table:
edges_columns = [
    TableColumn(field="source", title="Source node"),
    TableColumn(field="target", title="Target node"),
    TableColumn(field="weight", title="Weight"),
]
edges_data_table = DataTable(
    source=current_doc.edges_CDS,
    columns=edges_columns,
    min_height=500,
    max_height=120,
    width=300,
    height = 500,
    editable=True,
    height_policy="fit",
)
edges_data_table_title = Div(
    text='FCM node interconnections',
    width=400,
    style={'font-size': '100%', 'color': PARIS_REINFORCE_COLOR}
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



#Updatate the FCM figure renderers for the 1st time:
(graph_renderer, labels_renderer) = _update_graph_renderer(current_doc.fcm_layout_dict)
fcm_plot.renderers = []
fcm_plot.renderers = [graph_renderer, labels_renderer]


# Widget No.2: Collect global variables
execute_btn = Button(
    label="Execute simulation",
    button_type="success",
    width=550
)



# Simulation hyperparameters   ----------------------------------------

# Widget No.3: Variable inputs: Number of iterations
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
    low=0.001, high=20, step=0.5, value=0.1, width=150,
    disabled=True,
    name='lambda_spinner',
)

# Widget No.12: Set lambda:
lambda_autoselect_rb = CheckboxGroup(
    labels=["Autoselect lambda?"],
    active=[0]
)

# -------------------------

if current_doc.trans_func == 'sigmoid':
    x_range=[0, 1]
elif current_doc.trans_func == 'hyperbolic':
    x_range=[-1, 1]

# -----------------------------------
# Figure 1.: Input nodes
# -----------------------------------
f1 = figure(
   x_range=x_range,
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
    x_range=x_range,
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
    x_range=x_range,
    y_range=FactorRange(),
    height=500,width=900,
    title="Output nodes",
    toolbar_location='right',
    toolbar_sticky = False,
    name='f3'
)
f3.toolbar.logo = None

# ####################################################
tab1 = Panel(child=f1, title="Input nodes")
tab2 = Panel(child=f2, title="Intermediate nodes")
tab3 = Panel(child=f3, title="Output nodes")

tabs = Tabs(tabs=[tab1, tab2, tab3])

lambda_div = Div()

extract_btn = Button(label="Save results", button_type="success", width=550)


