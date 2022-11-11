# main.py

# general imports
import os
import bisect
import operator
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

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
    Circle, MultiLine, Range1d, Band,
)


# import internal modules
from backendcode.xlparse import get_fcm_layout
from backendcode.fcmmc_simulation import monte_carlo_simulation
from backendcode.fcm_layout_parameters import get_nx_graph

from frontendcode._callbacks import *
from frontendcode._internal_functions import (
    _plot_results, _update_graph_renderer, _display_msg, _display_lambda)


PARIS_REINFORCE_COLOR = '#9CAB35'
"""str: The default RGB color of PARIS REINFORCE
"""

# ----------------------------------------------------------------------------------
# Global variables   ---------------------------------------------------------------
# ----------------------------------------------------------------------------------

current_doc = curdoc()

fcm_layout_dict = {
    'source_nodes': [],
    'target_nodes': [],
    'weights': [],
    'nodes_order': [],
    'input_nodes': [],
    'output_nodes': [],
    'node_discription': [],
}

transfer_function = 'sigmoid'

input_iterations = 1
weight_iterations = 1

variance_on_zero_weights = False

sd_inputs = 0.1
sd_weights = 0.1

lamda = None
lamda_autoslect = True

input_nodes_cds = ColumnDataSource()
intermediate_nodes_cds = ColumnDataSource()
output_nodes_cds = ColumnDataSource()

nodes_CSD = ColumnDataSource()
edges_CSD = ColumnDataSource()

output_nodes_mc_values = {}

# ----------------------------------------------------------------------------------
# Supplementary functions   --------------------------------------------------------
# ----------------------------------------------------------------------------------

def _is_there_variation_on_weights(active):
    global sd_weights
    global weight_iterations

    current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=' ',
                msg_type='alert'
            )
    )

    if active:

        weight_sd_spinner.disabled = False
        weight_iterations_spinner.disabled = False

        weight_iterations = 2
        weight_iterations_spinner.value = 2
        weight_sd_spinner.value = 0.1

        weight_iterations_spinner.low = 2

        variable_zero_weights_radio_button.active = []
        variable_zero_weights_radio_button.disabled = False
        variance_on_zero_weights = False

    else:
        sd_weights = 0
        weight_iterations = 1
        weight_iterations_spinner.value = 1

        weight_sd_spinner.disabled = True
        weight_iterations_spinner.disabled = True

        weight_iterations_spinner.low = 1

        variable_zero_weights_radio_button.active = []
        variable_zero_weights_radio_button.disabled = True
        variance_on_zero_weights = False

    return None

def _set_input_sd(attr, old, new):
    global sd_inputs

    current_doc.add_next_tick_callback(
        partial(
            _display_msg,
            doc=current_doc,
            div=alert_msg_div,
            msg=' ',
            msg_type='alert'
        )
    )

    sd_inputs = new

    return None

def _is_there_variation_on_input_nodes(active):
    global sd_inputs
    global input_iterations

    current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=' ',
                msg_type='alert'
            )
    )

    if active:
        input_iterations_spinner.disabled = False
        input_iterations_spinner.value = 2
        input_iterations_spinner.low = 2
        input_iterations = 2

        inputs_sd_spinner.disabled = False
        inputs_sd_spinner.value = 0.1

    else:
        input_iterations_spinner.disabled = True
        input_iterations_spinner.value = 1
        input_iterations_spinner.low = 1
        input_iterations = 1

        inputs_sd_spinner.disabled = True
        sd_inputs = 0

    return None

def _is_zero_weights_variable(active):
    global variance_on_zero_weights

    current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=' ',
                msg_type='alert'
            )
    )

    if active:
        variance_on_zero_weights = True
    else:
        variance_on_zero_weights = False

    return None

def _lambda_autoselect(active):
    global lamda_autoslect

    current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=' ',
                msg_type='alert'
            )
    )

    if active:
        lambda_spinner.disabled = True
        lamda_autoslect = True
        lamda = None

    else:
        lamda_autoslect = False
        lambda_spinner.disabled = False
        lambda_spinner.value = 0.5

    return None


def _set_lambda(attr, old, new):
    global lamda

    current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=' ',
                msg_type='alert'
            )
    )

    lamda = new

    return None

def _set_transfer_function(attr, old, new):
    global transfer_function, f1,f2, f3

    current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=' ',
                msg_type='alert'
            )
    )

    transfer_function = new

    return None

def _clear_msg(attr, old, new):
    alert_msg_div.text = ' '

def _set_weights_sd(attr, old, new):
    global sd_weights

    current_doc.add_next_tick_callback(
            partial(
                _display_msg,
                doc=current_doc,
                div=alert_msg_div,
                msg=' ',
                msg_type='alert'
            )
    )

    sd_weights = new

    return None


# ----------------------------------------------------------------------------------
# Fundamental functions   ----------------------------------------------------------
# ----------------------------------------------------------------------------------

def collect_global_var():

    global fcm_layout_dict
    global transfer_function
    global input_iterations
    global weight_iterations
    global variance_on_zero_weights
    global sd_inputs
    global sd_weights
    global input_nodes_cds
    global intermediate_nodes_cds
    global output_nodes_cds
    global output_nodes_mc_values
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

    error1 = not bool(input_xlsx_wgt.filename)
    error2 = not bool(fcm_layout_dict)
    if not error2:
        error3 = not bool(fcm_layout_dict['source_nodes'])
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
        _expr1 = input_iterations == weight_iterations
        _expr2 = (input_iterations < 2) or (weight_iterations < 2)

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
                fcm_layout_dict,
                input_iterations,
                weight_iterations,
                sd_inputs,
                sd_weights,
                variance_on_zero_weights,
                transfer_function,
                lamda,
                lamda_autoslect,
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

            if transfer_function == 'sigmoid':
                _x = list(np.linspace(0, 1, N))
                bisect.insort(_x, 0.5)
                _set_x_range(0,1)

            elif transfer_function == 'hyperbolic':
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
            current_doc.add_next_tick_callback(
                partial(
                    _display_msg,
                    doc=current_doc,
                    div=alert_msg_div,
                    msg='[ALERT]: The number of iterations (Weight & Input) must be equal!',
                    msg_type='alert'
                )
            )
            pass

    return None


# ----------------------------------------------------------------------------------
# Webpage elements   ---------------------------------------------------------------
# ----------------------------------------------------------------------------------

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
alert_msg_div = Div(text='', width=300)

# Interconnection table message Div:
excel_parse_msg_div = Div(text='', width=300)

# Insert input excel button:
input_xlsx_wgt = FileInput(accept=".xlsx", multiple=False)

# Node table:
nodes_columns = [
    TableColumn(field="name", title="Node name"),
    TableColumn(field="desc", title="Node description"),
    TableColumn(field="type", title="Node Type [Input/Intermediate/Output]"),
]
nodes_data_table = DataTable(
    source=nodes_CSD,
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
    source=edges_CSD,
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


# FCM topology figure:
# --------------------

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

fcm_plot.on_change('renderers', _clear_msg)

#Updatate the FCM figure renderers for the 1st time:
(graph_renderer, labels_renderer) = _update_graph_renderer(fcm_layout_dict)
fcm_plot.renderers = []
fcm_plot.renderers = [graph_renderer, labels_renderer]


# Widget No.2: Collect global variables
execute_btn = Button(label="Execute simulation", button_type="success", width=550)
execute_btn.on_click(collect_global_var)


# Simulation hyperparameters   ---------------------------------------------

# Widget No.3: Variable inputs: Number of iterations
input_iterations_spinner = Spinner(
    title="Number of MC iterations (variable inputs):",
    low=1, high=1000000, step=1, value=1, width=300,
    disabled = True,
)

# Widget No.4: Standard deviation (variable inputs):
inputs_sd_spinner = Spinner(
    title="Standard deviation (variable inputs)",
    low=0, high=1, step=0.05, value=0, width=210,
    disabled=True,
)
inputs_sd_spinner.on_change('value', _set_input_sd)


# Widget No.5: Input nodes variation:
variable_input_nodes_radio_button = CheckboxGroup(labels=["Input nodes variation"], active=[])
variable_input_nodes_radio_button.on_click(_is_there_variation_on_input_nodes)




# Widget No.6: Standard deviation (variable inputs)
weight_sd_spinner = Spinner(
    title= 'Standard deviation (variable weights)',
    low=0, high=1, step=0.05, value=0.1, width=210,
    disabled=True,
)
weight_sd_spinner.on_change('value', _set_weights_sd)


# Widget No.7: Variance on zero weights
LABELS = ["Variance on zero weights?"]
variable_zero_weights_radio_button = CheckboxGroup(labels=LABELS, active=[], disabled=True)
variable_zero_weights_radio_button.on_click(_is_zero_weights_variable)


# Widget No.8: Variable weights: Number of iterations:
weight_iterations_spinner = Spinner(
    title="Number of Monte Carlo iterations (variable weights):",
    low=1, high=1000000, step=1, value=1, width=300,
    disabled=True,
)

# Widget No.9: Weights variation:
# -------------------------------------------------
variable_weights_radio_button = CheckboxGroup(labels=["Weights variation"], active=[])
variable_weights_radio_button.on_click(_is_there_variation_on_weights)

# Widget No.10: Select transfer function
tr_function_select = Select(
    title="Transfer function:", value="Sigmoid",
    options=["sigmoid", "hyperbolic",], width=150)
tr_function_select.on_change("value", _set_transfer_function)

# Widget No.11: Set labbda:
lambda_spinner = Spinner(
    title= 'Set lambda value',
    low=0.001, high=20, step=0.5, value=0.1, width=150,
    disabled=True,
)
lambda_spinner.on_change('value', _set_lambda)

# Widget No.12: Set lambda:
lambda_autoselect_radio_button = CheckboxGroup(labels=["Autoselect lambda?"], active=[0])
lambda_autoselect_radio_button.on_click(_lambda_autoselect)

# -------------------------

if transfer_function == 'sigmoid':
    x_range=[0, 1]
elif transfer_function == 'hyperbolic':
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
)
f3.toolbar.logo = None

# ####################################################
tab1 = Panel(child=f1, title="Input nodes")
tab2 = Panel(child=f2, title="Intermediate nodes")
tab3 = Panel(child=f3, title="Output nodes")

tabs = Tabs(tabs=[tab1, tab2, tab3])

lambda_div = Div()

extract_btn = Button(label="Save results", button_type="success", width=550)

# ----------------------------------------------------------------------------------
# Webpage layout  --- --------------------------------------------------------------
# ----------------------------------------------------------------------------------

# FCM plot and tables display:
fcm_display_layout = layout(
    row(
        fcm_plot,
        column(
            separator(width=550, height=15),
            input_xlsx_wgt,
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
    variable_input_nodes_radio_button,
    [input_iterations_spinner, inputs_sd_spinner],
    separator(width=550, height=15)
)
weights_layout = layout(
    separator(width=550, height=15),
    variable_weights_radio_button,
    [weight_iterations_spinner, weight_sd_spinner],
    variable_zero_weights_radio_button,
    separator(width=550, height=15)
)
lambda_layout = layout(
    separator(width=550, height=15),
    lambda_autoselect_radio_button,
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

# ----------------------------------------------------------------------------------
# Run bokeh server   ---------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Append web page layout to curent bokeh layout
current_doc.add_root(web_page_layout)

os.system('bokeh serve --show ./')



# ----------------------------------------------------------------------------------
# Assign callbacks on widgets    ---------------------------------------------------
# ----------------------------------------------------------------------------------
input_xlsx_wgt.on_change('value', _get_xlsx)

weight_iterations_spinner.on_change(
    'value',
    partial(
        _set_iter_when_weights_vary,
        doc=current_doc, div=alert_msg_div,
        var_zero_weights_rd_btn=variable_zero_weights_radio_button,
        weight_sd_spinner=weight_sd_spinner,
    ),

)

input_iterations_spinner.on_change(
    'value',
    partial(
        _set_iterations_when_inputs_vary,
        doc=current_doc,
        div=alert_msg_div,
        inputs_sd_spinner=inputs_sd_spinner,
    )
)