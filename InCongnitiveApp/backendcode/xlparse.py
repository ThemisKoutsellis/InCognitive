# xlparse.py

"""This module provides a function, 'get_fcm_layout', which
returns one of the core dictionaries, 'fcm_layout_dict', of
the InCognitive application.

The 'fcm_layout_dict' dict is imported through many backend
modules of the InCognitive app and contains most of the
necessary info regarding the FCM topology and parameters,
as provided by the user. It can be updtated either from the
input excel files or by the app's GUI.


INPUT EXCEL FILE'S FORMAT:
-------------------------

The input excel file must have a certain type of format:
It consists of three (3) sheets with the following names:
    1st sheet name : 'nodes-order'
    2nd sheet name : 'input-output-nodes'
    3rd sheet name : 'fcm-topology'

1st sheet:
----------
The 1st sheet provides info on all the FCM nodes regardless of
being input, intermediate or output nodes.
It consists of five (5) columns:
    col1 : 'nodes order'
        The name of the nodes by the desired order. The order of cols
        and rows of the weight matrix are consistent to this order.
    col2 : 'node description'
        A brief description of each node i.e. its system functionality
    col3 : 'initial value'
        The initial node values before the FCM simulation. For further
        info see https://doi.org/10.1007/s12351-022-00717-x
    col4 : 'auto weights'
        The auto weight expresses the degree of correlation of the
        node's current value with its past value. The position of past
        value in time domain, t-lag, is defined by the lag parameter.
    col5 : 'auto lags'
        If the auto-weight is non-zero, the auto-lag defines the
        time interval between the node's current value end the
        correspinding past one by which it depends on.

2nd sheet:
----------
The 2nd sheet provides info on which are the input and output nodes.
It consist of two (2) columns:
    col1 : 'input nodes'
        The name of input nodes. If there are not input nodes,
        leave the col empty.
    col2 : 'output nodes'
        The name of output nodes. If there are not output nodes,
        leave the col empty.

3rd sheet:
----------
The 3rd sheet provides info on the FCM edges. Each edge is characterized
by the 'source' and 'target' node; i.e. the beginning and ending node,
respectivelly. Moreover, there is a weight value assigned on each edge.
This weight value indicates the correlation of the target node's value
of time instance t with the source node's value of time instance t-lag.

The layout of the 3rd sheet consist of forth (4) columns:
    col1 : 'source node'
        The source node, the beginning node, of the edge.
    col2 : 'target node'
        The target node, ending node, of the edge.
    col3 : 'weight'
        the weight of the edge.
    col4 : 'lag'
        the correlation lag between the target and source node values.

NOTE: The excel sheets are passed to the 'get_fcm_layout' function as
pandas dataframes parameter. These dataframers have the same layout as
the excel sheets.

"""


import pandas as pd

__all__ = ('get_fcm_layout',)


def get_fcm_layout(df_nodes_order, df_fcm_topology, df_in_out_nodes):
    """This function provides the core dictionary of the InCognitive
    app, the 'fcm_layout_dict'. It contains the most critical info of
    the FCM layout. It gets three (3) dataframes (df) as parameters.

    Parameters
    ----------
    df_nodes_order : dataframe
        The dataframe related to the 1st sheet of the input excel file.
    df_fcm_topology : dataframe
        The dataframe related to the 2nd sheet of the input excel file.
    df_in_out_nodes : dataframe
        The dataframe related to the 3rd sheet of the input excel file.

    Returns
    -------
    dict
        the 'fcm_layout_dict' dictionary with the following keys:
            1. 'nodes_order'
            2. 'nodes_discription'
            3. 'auto_weights'
            4. 'auto_lags'
            5. 'initial_values'
            6. 'input_nodes'
            7. 'output_nodes'
            8. 'source_nodes'
            9. 'target_nodes'
            10. 'weights'
            11. 'lags'

        """

    # initialize fcm_layout_dict dctionary
    fcm_layout_dict = {}

    # Fill the dictionary field values
    fcm_layout_dict["nodes_order"] = list(df_nodes_order['nodes order'])
    fcm_layout_dict["nodes_order"] = [
        x for x in fcm_layout_dict["nodes_order"] if pd.isnull(x) == False]

    fcm_layout_dict["nodes_discription"] = {
        v:list(df_nodes_order['node description'])[i] \
        for i,v in enumerate(fcm_layout_dict["nodes_order"])
    }
    fcm_layout_dict["nodes_discription"] = [
        x for x in fcm_layout_dict["nodes_discription"] if pd.isnull(x) == False]

    fcm_layout_dict["auto_weights"] = list(df_nodes_order['auto weights'])
    fcm_layout_dict["auto_weights"] = [
        x for x in fcm_layout_dict["auto_weights"] if pd.isnull(x) == False]

    fcm_layout_dict["auto_lags"] = list(df_nodes_order['auto lags'])
    fcm_layout_dict["auto_lags"] = [
        x for x in fcm_layout_dict["auto_lags"] if pd.isnull(x) == False]

    fcm_layout_dict["initial_values"] = list(df_nodes_order['initial value'])
    fcm_layout_dict["initial_values"] = [
        x for x in fcm_layout_dict["initial_values"] if pd.isnull(x) == False]

    fcm_layout_dict["input_nodes"] = list(df_in_out_nodes['input nodes'])
    fcm_layout_dict["input_nodes"] = [
        x for x in fcm_layout_dict["input_nodes"] if pd.isnull(x) == False]

    fcm_layout_dict["output_nodes"] = [
        x for x in list(df_in_out_nodes['output nodes']) if pd.isnull(x) == False]
    fcm_layout_dict["output_nodes"] = [
        x for x in fcm_layout_dict["output_nodes"] if pd.isnull(x) == False]

    fcm_layout_dict["source_nodes"] = list(df_fcm_topology['source node'])
    fcm_layout_dict["source_nodes"] = [
        x for x in fcm_layout_dict["source_nodes"] if pd.isnull(x) == False]

    fcm_layout_dict["target_nodes"] = list(df_fcm_topology['target node'])
    fcm_layout_dict["target_nodes"] = [
        x for x in fcm_layout_dict["target_nodes"] if pd.isnull(x) == False]

    fcm_layout_dict["weights"] = list(df_fcm_topology['weight'])
    fcm_layout_dict["weights"] = [
        x for x in fcm_layout_dict["weights"] if pd.isnull(x) == False]

    fcm_layout_dict["lags"] = list(df_fcm_topology['lag'])
    fcm_layout_dict["lags"] = [
        x for x in fcm_layout_dict["lags"] if pd.isnull(x) == False]

    return fcm_layout_dict
