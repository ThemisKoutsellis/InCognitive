# fcm_simulator.py

"""This module provides the 'execute_siluation' function which
deploys the FCM simulation [1].

Each value of the FCM nodes is derived by the iterative equation:
A^(k+1) = f(x^(k+1)) [1],
where A^(k+1) is the node value at (k+1) iteration, f the FCM
transfer function and x^(k+1) the summation of all contributions
of all connected nodes to the current node. The x^(k+1) is a function
of the weight matrix and the past values of all connected nodes [1].

When k=0, the essamble of all node values is called initial FCM
state vector or Ao.

The simulation stops either right after the number of iteration
is greater than the ITERATIONS attribute of the FCMap object or
at that iteration where there is convergence for all nodes,
|A^(k+2)-A^(k+1)|<ε).

* [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

"""

import pandas as pd

__all__ = ('exec_fcm_simulation')

#######################################################################
def _fcm_generator(
    A,
    Arguments,
    lamda,
    F,
    w_matrix,
    lag_matrix,
    nodes_order,
    input_nodes,
    output_nodes,
    _ITERATIONS,
    MIN_NUM_OF_ITERATIONS,
    ):

    j = 0
    _error = 10
    _error_threshold = 0.0001

    while (j<=MIN_NUM_OF_ITERATIONS+1) or ((_error>=_error_threshold) and (j<_ITERATIONS)):

        # initialize A_next vector:
        A_next = {v:None for i, v in enumerate(nodes_order)}
        _Arguments = {v:None for i, v in enumerate(nodes_order)}

        # Calculate A_next vector:
        for key in A.keys():

            ## Find the previous lag A values:
            _index = nodes_order.index(key)
            _row_i = lag_matrix[_index, :].tolist()[0]

            _len = len(list(A.values())[0])
            _lag = {
                v: int(_row_i[i]) if _row_i[i] < _len else _len \
                for i, v in enumerate(nodes_order)
            }
            # initialize A_prev_with_lags:
            A_prev_with_lags = {k:v[-_lag[k]] for k, v in A.items()}

            # Calculate A_next:
            if input_nodes:
                if key in input_nodes:
                    A_next[key] = A_prev_with_lags[key]
                else:
                    _index = nodes_order.index(key)
                    _row_i = w_matrix[_index, :].tolist()[0]
                    _w = {v: float(_row_i[i]) for i, v in enumerate(nodes_order)}

                    _argument = sum([_w[k]*v for k, v in A_prev_with_lags.items()])
                    A_next[key] = F(lamda, _argument)
                    _Arguments[key] = _argument
            else:
                _index = nodes_order.index(key)
                _row_i =w_matrix[_index, :].tolist()[0]
                _w = {v: float(_row_i[i]) for i, v in enumerate(nodes_order)}
                _argument = sum([_w[k]*v for k, v in A_prev_with_lags.items()])
                A_next[key] = F(lamda, _argument)
                _Arguments[key] = _argument

        # Append A_next to A:
        [A[key].append(A_next[key]) for key in A.keys()]

        # Append _Arguments to Arguments:
        [Arguments[key].append(_Arguments[key]) for key in A.keys()]

        # Estimate the error between A_next, A_prev_with_lags:
        no_input_nodes = [x for x in nodes_order if x not in input_nodes]
        _error = max([abs(A[key][-1] - A[key][-2]) for key in no_input_nodes])
        # next j-th iteration:
        j+=1
        # yielded value:
        yield A

#######################################################################
def _normalise_outcomes(
    intermediate_df,
    output_df,
    activation_function_name,
    lamda,
    intermediate_x_f,
    output_x_f,
    ):

    if activation_function_name=='sigmoid':
        # Intermediate nodes
        if not intermediate_df.empty:
            _last_row = intermediate_df.tail(1).values[0]
            _norm_last_row = [
                v  +((0.09*lamda)*intermediate_x_f[i])
                for i,v in enumerate(_last_row)
            ]
            normilised_intermediate_df = intermediate_df
            normilised_intermediate_df.iloc[-1,:] = _norm_last_row
        else:
            normilised_intermediate_df = pd.DataFrame()

        # Output nodes
        if not output_df.empty:
            _last_row = output_df.tail(1).values[0]
            _norm_last_row = [
                v  +((0.09*lamda)*output_x_f[i])
                for i,v in enumerate(_last_row)
            ]
            normilised_output_df = output_df
            normilised_output_df.iloc[-1,:] = _norm_last_row
        else:
            normilised_output_df = pd.DataFrame()
    elif activation_function_name=='hyperbolic':
        # Intermediate nodes
        if not intermediate_df.empty:
            normilised_intermediate_df = intermediate_df*1.733
        else:
            normilised_intermediate_df = pd.DataFrame()
        # Output nodes
        if not output_df.empty:
            normilised_output_df = output_df*1.733
        else:
            normilised_output_df = pd.DataFrame()

    return normilised_intermediate_df, normilised_output_df

#######################################################################
def _create_outcomes_dframes(nodes_order, input_nodes, output_nodes, A):

    intermediate_nodes = \
        [x for x in nodes_order \
         if (x not in input_nodes) and (x not in output_nodes) ]

    if input_nodes:
        input_data_dict = {i:A[i] for i in input_nodes}
    else:
        input_data_dict = {}

    if output_nodes:
        output_data_dict = {i:A[i] for i in output_nodes}
    else:
        output_data_dict = {}

    if intermediate_nodes:
        intermediate_data_dict = {i:A[i] for i in intermediate_nodes}
    else:
        intermediate_data_dict = {}

    # Data frames:
    input_df = pd.DataFrame.from_dict(input_data_dict)
    intermediate_df = pd.DataFrame.from_dict(intermediate_data_dict)
    output_df = pd.DataFrame.from_dict(output_data_dict)

    return input_df, intermediate_df, output_df

#######################################################################
def exec_fcm_simulation(
    A,
    Arguments,
    input_nodes,
    intermediate_nodes,
    output_nodes,
    MIN_NUM_OF_ITERATIONS,
    activation_function_ref,
    activation_function_name,
    lambda_autoselect,
    lamda,
    w_matrix,
    lag_matrix,
    nodes_order,
    ITERATIONS,
    ):
    """This public function deploys the FCM simulation [1].

    Firstly, it 'consumes' the iternal generator which returns the
    final A dictionary. The A dict (see also Parameters section)
    contains all  A^k [1] per node for all iterations of the FCM
    simulation. Then the results are normilised through the internal
    function '_normalise_outcomes' (see method in [2]) and besed on
    this normalisation the final dataframes are derived through the
    internal function '_create_outcomes_dframes'.

    Parameters
    ----------
    A : dict
        a collection of lists with the node values A^k per node
        for all iterations of the FCM simulation [1]. Initially,
        each list contains one element, which corresponds to the
        initial node velue of the corresponding node.
    Arguments : dict
        a collection of lists with the x^k arguments per node
        for all iterations of the FCM simulation.
        Initially, each list contains one zero element,[0].
        It is used for the normilisation procedure of [2].
    input_nodes : list
        a list with the input nodes of the FCM.
    intermediate_nodes : dict
        a list with the intermediate nodes of the FCM.
    output_nodes : dict
        a list with the output nodes of the FCM.
    MIN_NUM_OF_ITERATIONS : dict
        the minimum iterations of the FCM simulation. It is equal to
        the min path of the FCM laytout (from a arbitrary input node
        to an arbitrary output node).
    activation_function_ref : memory ref
        a memory reference to the choosen tranfer function
        of the FCM. The transfer functions are deployed by
        to the 'activation_function' module.
    activation_function_name : str
        the accepted values are 'sigmoid' and 'hyperbolic'.
    lambda_autoselect : bool
        True of the user wants the app to choose the lambda
        paremeter of the FCM tranfer function based on [1].
    lamda : float
        the lambda parameter value to be used if lambda_autoselect=False
    w_matrix : numpy array
        the weight matrix of the FCM layout.
    lag_matrix : dict
        the weight matrix of the FCM layout.
    ITERATIONS : dict
        The max number of FCM simulation iteration [1]. After
        this number the function stops the FCM simulation and
        returns the node values as is.

    Returns
    -------
        normilised_output_final_values : dict
            a dictionary which contains the normilised final values
            of the output nodes
        normilised_intermediate_final_values : dict
            a dictionary which contains the normilised final values
            of the intermediate nodes
        normilised_intermediate_df : pandas dataframe
            a dataframe with all values of the FCM simulation
            values [1] per intermediate nodes.
        normilised_output_df : pandas dataframe
            a dataframe with all values of the FCM simulation
            values [1] per output nodes


    * [1] 'Parameter analysis for sigmoid and
    hyperbolic transfer functions of fuzzy cognitive maps',
    https://doi.org/10.1007/s12351-022-00717-x

    * [2] 'Normalising the Output of Fuzzy Cognitive Maps'
    IISA-2022 Confernece.
    """

    # Consume fcm_generator
    for gen in _fcm_generator(
        A,
        Arguments,
        lamda,
        activation_function_ref,
        w_matrix,
        lag_matrix,
        nodes_order,
        input_nodes,
        output_nodes,
        ITERATIONS,
        MIN_NUM_OF_ITERATIONS,
    ):
        pass

    x_f = [Arguments[k][-1] for k, v in Arguments.items()]
    # get the argument value of the last
    # itaration for the intermediate nodes
    intermediate_x_f = [x_f[i] for i,v in enumerate(nodes_order)\
        if (v not in (input_nodes or output_nodes))]
    # get the argument value of the
    # last itaration for the output nodes
    output_x_f = [x_f[i] for i,v in enumerate(nodes_order)\
        if (v in output_nodes) ]
    # create the outcome data frames
    (
    input_df,
    intermediate_df,
    output_df
    ) = _create_outcomes_dframes(
        nodes_order,
        input_nodes,
        output_nodes,
        A,
    )
    # normalise the outcome dataframes:
    (normilised_intermediate_df,
    normilised_output_df
    ) = _normalise_outcomes(
        intermediate_df,
        output_df,
        activation_function_name,
        lamda,
        intermediate_x_f,
        output_x_f,
    )
    # derive the final values
    final_A_values= {}
    for key, value in A.items():
        final_A_values[key] = value[-1]
    if not normilised_intermediate_df.empty:
        norm_inter_final_values_list = \
            list(normilised_intermediate_df.iloc[-1])
    else:
        norm_inter_final_values_list = []
    if not normilised_output_df.empty:
        norm_output_final_values_list = list(normilised_output_df.iloc[-1])
    else:
        norm_output_final_values_list = []
    normilised_intermediate_final_values = \
        {intermediate_nodes[count]:ele \
            for count, ele in enumerate(norm_inter_final_values_list)
        }
    normilised_output_final_values = \
        {output_nodes[count]:ele \
            for count, ele in enumerate(norm_output_final_values_list)
        }

    return (
        normilised_output_final_values,
        normilised_intermediate_final_values,
        normilised_intermediate_df,
        normilised_output_df,
    )
