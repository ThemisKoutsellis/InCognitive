# fcmmc_simulation.py

"""This module provides the 'monte_carlo_simulation' function which
deploys the combination of FCM Simulation [1] and Monte Carlo
Simulation (MCS).

* [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

"""

import numpy as np
from backendcode.fcm_object import FCMap
from backendcode.fcmmc_object import MCarloFcm

__all__ = ('monte_carlo_simulation')

#######################################################################
def monte_carlo_simulation(
    fcm_layout_dict,
    inputs_iterations,
    weights_iterations,
    sd_inputs,
    sd_weights,
    variance_on_zero_weights,
    transfer_func,
    lamda,
    lamda_autoslect,
):

    """

    Parameters
    ----------
    fcm_layout_dict : dict
        the core dictionary which contains all the necessary
        info for the FCM layout.
    inputs_iterations :
        number of Monte Carlo iterations considering that
        the values of the input nodes are random variables.
        Instead, if inputs_iterations=0 or inputs_iterations=1
        the values of input nodes are constants.
    weights_iterations :
        the number of Monte Carlo iterations considering
        that the weigth values are random variables. Instead,
        if weights_iterations=0 or weights_iterations=1 the
        values of weights are constants.
    sd_inputs :
        the standard deviation (sd) of the input nodes values.
        zero if the values of input nodes are considered constant.
    sd_weights :
        the standard deviation (sd) of the weights values.
        zero if the weights are considered constant.
    variance_on_zero_weights : bool
        default valued = True. if True the zero weights
        are considered random distriution with mean value
        zero. otherwies, if False, the weights are constant
        and the zero value indicate non correlation between
        the nodes of the edge.
    transfer_func : str
        the accepted values are 'sigmoid' and 'hyperbolic'.
    lamda :
        the lambda parameter value to be used if
        lambda_autoselect=False
    lamda_autoslect:
        True of the user wants the app to choose the lambda
        paremeter of the FCM tranfer function based on [1].

    Returns
    -------
      mc_lambda : float
          the lambda parameter of the transfer function as
          estimated by the __init__ method of MCarloFcm object.
      input_nodes_values :
          the ensamble of the final normilised [2] values
          per MCS iteration of the input nodes.
      output_nodes_values :
          the ensamble of the final normilised [2] values
          per MCS iteration of the output nodes.
      intermediate_nodes_values :
          the ensamble of the final normilised [2] values
          per MCS iteration of the intermediate nodes.
      baseline_input_nodes_values : dict
            a dictionary which contains the normilised [2] final
            values of the input nodes when all FCM parameters
            (values of input node and weights) are constant.
      baseline_output_nodes_values : dict
            a dictionary which contains the normilised [2] final
            values of the output nodes when all FCM parameters
            (values of input node and weights) are constant.
      baseline_intermediate_nodes_values : dict
            a dictionary which contains the normilised [2] final
            values of the intermediate nodes when all FCM parameters
            (values of input node and weights) are constant.

    * [1] 'Parameter analysis for sigmoid and
    hyperbolic transfer functions of fuzzy cognitive maps',
    https://doi.org/10.1007/s12351-022-00717-x

    * [2] 'Normalising the Output of Fuzzy Cognitive Maps'
    IISA-2022 Confernece.
    """

    # check if WEIGHTS_ITERATIONS and INPUTS_ITERATIONS constants match
    _weights_iterations_warning = (
        '\nFcm obj destroyed. '
        'Weights iterations is {}'.format(weights_iterations)
    )
    if weights_iterations<1:
        raise Exception(_weights_iterations_warning)

    _inputs_iterations_warning = (
        '\nFcm obj destroyed. '
        'Inputs iterations is {}'.format(inputs_iterations)
    )
    if inputs_iterations<1:
        raise Exception(_inputs_iterations_warning)

    _iteration_dont_match_warning = (
        '\nFcm obj destroyed. '
        'Provided iterations (inputs, weights) do not match!'
    )
    if (weights_iterations>1 and  inputs_iterations>1) and\
        (not(weights_iterations == inputs_iterations)):
        raise Exception(_iteration_dont_match_warning)


    fcm = FCMap(fcm_layout_dict, transfer_func)

    fcm_mc = MCarloFcm(
      inputs_iterations,
      weights_iterations,
      sd_inputs,
      sd_weights,
      fcm,
      lamda,
      lamda_autoslect,
      variance_on_zero_weights,
    )

    fcm_mc.fcm_mc_execute(fcm_layout_dict, )

    # prepare the output variables
    mc_lambda = fcm_mc.lamda
    input_nodes_values = fcm_mc.input_nodes_values
    output_nodes_values = fcm_mc.output_nodes_values
    intermediate_nodes_values = fcm_mc.intermediate_nodes_values

    baseline_output_nodes_values = fcm_mc.baseline_output_node_values
    baseline_intermediate_nodes_values = fcm_mc.baseline_intermediate_node_values
    baseline_input_nodes_values = fcm_mc.baseline_input_node_values

    # del all objects
    del fcm
    del fcm_mc

    return (
      mc_lambda,
      input_nodes_values,
      output_nodes_values,
      intermediate_nodes_values,
      baseline_input_nodes_values,
      baseline_output_nodes_values,
      baseline_intermediate_nodes_values,
    )

