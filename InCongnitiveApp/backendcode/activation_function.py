# activation_function.py

"""This module provides the transfer functions of FCMs.

It returns a dictionary, 'act_functions', with memory
referneces of the impliment tranfer functions.

Current implimented tranfer function are:
    1. Sigmoid,
    2. Hyperbolic tangent.

"""

__all__ = ('act_functions')

import math

## Sigmoid function:
def _sigmoid(lamda, argument):
    """The sigmoid function : 1/(1+exp(-λx))

    Parameters
    ----------
    lamda : float
        the lambda parameter of sigmoid function.
    argument : float
        the x argument of sigmoid function.

    Returns
    -------
    float
        the value of sigmoid function at x, given λ.

    """
    try:
        exp = math.exp(-(lamda*argument))
    except OverflowError:
        exp = float('inf')

    return 1/(1 + exp)

## Hyperbolic tangent function:
def _hyperbolic(lamda, argument):
    """The hyperbolic tangentfunction :
    [exp(λx)-exp(-λx)]/[exp(λx)+exp(-λx)]

    Parameters
    ----------
    lamda : float
        the lambda parameter of hyperbolic tangent function.
    argument : float
        the x argument of hyperbolic tangent function.

    Returns
    -------
    float
        the value of hyperbolic tanget function at x, given λ.

    """
    try:
        exp1 = math.exp(lamda*argument)
        exp2 = math.exp(-lamda*argument)
    except OverflowError:
        exp1 = float('inf')
        exp2 = float('inf')

    return (exp1-exp2)/(exp1+exp2)

act_functions = {
    'sigmoid': _sigmoid,
    'hyperbolic': _hyperbolic,
}
