# fcm_layout_parameters.py

"""This module contains functions which return
some necessary parameters of FCMs.

These parameters are:
    1. the weight matrix, W, of a FCM,
    2. the lag matrix, L, of a FCM and
    3. the lambda parameter of the transfer function of a FCM.

The lambda parameter is estimated based on the methodology of [1]

The module also impliments a function, get_nx_graph(), which
provides the FCM in a networkX object format.

* [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

"""

__all__ = (
    'get_nx_graph',
    'get_w_matrix',
    'get_lag_matrix',
    'select_lambda',
)

# ΝetworkΧ
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix

# Maths
import numpy as np
from numpy import linalg as LA


def get_nx_graph(source, target, weight):
    """This function provides a networkX represenation of a FCM.

    It gets three arguments, i.e. three list of the same length
    and returns a networkX object. The first two lists are describe
    the edges of the FCM. The 1st list, the 'source' list, is a
    collection of the beginning nodes of the directed FCM edges.
    The 2nd list, the 'target' list, contains the corresponding
    ending nodes of the FCM directed edges. Finally, the 3rd list,
    the 'weights' list, contains the corresponding weights of each
    edge. Example:
    source_list = [1, 2, 4]
    target_list = [3, 4, 3]
    weights_list = [0.5, 0.6, 0.3]
    The 1st edge connects the node 1 with 3 and the corresponding
    weight is 0.5.

    Parameters
    ----------
    source : list
        the source nodes.
    target : list
        the target nodes.
    weight : list
        the weights.

    Returns
    -------
    DiGraph
        a NetworkX DiGraph object.

    """
    # creates a list of tuples with all necessary
    # elements for a networkX graph.
    edges_list = list(zip(source, target, weight))

    # creates the netwokX graph:
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(edges_list)

    return nx_graph

def get_w_matrix(nx_graph, nodes_order, input_nodes, auto_w):
    """This function provides the weight matrix, W, of a FCM.

    Typically, an element of W matrix, w_ij, corresponds to the
    directed inter-connection between the source node C_i and the
    target node C_j. Therefore, the rows of W matrix corresponds
    to the source nodes and the columns to the target nodes.
    The range of w_ij values is [-1, 1] and describe the degree
    the values of node C_j are affected by the values of node C_i.

    The first parameter, the 'nx_grpah', is a networkX object which
    contains all necessery information regarding the edges and
    the corresponding weights of the FCM.

    The order of the rows and columns of W matrix is determined
    by the second parameter, the 'nodes_order'.
    Example: Given a FCM with three nodes, the C1, C2 and C3, if
    nodes_order = ['C3', 'C1', 'C2'] then the first row of W matrix,
    [w_11, w_12, w_13], contains weights which correspond to the edges
    [C3->C3, C3->C1, C3->C2], respectively. The second row,
    [w_21, w_22, w_23] contains weights which correspond to the edges
    [C1->C3, C1->C1, C1->C2], respectively. Finally, the third row,
    [w_31, w_32, w_33] contains weights which correspond to the
    edges [C2->C3, C2->C1, C2->C2], respectively.

    The third parameter, the 'input_nodes', is neccesary to assign
    weights equal to one to all w_ii elements which correspond to the
    input nodes. By doing so, the input nodes maintain their initial
    value constant during the FCM simulation [1].

    The forth parameter, the 'auto_w', defines the values of w_ii elements.
    The w_ii elements define the degree of correlation between present
    and past values of the same node.

    It should be noted that the returned W matrix is the transpose
    of the typical W matrix, as described above, to comply with [1].

    Parameters
    ----------
    nx_graph : DiGraph
        A NetworkX DiGraph object.
    nodes_order : list of str
        The order of nodes.
    input_nodes : list of str
        If there are not any input nodes the nodes_order = [].
    auto_w: list of auto-weights
        A list of the same lenght as the nodes_order list

    Returns
    -------
    array
        A SciPy sparse matrix.

    * [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
    of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

    """
    w_matrix = adjacency_matrix(
        nx_graph,
        nodelist=nodes_order,
        weight='weight',
    )

    # The corresponding definition of DOI
    w_matrix = np.transpose(w_matrix)
    w_matrix = w_matrix.todense()

    # Add auto-weight to weight array:
    for i, ele in enumerate(nodes_order):
        w_matrix[i,i] = auto_w[i]

    # w_ii element of input nodes equal to one
    for i, v in enumerate(nodes_order):
        if v in input_nodes:
            w_matrix[i,i] = 1

    return w_matrix

def get_lag_matrix(source, target, lag, nodes_order, auto_lag):
    """This function provides the lag matrix of a FCM.

    Parameters
    ----------
    source : list
        The source nodes.
    target : list
        The target nodes.
    lag : list
        The lag per edge.
    nodes_order : list
        A list of str with the order of nodes.
    auto_lag : list
        The l_ii lags.

    Returns
    -------
    array
        A SciPy sparse matrix.

    """
    # create list of tuples with all necessary elements for graph
    lag_list = list(zip(source, target, lag))

    # create lag netwokx graph:
    nx_lag_graph = nx.DiGraph()
    nx_lag_graph.add_weighted_edges_from(lag_list)

    # create lag array:
    lag_matrix = adjacency_matrix(nx_lag_graph, nodelist=nodes_order, weight='weight')
    lag_matrix = np.transpose(lag_matrix)
    lag_matrix = lag_matrix.todense()

    # Add auto-lag to lag array:
    for i, ele in enumerate(nodes_order):
        lag_matrix[i,i] = auto_lag[i]

    return lag_matrix

def _calc_lambdas(W_star, variable_weights=False, var_on_zero_weights=False):
    """ This iternal function calculates all necessary norms of [1]
    and then the corresponding lambdas:
        1. l_s_prime,
        2. l_h_prime,
        3. l_s_star,
        4. l_h_star.
    It can also handle the case the weights are random variables.
    In this case, the values of the weight matrix are the mean
    values of the ramdom distribution.

    Parameters
    ----------
    W_star : numpy array
        The FCM weight matrix for the non-input nodes.
    variable_weights : bool
        True if the weights are considered random variables. This
        functionality is necessary when the FCMs are combined with
        the Monte Carlo Simulation.
    var_on_zero_weights : bool
        False if the zero value indicates no connection between the
        source and target node. True if there is a correlation between
        source and target node and the weights are considered random
        variables too. In this case, the weight's random distribution
        has zero mean.

    Returns
    -------
    float
        all lambdas of :
            1. l_s_prime
            2. l_h_prime
            3. l_s_star
            4. l_h_star

    * [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
    of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x
    """

    #a. W-2 norm
    if variable_weights:
        if var_on_zero_weights:
            _W_star = np.ones((W_star.shape[0], W_star.shape[1]))
            w_norm_2 = LA.norm(_W_star, ord='fro')
        else:
            _W_star = np.where(W_star !=0, 1, W_star)
            w_norm_2 = LA.norm(_W_star, ord='fro')
    else:
        w_norm_2 = LA.norm(W_star, ord='fro')

    #b. W-inf norm:
    if variable_weights:
        if var_on_zero_weights:
            _W_star = np.ones((W_star.shape[0], W_star.shape[1]))
            w_norm_inf = LA.norm(_W_star, ord=np.inf)
        else:
            _W_star = np.where(W_star !=0, 1, W_star)
            w_norm_inf = LA.norm(_W_star, ord=np.inf)
    else:
        w_norm_inf = LA.norm(W_star, ord=np.inf)

    #c. W-s norm:
    def separate_neg_pos(mixed_list):
        pos = [i for i in mixed_list if i > 0]
        neg = [i for i in mixed_list if i < 0]

        return pos, neg

    def row_s_norm(mixed_list):
        pos, neg = separate_neg_pos(mixed_list)

        arg1 = abs(0.789*sum(pos) + 0.211*sum(neg))
        arg2 = abs(0.211*sum(pos) + 0.789*sum(neg))

        return max(arg1, arg2)

    def calc_s_norm(W, var_on_zero_weights, variable_weights=False):

        if variable_weights:
            if var_on_zero_weights:
                _W = np.ones((W.shape[0], W.shape[1]))
                row_s_norms = [row_s_norm(r) for r in _W]
                result = max(row_s_norms)
            else:
                _W = np.where(W !=0, 1, W)
                row_s_norms = [row_s_norm(r) for r in _W]
                result = max(row_s_norms)
        else:
            row_s_norms = [row_s_norm(r) for r in W]
            result = max(row_s_norms)

        return result

    w_s_norm = calc_s_norm(W_star, var_on_zero_weights, variable_weights)

    # Calculate all lambdas
    l_s_prime = 4/(w_norm_2)
    l_h_prime = 1/(w_norm_2)

    l_s_star = 1.317/(w_s_norm)
    l_h_star = 1.14/(w_norm_inf)

    return l_s_prime, l_h_prime, l_s_star, l_h_star

def select_lambda(
    w_matrix,
    nodes_order,
    input_nodes,
    activation_function,
    lamda_value,
    lamda_autoslect,
    var_on_zero_weights=False,
    variable_weights=False,
    ):
    """This function returns the FCM lambda paramater based on the
    methodology of [1].

    Parameters
    ----------
    w_matrix : numpy array
        The weight matrix of the FCM.
    nodes_order : list of str
        The order of the nodes. Specifies to which node
        each row and col of the w_matrix refers to.
    input_nodes : list
        A list of input nodes. If there are not any input
        nodes then input_nodes=[]
    activation_function : str
        Accepter values: 'sigmoid' or 'hyperbolic'
    lamda_value : float
        The specified lambda parameter of the transfer function.
    lamda_autoslect : bool
        If True, the function estimates the lambda parameter
        based on [1]. If False, the function return the lamda_value.
    var_on_zero_weights : bool
        True if the zero weights are randon variables with distribution
        of zero mean. False if the zero value indicates no connection
        between source and target node. The default value is False.
    variable_weights : bool
        True if the weights are considered random variables. This
        functionality is necessary when the FCM are combined with
        the Monte Carlo Simulation approach. The default value is False.

    Returns
    -------
    lamda : float
        the selected lambda parameters:
    lamda_autoslect : bool
        equal to the corresponding parameter

    * [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
    of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

    """

    if lamda_autoslect:

        ### the W matrix without the rows of input nodes
        # a. get the index of non-steady nodes from 'nodes_order' list
        if input_nodes:
            no_input_indexes = [i for i,v in enumerate(nodes_order) if v not in input_nodes]
        else:
            no_input_indexes = [i for i,v in enumerate(nodes_order)]
        # b. initialize the W_star matrix
        W_star = np.zeros([len(no_input_indexes), len(nodes_order)])
        # c. fill in the W_star matrix
        for i, v in enumerate(no_input_indexes):
            W_star[i,:] = w_matrix[v,:]

        ### get all lambdas
        (l_s_prime, l_h_prime, l_s_star, l_h_star) = \
            _calc_lambdas(W_star, variable_weights, var_on_zero_weights)

        ### select lambda
        if activation_function=='sigmoid':
            lamda = float(min(l_s_prime, l_s_star))
        elif activation_function=='hyperbolic':
            lamda = float(min(l_h_prime, l_h_star))
        lamda = round(lamda, 3)

    else:
        lamda = lamda_value

    return lamda, lamda_autoslect
