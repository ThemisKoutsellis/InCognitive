# fcm_object.py

"""This module defines the FCM class and initialize its state vector.

The FCMap class provides attributes which contain all the necessary
information of a FCM layout. The derived FCMap object feeds the
'fcm_simulatior' module which then executes the FCM simulation [1]
to derive the final values of the FCM.

* [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

"""

import networkx as nx

from backendcode.activation_function import act_functions
from backendcode.fcm_layout_parameters import (
    get_nx_graph, get_w_matrix, get_lag_matrix)

__all__ = ('FCMap')

#######################################################################
class FCMap(object):
    """A class representing the Fazzy Cognitive Map.

    Attributes
    ----------
    ITERATIONS : int
        the number of max iteration after which the FCM simulation
        stops regardeless of the the outcome
    MIN_NUM_OF_ITERATIONS : int
        it is the minumum number of itertion of the FCM
        simulation [1] after which all values of the input
        nodes propagate all the way to the output nodes.
        Equavalently, it is the min path of the FCM network layout.
        Also ITERATIONS >= MIN_NUM_OF_ITERATIONS
    fcm_layout_dict : dict
        the core dictionary which contains all the necessary info
        for the FCM.
    dont_plot_graphs : bool
        if True the app plot the results of the FCM simulation.
    nodes_order : list ofstr
        The order of nodes as they appear in weigth matrix
    nodes_discription : list of str
        a  brief discription of nodes functionality
    input_nodes : list of str
        the input nodes. if no input nodes then input_nodes = []
    output_nodes : list of str
        the output nodes. if no output nodes then output_nodes = []
    intermediate_nodes : list of str
        the intermediate nodes. if no output nodes then
        intermediate_nodes = [].
    source_nodes : list of str
        the starting points ofthe edges.
    target_nodes : list of str
        the ending points of hte edges
    weights : list of float in [-1, 1]
        the degree of correlation among nodes
    lags : list of int
        the time interval between the intercorrelated node values
    auto_weights : list of floats in [-1, 1]
        the degree of correlation between the current node's value
        and its past values
    auto_lags : list of int
        the interval - as a number of time instances - between the
        current node value and its past node value to which is
        correlated to.
    activation_function_name : str
        the accepted values are 'sigmoid' and 'hyperbolic'
    activation_function_ref :
        the memory reference of the activation function
    nx_graph : DiGraph object
        the networkX representation of the FCM.
    w_matrix : numpy array
        the weightmatrix of the FCM.
    lag_matrix :
        the lag matrix of the FCM.
    normalization: bool
        default value = True. if True, apply normalization
        (see [1] & [2]) to the results of the intermediate
        and output nodes. Otherwise, False.

    * [1] 'Parameter analysis for sigmoid and
    hyperbolic transfer functions of fuzzy cognitive maps',
    https://doi.org/10.1007/s12351-022-00717-x

    * [2] 'Normalising the Output of Fuzzy Cognitive Maps'
    IISA-2022 Confernece.

    """

    # public constants
    _MAX_FCMs = 1
    """The max number of allowed FCMap object per app.
    """
    #initialize class
    _number_of_maps = 0
    """The number of created FCMaps instances.
    """

    def __new__(cls,*args,**kwargs):
        '''Creates a new FCMap object and checks if there is already
        one in memory. If so, it raises an Exception.'''

        cls._number_of_maps += 1
        if cls._number_of_maps > cls._MAX_FCMs:
            # TODO: EXCEPTIONS
            print("There is already an FCM obj! I can't create another one!")
            cls._number_of_maps -= 1
            return
        else:
            return super(FCMap, cls).__new__(cls)

    # Initialize new object
    def __init__(
        self,
        fcm_layout_dict,
        activation_function,
        normalization = True,
    ):
        """Intialize the FCMap object.

        Parameters
        ----------
        fcm_layout_dict : dict
            the core dict which contains all necessary info of the FCM.
        activation_function : str
            Accepted values are 'sigmoid' and 'hyperbolic'.
        dont_plot_graphs : bool
            The default value is False.

        """

        self.ITERATIONS = 300
        self.fcm_layout_dict = fcm_layout_dict
        self.nodes_order = fcm_layout_dict['nodes_order']
        self.nodes_discription = fcm_layout_dict['nodes_discription']
        self.input_nodes = fcm_layout_dict['input_nodes']
        self.output_nodes = fcm_layout_dict['output_nodes']
        self.intermediate_nodes = [
            x for x in self.nodes_order \
            if (x not in self.input_nodes) and (x not in self.output_nodes)
        ]
        self.source_nodes = fcm_layout_dict['source_nodes']
        self.target_nodes = fcm_layout_dict['target_nodes']
        self.weights = fcm_layout_dict['weights']
        self.lags = fcm_layout_dict['lags']
        self.auto_weights = fcm_layout_dict['auto_weights']
        self.auto_lags = fcm_layout_dict['auto_lags']
        self.activation_function_name = activation_function
        self.activation_function_ref = act_functions[
            self.activation_function_name
        ]
        self.nx_graph = get_nx_graph(
            self.source_nodes,
            self.source_nodes,
            self.target_nodes,
            self.weights
        )
        self.w_matrix = get_w_matrix(
            self.nx_graph,
            self.nodes_order,
            self.input_nodes,
            self.auto_weights
        )
        self.lag_matrix = get_lag_matrix(
            self.source_nodes,
            self.target_nodes,
            self.lags,
            self.nodes_order,
            self.auto_lags,
        )
        self.normalization = normalization

        # find the minimum number of required iterations
        if self.input_nodes and self.output_nodes:
            paths = []
            for in_node in self.input_nodes:
                for out_node in self.output_nodes:
                    if not nx.has_path(
                        self.nx_graph, source=in_node, target=out_node):
                        warning_str = (
                            '* [WARNING]: No path between [input node ({})]'
                            '--> [output node ({})] .*'
                            .format(in_node, out_node)
                        )
                        #print(warning_str)
                    else:
                        len_of_min_path = len(
                            nx.shortest_path(
                                self.nx_graph,
                                source=in_node,
                                target=out_node
                            )
                        )
                        paths.append(len_of_min_path)
            self.MIN_NUM_OF_ITERATIONS = max(paths)
        else:
            self.MIN_NUM_OF_ITERATIONS = len(self.nodes_order)


    def __del__(self):
        '''Deleting the FCMap object'''

        FCMap._number_of_maps -= 1
        class_name = self.__class__.__name__
        #print(class_name, "destroyed")

    def set_initial_values(self, fcm_layout_dict):
        """This method initializes the node's state vector.

        Parameters
        ----------
        fcm_layout_dict : dict
            The core dictionary of the InCognitive app.

        Returns
        -------
        dict
            the keys are the names of nodes and the corresponding value
            the initial values of Eq. (2) in [1].

        * [1] 'Parameter analysis for sigmoid and hyperbolic
        transfer functions of fuzzy cognitive maps',
        https://doi.org/10.1007/s12351-022-00717-x
        """
        self.Ao_dict = {}
        self.Ao_dict = {str(v):[fcm_layout_dict["initial_values"][i]] \
            for i,v in enumerate(self.nodes_order)}

        return self.Ao_dict
