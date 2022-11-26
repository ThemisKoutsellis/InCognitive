# fcmmc_object.py

"""This module provides the class, 'MCarloFcm',
whose objects utilised to combine the FCM simulation [1]
with the Monte Carlo Simulation (MCS) aproach.
By doing so, the analyst is capable to explore the
uncertainty propagation through the FCM layout.

The sources of uncertainty are:
i. the weights values of node interconnections and
ii. the values of the input nodes.

When the weights and input values are considered random
variables (i.e. MCS execution), we assume Beta distributions
for all these random variables to quarantee that the support
of their distribution lies in a closed interval.

The 'fcm_mc_execute' class method can handle the following four
(4) cases:
- Case 1 : No variation for both weights and input node values.
- Case 2 : Variation on input node values only. Weights are constant.
- Case 3 : Variation on weights values only. The input node are
           constant.
- Case 4 : Both weights and input node values are random variables.

* [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x
"""

import numpy as np

from backendcode.fcm_simulator import exec_fcm_simulation
from backendcode.fcm_layout_parameters import select_lambda

__all__ = ('MCarloFcm', )

#######################################################################
class MCarloFcm(object):
    """This class is used to combine the FCM simulation [1] and the
    Monte Carlo Simulation (MCS) procedure.

    Attributes
    ----------
    INPUTS_SD : float
        the standard deviation (sd) of input node values.
        zero if the input node values are constant.
    WEIGHTS_SD : float
        the standard deviation (sd) of weights values.
        zero if the weights are constant.
    INPUTS_ITERATIONS : int
        the number of Monte Carlo iterations considering the input
        nodes ss random variables. Conversely, If INPUTS_ITERATIONS=0
        or INPUTS_ITERATIONS=1, the values of input nodes are constants
        during all MCS iterations.
    WEIGHTS_ITERATIONS :
        the number of Monte Carlo iterations considering
        the weigths values as random variables. Conversly,
        if WEIGHTS_ITERATIONS=0 or WEIGHTS_ITERATIONS=1, the
        weights are constants during all MCS iterations.
    VARIANCE_ON_ZERO_WEIGHTS : bool
        default valued = True. if 'True', the zero valued weights are
        random variables with mean=0. Otherwies, if 'False', the zero
        valued weights are constant and their value indicate absence
        of correlation between the nodes of the corresponding edge.
    fcm_obj: FCMap object
        see module 'fcm_onject.py'
    activation_function_name : str
        valid values 'sigmoid' and 'hyperbolic', only.
    output_nodes_values : dict of lists
        the ensamble of the final values of the output nodes, per MCS
        iteration.
    intermediate_nodes_values :
        the ensamble of the final values of the intermediate nodes, per
        MCS iteration.
    input_nodes_values :
        the ensamble of the final values  of the input nodes, per MCS
        iteration.
    lamda : float
        the lambda parameter value to be used if lambda_autoselect=False.
    lambda_autoselect : bool
        True if the user wants the app to auto-select the
        paremeter lambda of the FCM tranfer function based on [1].
    baseline_input_node_values : dict
        a dictionary which contains the normilised [2] final values of
        the input nodes when all FCM parameters (values of input node
        and weights) are constant.
    baseline_output_nodes_values : dict
        a dictionary which contains the normilised [2] final values
        of the output nodes when all FCM parameters (values of input
        node and weights) are constant.
    baseline_intermediate_node_values : dict
        a dictionary which contains the normilised [2] final values
        of the intermediate nodes when all FCM parameters (values
        of input node and weights) are constant.

    * [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
    of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

    * [2] 'Normalising the Output of Fuzzy Cognitive Maps' IISA-2022
    Confernece.

    """

    # public constants
    _MAX_FCM_MC = 1
    """The max number of allowed MCarloFcm object per app.
    """

    #initialize class
    _number_of_fcm_mc = 0
    """The number of created MCarloFcm instances.
    """

    ###################################################################
    def __new__(cls,*args,**kwargs):
        '''Creates a new FCMap object and checks if there is already
        one in memory. If so, it raises an Exception.'''

        cls._number_of_fcm_mc += 1
        if cls._number_of_fcm_mc > cls._MAX_FCM_MC:
            # TODO: EXCEPTIONS
            print("There is already an Fcm Monte Carlo Object.\
                  I can't create another one. Sorry!")
            cls._number_of_fcm_mc -= 1
            return
        else:
            return super(MCarloFcm, cls).__new__(cls)

    ###################################################################
    # Initialize new object
    def __init__(
        self,
        N_INPUTS,
        N_WEIGHTS,
        SD_INPUTS,
        SD_WEIGTHS,
        fcm_obj,
        lamda,
        lamda_autoslect,
        VARIANCE_ON_ZERO_WEIGHTS=True,
    ):
        """Intialize the FCMap object.

        Parameters
        ----------
        N_INPUTS : int
            number of Monte Carlo iterations considering that
            the values of the input nodes are random variables.
            Instead, if N_INPUTS=0 or N_INPUTS=1 the values of input
            nodes are constants.
        N_WEIGHTS: int
            the number of Monte Carlo iterations considering
            that the weigth values are random variables. Instead,
            if N_WEIGHTS=0 or N_WEIGHTS=1 the values of weights
            are constants.
        SD_INPUTS : float
            the standard deviation (sd) of the input nodes values.
            zero if the values of input nodes are considered constant.
        SD_WEIGTHS : float
            the standard deviation (sd) of the weights values.
            zero if the weights are considered constant.
        fcm_obj: FCMap object
            see module fcm_onject.py
        lamda : float
            the lambda parameter value to be used if
            lambda_autoselect=False.
        lamda_autoslect : bool
            True of the user wants the app to choose the lambda
            paremeter of the FCM tranfer function based on [1].
        VARIANCE_ON_ZERO_WEIGHTS : bool
            default valued = True. if True the zero weights are considered
            random distriution with mean value zero. otherwies, if False,
            the weights are constant and the zero value indicate non
            correlation between the nodes of the edge.

        * [1] 'Parameter analysis for sigmoid and
        hyperbolic transfer functions of fuzzy cognitive maps',
        https://doi.org/10.1007/s12351-022-00717-x

        """

        self.INPUTS_SD = SD_INPUTS
        self.WEIGHTS_SD = SD_WEIGTHS

        self.INPUTS_ITERATIONS = N_INPUTS
        self.WEIGHTS_ITERATIONS = N_WEIGHTS

        self.VARIANCE_ON_ZERO_WEIGHTS = VARIANCE_ON_ZERO_WEIGHTS

        self.fcm_obj = fcm_obj

        self.activation_function_name = fcm_obj.activation_function_name
        self.output_nodes_values = {k:[] for k in fcm_obj.output_nodes}
        self.intermediate_nodes_values = {
            k:[] for k in fcm_obj.intermediate_nodes}
        self.input_nodes_values = {k:[] for k in fcm_obj.input_nodes}

        if N_WEIGHTS>1:
            self.lamda, self.lambda_autoselect  = select_lambda(
                self.fcm_obj.w_matrix,
                self.fcm_obj.nodes_order,
                self.fcm_obj.input_nodes,
                self.activation_function_name,
                lamda,
                lamda_autoslect,
                self.VARIANCE_ON_ZERO_WEIGHTS,
                True,
            )
        else:
            self.lamda, self.lambda_autoselect  = select_lambda(
                self.fcm_obj.w_matrix,
                self.fcm_obj.nodes_order,
                self.fcm_obj.input_nodes,
                self.activation_function_name,
                lamda,
                lamda_autoslect,
            )

    ###################################################################
    def __del__(self):
        '''Deleting the MCarloFcm Object'''
        MCarloFcm._number_of_fcm_mc -= 1
        class_name = self.__class__.__name__
        print(class_name, "destroyed")

    ###################################################################
    def _find_a(self, m, sd):
        _a = ((m*(1-m))/((sd)**(2))) -m

        return _a

    ###################################################################
    def _find_b(self, m, a):
        _b = ((a/m) - a)

        return _b

    ###################################################################
    # Monte Carlo generator when inputs are variables
    def _var_input_mc_gen(self, fcm_object):
        """Case 2 deployment.
        """

        for i in range(self.INPUTS_ITERATIONS):

            # Initialise Ao
            Ao = fcm_object.set_initial_values(fcm_object.fcm_layout_dict)
            Arguments = {str(i):[0]  for i in self.fcm_obj.nodes_order}

            # create a sample of input values
            samples_dict = {}
            if fcm_object.activation_function_name == 'hyperbolic':
                for k, v in Ao.items():
                    if v[0]==float(-1):
                        samples_dict[k] = [-1]
                    elif v[0]==float(1):
                        samples_dict[k] = [1]
                    else:
                        # we need to tranfer the sample
                        # from the [-1,1] domain to [0,1].
                        # This is why the pseudo_sd &
                        # pseudo_mean are needed
                        pseudo_sd = self.INPUTS_SD/2
                        pseudo_mean = (v[0]+1)/2

                        _a = self._find_a(pseudo_mean, pseudo_sd)
                        _b = self._find_b(pseudo_mean, _a)
                        samples_dict[k] = [(np.random.beta(_a, _b,))*2-1]
            else:
                for k, v in Ao.items():
                    if v[0]==float(0):
                        samples_dict[k] = [0]
                    elif v[0]==float(1):
                        samples_dict[k] = [1]
                    else:
                        _a = self._find_a(v[0], self.INPUTS_SD)
                        _b = self._find_b(v[0], _a)
                        samples_dict[k] = [np.random.beta(_a, _b,)]

            fcm_object.Ao_dict = samples_dict
            Ao = samples_dict

            # store the input values of each iteration
            for k in fcm_object.input_nodes:
                self.input_nodes_values[k].append(samples_dict[k][-1])

            # fcm simulation execution:
            (
                normilised_output_final_values,
                normilised_intermediate_final_values,
                normilised_intermediate_df,
                normilised_output_df,
            ) = exec_fcm_simulation(
                Ao,
                Arguments,
                fcm_object.input_nodes,
                fcm_object.intermediate_nodes,
                fcm_object.output_nodes,
                fcm_object.MIN_NUM_OF_ITERATIONS,
                fcm_object.activation_function_ref,
                fcm_object.activation_function_name,
                self.lambda_autoselect,
                self.lamda,
                fcm_object.w_matrix,
                fcm_object.lag_matrix,
                fcm_object.nodes_order,
                fcm_object.ITERATIONS,
                fcm_object.normalization,
            )

            # store the outcome of each iteration:
            for k  in fcm_object.intermediate_nodes:
                self.intermediate_nodes_values[k].append(
                    normilised_intermediate_final_values[k])
            for k  in fcm_object.output_nodes:
                self.output_nodes_values[k].append(
                    normilised_output_final_values[k])

            yield  (
                self.input_nodes_values,
                self.intermediate_nodes_values,
                self.output_nodes_values
            )

    ###################################################################
    def _sample_w_matrix(self, w_matrix, VARIANCE_ON_ZERO_WEIGHTS):

        w_array = np.array(w_matrix)

        for i in range(len(w_array)):
            for j in range(len(w_array)):

                if w_array[i,j] == float(1):
                    pass
                elif w_array[i,j] == float(-1):
                    pass
                elif w_array[i,j] == float(0):
                    if VARIANCE_ON_ZERO_WEIGHTS:
                        # we need to tranfer the sample
                        # from the [-1,1] domain to [0,1].
                        # This is why the pseudo_sd &
                        # pseudo_mean are needed

                        pseudo_sd = self.WEIGHTS_SD/2
                        pseudo_mean = (w_array[i,j]+1)/2

                        _a = self._find_a(pseudo_mean, pseudo_sd)
                        _b = self._find_b(pseudo_mean, _a)
                        w_array[i,j] = (np.random.beta(_a, _b,))*2-1
                    else:
                        pass
                else:
                    # we need to tranfer the sample
                    # from the [-1,1] domain to [0,1].
                    # This is why the pseudo_sd &
                    # pseudo_mean are needed

                    pseudo_sd = self.WEIGHTS_SD/2
                    pseudo_mean = (w_array[i,j]+1)/2

                    _a = self._find_a(pseudo_mean, pseudo_sd)
                    _b = self._find_b(pseudo_mean, _a)
                    w_array[i,j] = (np.random.beta(_a, _b,))*2-1

        return np.matrix(w_array)

    ###################################################################
    # Monte Carlo generator when weights are variant
    def _var_weights_mc_gen(self, fcm_object):
        """Case 3 deployment
        """

        _w_matrix = fcm_object.w_matrix

        for i in range(self.WEIGHTS_ITERATIONS):

            # Initialise Ao

            Ao = fcm_object.set_initial_values(
                fcm_object.fcm_layout_dict)
            Arguments = {
                str(i):[0]  for i in self.fcm_obj.nodes_order}

            fcm_object.Ao_dict = Ao

            fcm_object.w_matrix = self._sample_w_matrix(
                _w_matrix,
                self.VARIANCE_ON_ZERO_WEIGHTS
            )

            # fcm simulation execution:
            (
                normilised_output_final_values,
                normilised_intermediate_final_values,
                normilised_intermediate_df,
                normilised_output_df,
            ) = exec_fcm_simulation(
                Ao,
                Arguments,
                fcm_object.input_nodes,
                fcm_object.intermediate_nodes,
                fcm_object.output_nodes,
                fcm_object.MIN_NUM_OF_ITERATIONS,
                fcm_object.activation_function_ref,
                fcm_object.activation_function_name,
                self.lambda_autoselect,
                self.lamda,
                fcm_object.w_matrix,
                fcm_object.lag_matrix,
                fcm_object.nodes_order,
                fcm_object.ITERATIONS,
                fcm_object.normalization,
            )

            # store the outcome of each iteration:
            for k  in fcm_object.intermediate_nodes:
                self.intermediate_nodes_values[k].append(
                    normilised_intermediate_final_values[k])

            for k  in fcm_object.output_nodes:
                self.output_nodes_values[k].append(
                    normilised_output_final_values[k])

            if fcm_object.input_nodes:
                _input_values = fcm_object.set_initial_values(
                    fcm_object.fcm_layout_dict)
                for k in self.input_nodes_values.keys():
                    self.input_nodes_values[k] = _input_values[k][0]

            yield  (
                self.intermediate_nodes_values,
                self.output_nodes_values
            )


    ###################################################################
    def _var_inputs_n_weights_mc_gen(self, fcm_object):
        """Case 4 deployment
        """

        _w_matrix = fcm_object.w_matrix

        for i in range(self.WEIGHTS_ITERATIONS):

            # Initialise Ao
            Ao = fcm_object.set_initial_values(
                fcm_object.fcm_layout_dict)
            Arguments = {
                str(i):[0]  for i in self.fcm_obj.nodes_order}


            # create a sample of input values
            samples_dict = {}
            if fcm_object.activation_function_name == 'hyperbolic':
                for k, v in Ao.items():
                    if v[0]==float(-1):
                        samples_dict[k] = [-1]
                    elif v[0]==float(1):
                        samples_dict[k] = [1]
                    else:
                        # we need to tranfer the sample
                        # from the [-1,1] domain to [0,1].
                        # This is why the pseudo_sd &
                        # pseudo_mean are needed
                        pseudo_sd = self.INPUTS_SD/2
                        pseudo_mean = (v[0]+1)/2

                        _a = self._find_a(pseudo_mean, pseudo_sd)
                        _b = self._find_b(pseudo_mean, _a)
                        samples_dict[k] = [(np.random.beta(_a, _b,))*2-1]
            else:
                for k, v in Ao.items():
                    if v[0]==float(0):
                        samples_dict[k] = [0]
                    elif v[0]==float(1):
                        samples_dict[k] = [1]
                    else:
                        _a = self._find_a(v[0], self.INPUTS_SD)
                        _b = self._find_b(v[0], _a)
                        samples_dict[k] = [np.random.beta(_a, _b,)]

            fcm_object.Ao_dict = samples_dict
            Ao = samples_dict

            # store the input values of each iteration
            for k in fcm_object.input_nodes:
                self.input_nodes_values[k].append(samples_dict[k][-1])

            fcm_object.w_matrix = self._sample_w_matrix(
                _w_matrix,
                self.VARIANCE_ON_ZERO_WEIGHTS
            )

            # fcm simulation execution:
            (
                normilised_output_final_values,
                normilised_intermediate_final_values,
                normilised_intermediate_df,
                normilised_output_df,
            ) = exec_fcm_simulation(
                Ao,
                Arguments,
                fcm_object.input_nodes,
                fcm_object.intermediate_nodes,
                fcm_object.output_nodes,
                fcm_object.MIN_NUM_OF_ITERATIONS,
                fcm_object.activation_function_ref,
                fcm_object.activation_function_name,
                self.lambda_autoselect,
                self.lamda,
                fcm_object.w_matrix,
                fcm_object.lag_matrix,
                fcm_object.nodes_order,
                fcm_object.ITERATIONS,
                fcm_object.normalization,
            )

            # store the outcome of each iteration:
            for k  in fcm_object.intermediate_nodes:
                self.intermediate_nodes_values[k].append(
                    normilised_intermediate_final_values[k])
            for k  in fcm_object.output_nodes:
                self.output_nodes_values[k].append(
                    normilised_output_final_values[k])

            yield  (
                self.input_nodes_values,
                self.intermediate_nodes_values,
                self.output_nodes_values
            )

    ###################################################################
    def fcm_mc_execute(self, fcm_layout_dict):
        """This class method deploys the combination of
        FCM simulation and the Monte Carlo Simulation.

        The following cases are deployed:
            - Case 1 : No variation for both weights and input node values.
            - Case 2 : Variation on input node values only. Weights are constant.
            - Case 3 : Variation on weights values only. The input node are constant.
            - Case 4 : Both weights and input node values are random variables.

        Parameters
        ----------
        fcm_layout_dict
            the core dictionary of the InCofnitive app. It contains
            most of the necessary info of the FCM layout.

        Returns
        -------
        None
            The results are stored to the atribbutes of the FCM object

        """

        fcm_object = self.fcm_obj

        # CASE 1 : Constant weights-Constant inputs
        # -----------------------------------------
        if (self.WEIGHTS_ITERATIONS<2) and (self.INPUTS_ITERATIONS<2):

            Ao = fcm_object.set_initial_values(fcm_layout_dict)
            Arguments = {str(i):[0]  for i in fcm_object.nodes_order}
            (
                normilised_output_final_values,
                normilised_intermediate_final_values,
                normilised_intermediate_df,
                normilised_output_df,
            ) = exec_fcm_simulation(
                Ao,
                Arguments,
                fcm_object.input_nodes,
                fcm_object.intermediate_nodes,
                fcm_object.output_nodes,
                fcm_object.MIN_NUM_OF_ITERATIONS,
                fcm_object.activation_function_ref,
                fcm_object.activation_function_name,
                self.lambda_autoselect,
                self.lamda,
                fcm_object.w_matrix,
                fcm_object.lag_matrix,
                fcm_object.nodes_order,
                fcm_object.ITERATIONS,
                fcm_object.normalization,
            )

            if not normilised_intermediate_df.empty:
                for k in self.intermediate_nodes_values.keys():
                    self.intermediate_nodes_values[k] = \
                        normilised_intermediate_df[k].tail(1).values[0]
            if not normilised_output_df.empty:
                for k in self.output_nodes_values.keys():
                    self.output_nodes_values[k] = \
                        normilised_output_df[k].tail(1).values[0]
            if fcm_object.input_nodes:
                input_values = fcm_object.set_initial_values(fcm_layout_dict)
                for k in self.input_nodes_values.keys():
                    self.input_nodes_values[k] = input_values[k][0]
            self.baseline_output_node_values = {}
            self.baseline_intermediate_node_values = {}
            self.baseline_input_node_values = {}

        # CASE 2 : Constant weights - Variable inputs
        # -------------------------------------------
        if (self.WEIGHTS_ITERATIONS<2) and (self.INPUTS_ITERATIONS>1):

            # The MC simulation
            for g in self._var_input_mc_gen(self.fcm_obj):
                pass

            # Run the baseline scenario
            Ao = fcm_object.set_initial_values(fcm_layout_dict)
            Arguments = {str(i):[0]  for i in fcm_object.nodes_order}
            self.fcm_obj.Ao_dict = Ao

            _original_Ao = fcm_object.set_initial_values(fcm_layout_dict)


            (
                baseline_normilised_output_final_values,
                baseline_normilised_intermediate_final_values,
                baseline_normilised_intermediate_df,
                baseline_normilised_output_df,
            ) = exec_fcm_simulation(
                Ao,
                Arguments,
                fcm_object.input_nodes,
                fcm_object.intermediate_nodes,
                fcm_object.output_nodes,
                fcm_object.MIN_NUM_OF_ITERATIONS,
                fcm_object.activation_function_ref,
                fcm_object.activation_function_name,
                self.lambda_autoselect,
                self.lamda,
                fcm_object.w_matrix,
                fcm_object.lag_matrix,
                fcm_object.nodes_order,
                fcm_object.ITERATIONS,
                fcm_object.normalization,
            )

            self.baseline_output_node_values = \
                baseline_normilised_output_final_values
            self.baseline_intermediate_node_values = \
                baseline_normilised_intermediate_final_values
            self.baseline_input_node_values = _original_Ao

        # CASE 3 : Variable weights - Constant inputs
        # -------------------------------------------
        if (self.WEIGHTS_ITERATIONS>1) and (self.INPUTS_ITERATIONS<2):

            _original_w_matrix = self.fcm_obj.w_matrix

            for g in self._var_weights_mc_gen(self.fcm_obj):
                pass

            # run the baseline scenario
            Ao = fcm_object.set_initial_values(fcm_layout_dict)
            Arguments = {str(i):[0]  for i in fcm_object.nodes_order}
            self.fcm_obj.Ao_dict = Ao
            _original_Ao = fcm_object.set_initial_values(fcm_layout_dict)

            self.fcm_obj.w_matrix = _original_w_matrix

            (
                baseline_normilised_output_final_values,
                baseline_normilised_intermediate_final_values,
                baseline_normilised_intermediate_df,
                baseline_normilised_output_df,
            ) = exec_fcm_simulation(
                Ao,
                Arguments,
                fcm_object.input_nodes,
                fcm_object.intermediate_nodes,
                fcm_object.output_nodes,
                fcm_object.MIN_NUM_OF_ITERATIONS,
                fcm_object.activation_function_ref,
                fcm_object.activation_function_name,
                self.lambda_autoselect,
                self.lamda,
                fcm_object.w_matrix,
                fcm_object.lag_matrix,
                fcm_object.nodes_order,
                fcm_object.ITERATIONS,
                fcm_object.normalization,
            )

            self.baseline_output_node_values = \
                baseline_normilised_output_final_values
            self.baseline_intermediate_node_values = \
                baseline_normilised_intermediate_final_values
            self.baseline_input_node_values = _original_Ao

        # CASE 4 : Variable weights - Variable inputs
        # -------------------------------------------
        if (self.WEIGHTS_ITERATIONS>1) and (self.INPUTS_ITERATIONS>1):

            _original_w_matrix = self.fcm_obj.w_matrix

            for g in self._var_inputs_n_weights_mc_gen(self.fcm_obj):
                pass

            # run the baseline scenario
            Ao = fcm_object.set_initial_values(fcm_layout_dict)
            Arguments = {str(i):[0]  for i in fcm_object.nodes_order}
            self.fcm_obj.Ao_dict = Ao
            _original_Ao = fcm_object.set_initial_values(fcm_layout_dict)

            self.fcm_obj.w_matrix = _original_w_matrix

            (
                baseline_normilised_output_final_values,
                baseline_normilised_intermediate_final_values,
                baseline_normilised_intermediate_df,
                baseline_normilised_output_df,
            ) = exec_fcm_simulation(
                Ao,
                Arguments,
                fcm_object.input_nodes,
                fcm_object.intermediate_nodes,
                fcm_object.output_nodes,
                fcm_object.MIN_NUM_OF_ITERATIONS,
                fcm_object.activation_function_ref,
                fcm_object.activation_function_name,
                self.lambda_autoselect,
                self.lamda,
                fcm_object.w_matrix,
                fcm_object.lag_matrix,
                fcm_object.nodes_order,
                fcm_object.ITERATIONS,
                fcm_object.normalization,
            )

            self.baseline_output_node_values = \
                baseline_normilised_output_final_values
            self.baseline_intermediate_node_values = \
                baseline_normilised_intermediate_final_values
            self.baseline_input_node_values = _original_Ao

        return
