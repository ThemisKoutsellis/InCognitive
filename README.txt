THE IN-COGNITIVE APP:
--------------------
The InCognitive App provides the necessary GUI and backend code for
Fuzzy Cognitive Map (FCM) applications.


THE IN-COGNITIVE FUNCTIONALITIES:
---------------------------------
The InCognitive app provides the following functionalities:

- Deployment and graphical representation of a FCM.

- Selection of the lambda parameter of the FCM tranfer function
  based on the methodology of [1]. The yielded lambda value
  guarantees that the the FCM simulation (see nect functionality)
  do not yield chaotic ot indefinite oucomes.

- FCM simulation based on [1] and [2].
  This simulation provides the final values of each FCM node
  given certain values of the input FCM nodes and FCM parameters
  (e.g. lambda parameter of the transfer function)

- A combination of FCM simulation and Monte Carlo (MC) procedure.
  This combination analyses the uncertainty proagation from input
  nodes all the way through the output nodes in case the weights
  of the FCM edges and the values of the input nodes are random
  variables.


EXECUTION:
----------
To run the application, the user needs only to execute the
main.py module. The GUI provides all necessary interaction
between the user and the backend code.


REFERENCES:
-----------
* [1] 'Parameter analysis for sigmoid and hyperbolic transfer functions
of fuzzy cognitive maps', https://doi.org/10.1007/s12351-022-00717-x

* [2] 'Normalising the Output of Fuzzy Cognitive Maps' IISA-2022
Confernece.