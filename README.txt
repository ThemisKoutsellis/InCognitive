THE IN-COGNITIVE APP:
--------------------
The InCognitive App provides a GUI and backend code for Fuzzy Cognitive
Map (FCM) simulations.

THE IN-COGNITIVE APP FUNCTIONALITIES:
-------------------------------------
- Deployment and graphical representation of a FCMs.

- Selection of FCM-transfer-function's parameter lambda; it is based on
  the methodology of [1]. The yielded lambda value guarantees that the
  stand-alone FCM or FCM-MC simulations (see next functionalities) do not
  yield chaotic or indefinite oucomes.

- Stand-alone-FCM simulation, based on [1] and [2]. It provides the final
  nodes' state vector, given predifined input-node values and FCM parameters
  (e.g. FCM-transfer-function's parameter lambda).

- A combination of FCM and Monte Carlo (MC) simulation. This combination,
  FCM-MC, analyses the uncertainty propagation from input nodes all the way
  to the output nodes, in case the "weights" of FCM edges and/or the input
  node values are random variables. It's outcome is the final node value
  distributions, given that some, or all, of the input nodes and/or the FCM
  parameters (e.g. weights) are random variables.


EXECUTION:
----------
To run the application, the user needs only to execute the main.py
module. The GUI provides all necessary interaction between the end-user
and backend-code. 
For further details see: https://github.com/ThemisKoutsellis/InCognitive/wiki/


REFERENCES:
-----------
[1] T. Koutsellis et al., "Parameter analysis for sigmoid and hyperbolic
transfer functions of fuzzy cognitive maps," 2022, Oper Res Int J, 22,
pp. 5733–5763, https://doi.org/10.1007/s12351-022-00717-x

[2] T. Koutsellis et al., "Normalising the Output of Fuzzy Cognitive Maps,"
2022 13th International Conference on Information, Intelligence, Systems
& Applications (IISA), 2022, pp. 1-7, doi: 10.1109/IISA56318.2022.9904369.
