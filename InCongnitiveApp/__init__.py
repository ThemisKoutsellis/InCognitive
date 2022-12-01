#######################################################################
""""The InCognitive App provides a GUI and backend code for Fuzzy
Cognitive Map (FCM) simulations. It's functionalities are:
a) stand alone FCM simulation and
b) combination of FCM and Monte Carlo (MC) simulation.

The 1st functionality provides the final nodes' state vector,
given predifined input-node values.

The 2nd functionality, the FCM-MC simulation, provides the final
node-values' distribution, given that some, or all, of the input
nodes and/or the FCM parameters (e.g. weights) are random variables.

The 'fcm_layout_dict' dictionary is the core InCognitive dictionary
variable, which contains all necessary info of the FCM layout. It
allows all modules to communicate one another: it is exchanged, as
an argument, through all function/class methods, from GUI all the way
down to the backend code.
"""