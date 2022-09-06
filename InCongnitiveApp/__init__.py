"""The InCognitive App provides a GUI and the necessary backend code
for Fuzzy Cognitive Map (FCM) applications. It provides the functionality
of: a) stand alone FCM simulation and b) FCM combined with Monte
Carlo Simulation (MCS), as well.

The stand alone FCM simulation provides the final state vector of the
nodes, given certain input values.

The FCM-MCS provides the distribution of the final node values,
given that some or all of the input nodes or FCM parameters (e.g. weights)
are random variables.

The 'fcm_layout_dict' dictionary is the core InCognitive dictionary which
contians all necessary info of the FCM layout. It allows all modules to
communicate one another and it propagates through all functions/class methods
from GUI all the way down to the backend code.

"""