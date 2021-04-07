import sys
import quantum_fitter as qf

# Create a instance of Qfit
a = qf.CreateFit()

# upper bound, lower bound, ....., relation
params_dict = {"a": 5,
               "b": 5}

def linear_model(x, a, b):
    y = a*x + b
    return y


a.set_model(linear_model)  # Probably a better way, either send in a function or string like "Exponential decay"

# a.set_params(params_dict)  # Dictionary is harder to remember. Set the individual variables might be easier.
a.set_params('a', val=0, upbound=1, lobound=-1)  # May add more values

a.do_fit()  # Or alternatively a.expdecay()?

a.q_plot()
