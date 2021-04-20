"""
To use the easy Qfit package, you need to first initialize an instance by
ins = quantum_fitter.Qfit(x, y, model, params_init); where params_init is a list for initial value
Then, use ins.do_fit() to fit through lmfit.
If you want to change any parameters' properties, use ins.set_params() to alter.

One example:
==========================================
# define a function method
def gaussian(x, amp, cen, wid):
    1-d gaussian: gaussian(x, amp, cen, wid)
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

# set the data value
x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500)

# create a list for initial value (in sequence of parameters in function method)
params_ini = [5, 5, 1]

# create an instance for Qfit, pass the data and initial value into it (params_ini can be empty)
a = qf.QFit(x, y, gaussian, params_ini)

# Set the property of amp. Mind here the value have to redefine
a.set_params(name='amp', value='5', vary, minimum, maximum, expression, brute_step)

a.do_fit()
a.pretty_print()
plt.show()

"""
from ._fit import QFit
try:
    from lmfit import Model
except ImportError:
    print('Why do you use this package with !!! lmfit !!! not installed')

