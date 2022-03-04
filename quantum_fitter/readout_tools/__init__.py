"""
## Introduction
The Readout Tool is a extension of the Qunatum Fitter package. It's pourpeses is to give the user a esay to use module to classify quantum states. This is done by using sklearn moduels for machine learning.
The main focus of this package is support vector machines (SVM), but it is equal to handle a lange range of machine buning algorithms:

- Support Vector Machines
- KNeighbors
- AdaBoost
- DecisionTree
- LinearDiscriminantAnalysis
- and more.


## How to use the package
Underneath are some of the main functionality highlighted in symbol examples.

### Example One - Arrays
To begin using the module get your groundstate and excited state QI vectors and Make sure they are in the right formate ([[i,q],[i,q],...]. 
The formating can be done by using the rdt.reformate function.

```python
import quantum_fitter.readout_tools as rdt
import Labber as lab
import os

# Data setup
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')

file = lab.LogFile(file_path)
data = rdt.reformate(file.getData())

state_0, state_1 = data[0], data[40]
states = [state_0,state_1] # in [[i,q],[i,q],...] formate.
```

When you got the data in the right formate, your able to run the package like so:

```python
# Set the instence 
rd = rdt.Readout(data=states, size=1000, verbose=1)

# Plot
rd.plot_classifier() # this plots the classifier with the data used for the fitting.

# Use
rd.predict(data[1]) # this returns a list of states for the single shot in data[1].

```

If a h5data file is available import this directly instead of only the QI vecrtors (see example two).




### Example Two - H5data file
To begin using the module get your h5data file path run it like shown.

```python
import quantum_fitter.readout_tools as rdt
import os

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')

# Set the instence 
rd = rdt.Readout(file_path, size=1000, verbose=1)

# Plot
rd.plot_classifier()
```

This will return a plot of the classifier and training data used for determination.


### Example Three - predicting
To use a classifier to predict the state of a single shot, run the following line.

```python
rd.predict(data)
```

where `data` is the QI-array or single-shot you want the predict.


### Example Four - saveing
After estimating a classifier you can save it as a pickle file, so that you keep the same classifier.

```python
import quantum_fitter.readout_tools as rdt

# Set the instence 
rd = rdt.Readout(file_path, size=1000, verbose=1)

# Fitting the classifier
rd.do_fit()

# Exporting classifier
rd.export_classifier(filePath=somewhere/on/your/computer/)

# Importing classifier 
rdt.import_classifier(filePath=somewhere/on/your/computer/)
```
    
If successful important print state will show as
    `"Got your pickle!"`
    
### Futher examples
There are more examples located in the example folder.



## Information
This submodule has been created and are being maintained by Malthe Asmus Marciniak Nielsen
mail: vpq602@alumni.ku.dk
github: MaltheAN

If there any questions please feel free to contact me.   
"""


from .loading import *
from .fitting import *
from .plotting import *
from .readout import *


__pdoc__ = {'quantum_fitter.readout_tools.examples' : False}


