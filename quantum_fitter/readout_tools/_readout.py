import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from ._plotting import Plotting
from qutip import *
from qutip.qip.operations import *
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 800

class Readout(Plotting):
    def __init__(self, filePath=None, channelName=None, entries=None, state_entries=None, labels=None, 
                 size=None, scalar=True, pca=True, cv_params=None, verbose=1, kfolds=10, figsize=(10, 6), 
                 alpha=0.7):
        super().__init__(filePath, channelName, entries, state_entries, labels, size, scalar, pca, 
                         cv_params, verbose, kfolds, figsize, alpha)
        
    def cal_expectation_values(self, X=None, size=None, state=1):
        if X is None:
            X = self.h5data
        
        if size:
            X = self.set_dataset_size(size=size, X=X)
        
        def inv_hadamard(n):
            hadamard = tensor([snot()]*n)*(1/np.sqrt(2))**n
            return hadamard.inv()
           
        def avg_function(self, Xi, state):
            predcition = self.cv_search.predict(Xi)
            n_states =  np.unique(predcition).size
            
            prob_list = []
            for i, state_i in enumerate(np.unique(predcition)):
                if state_i == state:
                    i_state = i
                    
                predcition_i = np.where((predcition == state_i), 1, 0)
                prob_predcition_i = np.average(predcition_i, axis=0)
                prob_list.append(prob_predcition_i)
            
            prob_list_temp = prob_list.copy()
            
            if n_states == 3:
                prob_list_temp[1] /= 2
                prob_list_temp.insert(1, prob_list_temp[1])
                
            exp_list = np.dot(prob_list_temp, inv_hadamard(n_states-1))[-1].real
            
            if state == 'all':
                prob_predcition_avg.append(prob_list)
            else:
                prob_predcition_avg.append(prob_list[i_state])
            exp_predcition_avg.append(exp_list)
            
            
        """        
        def avg_function(self, Xi, state):
            predcition = self.cv_search.predict(Xi)
            predcition = np.where((predcition == state), 1, 0)
        
            exp_predcition = np.interp(predcition, (0, 1), (1, -1))
            exp_predcition = np.average(exp_predcition, axis=0)
            prob_predcition = np.average(predcition, axis=0)
            
            exp_predcition_avg.append(exp_predcition)
            prob_predcition_avg.append(prob_predcition)
        """
                
        self._try_fit()
        
        exp_predcition_avg, prob_predcition_avg = [], []
      
        try:
            from tqdm import tqdm
        except ImportError:
            print('tqdm is not installed. No progress bar shown')
            
            for i in range(len(X)):
                if self.verbose > 0:
                    print(f'Calculated {i} out of {len(X)}.')
            
                avg_function(self, X[i], state)
        else:
            for i in tqdm(range(len(X))):
                avg_function(self, X[i], state)
                
        kernel = self.cv_search.best_estimator_[-1].kernel
        
        if hasattr(self, 'expectation_values') == False:
            self.expectation_values =  {kernel: exp_predcition_avg,}
        else:
            self.expectation_values[kernel] = exp_predcition_avg
        
        if hasattr(self, 'probability_values') == False:
            self.probability_values =  {kernel: prob_predcition_avg,}
        else:
            self.probability_values[kernel] = prob_predcition_avg
          
          