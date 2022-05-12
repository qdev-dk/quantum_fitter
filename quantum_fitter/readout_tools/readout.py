import numpy as np
from .plotting import *
from qutip import *
from qutip.qip.operations import *
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 800

class Readout(Plotting):
    """This class contains simple but effective functions functions.
    """
    def __init__(self, filePath=None, channelName=None, entries=None, state_entries=None, labels=None, 
                 size=None, scalar=True, pca=True, cv_params=None, verbose=1, kfolds=10, figsize=(10, 6), 
                 alpha=0.7, data=None):
        """Initializes the class and sits figure size an alpha val.

        Args:
            filePath (string, optional): File path for the Labber (h5file) file containing the IQ data. Defaults to None.
            channelName (string, optional): Channel name contained IQ data. If None the first trace in the file is used. Defaults to None.
            state_entries (list, optional): A list containing the wanted entries for classification. If None the two entries with the largest and smallest mean are used. Defaults to None.
            labels (list, optional): A list containing the labels for the states. Labels can be integers or a strings. If None numbers from 0 to len(number of states) is used. Defaults to None.
            size (int, optional): The size of the data set used. Must be integer. Defaults to None.
            scalar (bool, optional): If True the data is standardized in the pipeline. This ensures the optimal conditions for the machine learning. Defaults to True.
            pca (bool, optional): If True the data is transformed by PCA in the pipeline. This ensures the optimal conditions for the machine learning. Defaults to True.
            cv_params (_type_, optional): If None the standard parameters are used. These can be changed afterwards. Defaults to None.
            verbose (int, optional): If 0 only the result is returned. Defaults to 1.
            kfolds (int, optional): Number of splits in the dataset. Used for crossvalidation. Defaults to 10.
            figsize (tuple, optional): The size of the figure. Defaults to (10, 6).
            alpha (float, optional): Transparency value of beta points. Float between [0,1]. Defaults to 0.70.
        """
        super().__init__(filePath, channelName, entries, state_entries, labels, size, scalar, pca, 
                         cv_params, verbose, kfolds, figsize, alpha, data)
        
    def cal_expectation_values(self, X=None, size=None, state=1):
        """A function to calculate the expectation values.

        Args:
            X (_type_, optional): The dateset. If None self.h5data is used. Defaults to None.
            size (int, optional): The size of the data set used. Must be integer. Defaults to None.
            state (int, optional): The state to be calculated. Defaults to 1.
        """
        if X is None:
            X = self.h5data
        
        if size:
            X = self.set_dataset_size(size=size, X=X)
        
        def inv_hadamard(n):
            hadamard = tensor([snot()]*n)*(1/np.sqrt(2))**n
            return hadamard.inv()
           
        def avg_function(self, Xi, state):
            predcition = self.cv_search.predict(Xi)
            n_states =  len(self._states_labels)
            
            prob_list = []
            for i, state_i in enumerate(self._states_labels):
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
          
    def predict(self, data):
        """Prediction function for a dataset or a datapoint. It does the same as calling self.cv_search.predict(data).

        Args:
            data (list): The data point or data set to predict.

        Returns:
            predict (list): The predicted state for each data point in data.
        """
        self._try_fit()
        data = np.array(data)
        
        try:
            try:
                predict = self.cv_search.predict(data)
            except:
                predict = self.predict([data])
        except:
            try:
                predict = self.cv_search.predict(self._reformate(data))
            except:
                print('Something went wrong. Data may be in wrong format. Make sure data is in format: [[I1,Q1],[I2, Q2]]')
        
        return predict
    
    def get_best_params(self):
        """To get the best parameters for the highest scoring classifier. The same information can be fund by calling self.cv_search.cv_results_

        Returns:
            best params (list): Best prarameters
        """
        best_params = self.cv_search.cv_results_['params'][self.cv_search.best_index_]
        
        return best_params
    
    def _get_file_name_from_path(self, path, part='tail'):
        """Small function for getting the hit and sale of path.

        Args:
            path (string): Datafile path
            part (str, optional): If head, the main part is returned. If tail, the filename is retruned. Defaults to 'tail'.

        Returns:
            head (string): main part of path.
            tail (string): filename 
        """
        import os
        try:
            head, tail = os.path.split(path)
            
            if part == 'head':
                return head
            else:
                return tail
        except:
            #print('No filepath fund. Please make a title manually.')
            return ''
    
    
    def _reformate(self, X):
        """A function that reformates data if not in the right format. The wanted format is [[i,q],[i,q], ...]
        """ 
        h5data_reformated = []
        for i in range(len(X)):
            h5data_reformated.append(np.column_stack((X[i].real, X[i].imag)))
        return np.array(h5data_reformated)

    
    def calculate_temp(self, X=None, prob=None, freq=None, size=None):
        def temp(prob_1, freq):
            from scipy.constants import hbar, k, pi
            
            if np.array(prob_1).size > 1:
                prob_0 = prob[0]
                prob_1 = prob[1]
            else:
                prob_0 = 1 - prob_1
                    
            return (hbar*(2*pi*freq)) / (np.log(prob_0/prob_1) * k)
        
        def avg_function(self, Xi, state):
            predcition = self.cv_search.predict(Xi)
            n_states =  len(self._states_labels)
            
            prob_list = []
            for i, state_i in enumerate(self._states_labels):
                if state_i == state:
                    i_state = i
                    
                predcition_i = np.where((predcition == state_i), 1, 0)
                prob_predcition_i = np.average(predcition_i, axis=0)
                prob_list.append(prob_predcition_i)
            
            prob_list_temp = prob_list.copy()
            
            if n_states == 3:
                prob_list_temp[1] /= 2
                prob_list_temp.insert(1, prob_list_temp[1])

            if state == 'all':
                prob_predcition_avg.append(prob_list)
            else:
                prob_predcition_avg.append(prob_list[i_state])
        
        if freq == None:
            try:
                freq = self.h5file.getChannelValue('RS Drive - Frequency')
            except:
                print('There is no frequnecy to be fund in the h5file. Please imput freq.')
                
        if prob == None:
            try:
                prob = []
                for key in self.probability_values.keys():
                    prob.append(self.probability_values[key][0])
            except:
                self._try_fit()
                prob_predcition_avg = []
                
                if X == None:
                    X = self.h5data
                    
                X = self.set_dataset_size(size=size, X=X)
                avg_function(self, X[0], 1)
                
                prob = prob_predcition_avg
            
        if hasattr(self, 'temp_list') ==  False:
            self.temp_list = []
            self.temp_prob_list = []
     
        for prob_i in prob:
            self.temp_prob_list.append(prob_i)
            self.temp_list.append(temp(prob_i, freq))
 
def reformate(X):
    """A function that reformates data if not in the right format. The wanted format is [[i,q],[i,q], ...]
    """ 
    h5data_reformated = []
    for i in range(len(X)):
        h5data_reformated.append(np.column_stack((X[i].real, X[i].imag)))
    return np.array(h5data_reformated)


