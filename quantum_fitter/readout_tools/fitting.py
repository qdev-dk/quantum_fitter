import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform
from sklearn.utils.fixes import loguniform
from sklearn import svm

from .loading import *


class Fitting(DataImport):
    """This class contains all fitting functions.
    """
    def __init__(self, filePath=None, fileName=None, channelName=None, entries=None, state_entries=None, labels=None, size=None, 
                 scalar=True, pca=True, cv_params=None, verbose=1, kfolds=10, data=None):
        """Creates instance of the class and set the default classifier, pipeline, parameters and cv_search values.

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
        """
        super().__init__(filePath=filePath, fileName=fileName, channelName=channelName, entries=entries, state_entries=state_entries, labels=labels, size=size, kfolds=kfolds, data=data)  
        
        self.verbose = verbose
        
        self.set_classifier()  
        self.set_pipeline(scalar, pca)
        self.set_cv_params(cv_params)
        self.set_cv_search()
      
    def set_pipeline(self, scalar=True, pca=True):
        """Generates a pipeline with a specific classifier, a scaler function and a PCA.

        Args:
            scalar (bool, optional): If True the data is standardized in the pipeline. This ensures the optimal conditions for the machine learning. Defaults to True.
            pca (bool, optional): If True the data is transformed by PCA in the pipeline. This ensures the optimal conditions for the machine learning. Defaults to True.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        if scalar == True:
            scalar = StandardScaler()
        else:
            scalar = None
        
        if pca == True or int:
            if pca == int:
                pca = PCA(n_components=pca)
            else:
                pca = PCA(n_components=2)
        else:
            pca = None
            
        self.pipeline = Pipeline(steps=[
                                  ('transformer', scalar), 
                                  ('PCA', pca), 
                                  ('classifier', self.classifier)
                                  ])
      
    def set_classifier(self, classifier=None):
        """Sets the default classifier to be used.

        Args:
            classifier (object, optional): The classified to be used. This can be a SVM or another type of classifier. If none SVM RBF is used. Defaults to None.
        """
        from sklearn import svm
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    
        
        if classifier:
            self.classifier = classifier
        else:
            self.classifier = svm.SVC(kernel='rbf', probability=True)
    
    def set_cv_params(self, params=None, remove=None): 
        """Sets the parameters for the cross validation. 

        Args:
            params (list, optional): To add or change a parameter the following one used: self._int_params['parameter'] = {'parameter name': paramter values}. Defaults to None.
            remove (list, optional): Removes a unwanted paramete. Defaults to None.
        """
        import copy
        
        if params:
            self._int_params = params
        
        if remove:
            self._int_remove = remove
        
        self.cv_params_all = {
                    # SVM
                    'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'], #'poly', 'sigmoid'
                    'classifier__degree': randint(2,4), 
                    'classifier__gamma': ['scale', 'auto'],   #or use: loguniform(1e-2, 1e2)
                    'classifier__coef0': randint(1,15),
                    'classifier__C': loguniform(1e-3, 1e4),
                    #'classifier__class_weight':['balanced', None],
                    
                    # KNeighbors
                    'classifier__n_neighbors': randint(5, 50),
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__leaf_size':randint(5, 50),
                    'classifier__p':randint(1, 4),
                    
                    # AdaBoost
                    'classifier__learning_rate': uniform(0.05, 1),
                    'classifier__algorithm': ['SAMME', 'SAMME.R'],
                    
                    # DecisionTree
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__splitter': ['best', 'random'],
                    'classifier__max_depth': randint(5, 50),
                    'classifier__min_samples_split': randint(2, 10),
                    
                    # LinearDiscriminant
                    'classifier__solver': ['svd', 'lsqr', 'eigen']
                    }
        
        self.cv_params = copy.deepcopy(self.cv_params_all)
        
        if hasattr(self, '_int_params'):
            for key in self._int_params.keys():
                if key in self.cv_params:
                    self.cv_params[key] = self._int_params[key]
                else:
                    self.cv_params[key] = self._int_params[key]
                    print(key, 'is not in cv_params. Key was made.')
        
        if hasattr(self, '_int_remove') :
            for key in self.__int_remove:
                self.cv_params.pop(key, None)
                print(key, 'is no longer in cv_params')  
    
    def set_cv_search(self, mode='random', aggressive_elimination=False, min_resources='smallest'):
        """Sets the type of cross validation.

        Args:
            mode (str, optional): If 'random' HalvingRandomSearchCV is used. HalvingRandomSearchCV takes parameters like: randint(5, 50). If 'grid' GridSearchCV is used. GridSearchCV takes parameters like: range(5, 50). Defaults to 'random'.
            aggressive_elimination (bool, optional): If True aggressive elimination is use, which can improve wall clock time but can return a less precise classifier. Defaults to False.
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import StratifiedKFold
        from sklearn.experimental import enable_halving_search_cv # noqa
        from sklearn.model_selection import HalvingRandomSearchCV
        
        if hasattr(self, '_int_mode') == False:
            self._int_mode = mode
        if hasattr(self, '_int_aggressive_elimination') == False or aggressive_elimination == True:
            self._int_aggressive_elimination = aggressive_elimination
        
        self._kfold = StratifiedKFold(n_splits=(self.kfolds-1), random_state=0, shuffle=True)
        
        if self._int_mode == 'random':
            self.cv_search = HalvingRandomSearchCV(self.pipeline, param_distributions=self.cv_params,
                        random_state=0, cv=self._kfold, n_jobs=-1, aggressive_elimination=self._int_aggressive_elimination, 
                        error_score=np.NaN, verbose=self.verbose, min_resources=min_resources)
    
        elif self._int_mode == 'grid':
            self.cv_search = GridSearchCV(self.pipeline, self.cv_params, scoring='average_precision', 
                                             cv=self._kfold-1, refit=True, verbose=self.verbose)
            
        else:
            self.cv_search = None
            print("No search type added. You can change the search type be calling: self._serch_type or by setting the mode = random or grid")
    
    def _reduce_params(self, verbose):
        """A function that reduces the number of parameters to only contain the ones needed by the classifier.

        Args:
            verbose (int, optional): If 0 only the result is returned. Defaults to 1.
        """
        def _replace_function(str):
            return list(map(lambda item: (item.replace("classifier__","").replace("estimator__classifier__","")), str))
            
        params_all = _replace_function(self.cv_params_all.keys())
       
        popped = []
        for key in list(self.cv_params):
            if key.replace("classifier__","") not in set(params_all).intersection(self.classifier.get_params().keys()):
                self.cv_params.pop(key, None)
                popped.append(key.replace("classifier__",""))
          
        if verbose > 0:
            print('Popped:', popped)           
                
    def do_fit(self, X=None, y=None, weights=None):
        """To train the classifier on data use the function do_fit. Other functions call this if this is not manually run.

        Args:
            X (list, optional): The X-data to use. If None the Initial states are used: self.X_train. Defaults to None.
            y (list, optional): The y-data to use. If None the Initial states are used: self.y_train. Defaults to None.
            weights (list, optional): The calculated weights. Defaults to None.
        """
        if X == None:
            X, y, weights = self.X_train, self.y_train, self.weights

        self.set_pipeline()
        self.set_cv_params()
        self.set_cv_search()
        self._reduce_params(verbose=self.verbose)
        
        self.cv_search.fit(X, y, classifier__sample_weight=weights)
        
    # oscillation fitting
    def oscillations_guess(self, x, y):
        """Estimates initial guesses for an  oscillation function. (Adapted from QDev wrappers, `qdev_fitter` and modified)

        Args:
            x (list): The x data set
            y (list): The y data set

        Returns:
            list: initial values in the formate: [A, omega, phi, c]
        """
        from scipy import fftpack
        y = np.array(y)
        sorted_index_array = np.argsort(y)
        y_sorted = y[sorted_index_array]
        
        A = (np.mean(y_sorted[5 : 10]) - np.mean(y_sorted[-10 : -5])) / 2
        c = np.mean(y)
        yhat = fftpack.rfft(y-np.mean(y))
        idx = (yhat**2).argmax()
        freqs = fftpack.rfftfreq(len(x), d = (x[1]-x[0])/(2*np.pi))
        omega = freqs[idx]
        factor = np.mean(y)/np.mean(y_sorted[5 : 10]) - np.mean(y[:3])/np.mean(y_sorted[5 : 10])
        phi = factor * 3.14
      
        return [A, omega, phi, c]

    def oscillations(self, x, A, omega, phi, c):
        """Oscillation function used to fit.
        """
        return (A * np.sin(omega * x + phi) + c)
    
    def do_fit_oscillation(self, x, y, label=None, ax=None, color=None):
        """function to be used to sit oscillations. For more information look at example "quick_run.py".

        Args:
            x (list): The x data set
            y (list): The y data set
            label (string, optional): The label assigned to the data. Defaults to None.
            ax (objects, optional): Plot ax. Defaults to None.
            color (string, optional): Color to plant with. Defaults to None.

        Returns:
            _type_: _description_
        """
        import quantum_fitter as qf
        from lmfit import Model
        import lmfit
        
        def str_none_if_none(stderr):
            if stderr is None:
                return 'None'
            else:
                return stderr
        
        if ax is None:
            ax = plt.gca()
        
        A, omega, phi, c = self.oscillations_guess(x, y)
        
        # fitting
        t1 = qf.QFit(np.array(x), np.array(y), model=Model(self.oscillations))
        t1.set_params('A', A)
        t1.set_params('c', c)
        t1.set_params('omega', omega)
        t1.set_params('phi', phi)
        
        t1.do_fit()
        
        
        """def resid(params, x, ydata):
            A = params['A'].value
            c = params['c'].value
            omega = params['omega'].value
            phi = params['phi'].value

            y_model = self.oscillations(x, A, omega, phi, c)
            return y_model - ydata"""
        
        
        x_eval = np.linspace(min(x), max(x), 100)
        y_eval = t1.eval(x=x_eval)
        
        """params_t1 = t1.fit_params()
        
        
        params = lmfit.Parameters()
        for key in params_t1.keys():
            params.add(key, params_t1[key])
      
        print(params)
        
        method = 'L-BFGS-B'
        o1 = lmfit.minimize(resid, params, args=(np.array(x), np.array(y)), method=method,
                    reduce_fcn='neglogcauchy')
        lmfit.report_fit(o1)
        
        ax.plot(np.array(x), np.array(y)+o1.residual, label='resid', c='red')"""
        
        fit_params, error_params = t1.result.best_values, t1._params_stderr()
        
        ax.plot(x_eval, y_eval, label=label, c=color)
        ax.scatter(x, y, c=color)
        
        for key in fit_params.keys():
            ax.plot(x_eval[0], y_eval[0], 'o', markersize=0,
                    label='{}: {:4.4}±{:4.4}'.format(key, fit_params[key], str_none_if_none(error_params[key])))
    
    # other
    def _try_fit(self, classifier=None):
        """Function to try if classifier is fitted. If not it fits. This function is insurance that even without the do_fit called function the result is returne.
        
        Args:
            classifier (object, optional): The classifier to fit. It's None the self.cv_search classifiers used. Defaults to None.
        """
        if classifier is None:
            classifier = self.cv_search
        try:
            classifier.predict([[0,0]])
        except:
            print('Classifier not fitted yet. Fitting classifier now.')
            self.do_fit()
            
    def set_plot_dir(self, param_value, param_name=None, score_name=None):
        """This function can be used to plot the effect of different parameters.

        Args:
            param_value (float): The parameter value to save.
            param_name (string, optional): The parameter name. Defaults to None.
            score_name (string, optional): The score name. Defaults to None.
        """
        if score_name == None:
            score_name = 'mean_test_score'
            score_std = 'std_test_score'
        
        if param_name == None:
            param_name = 'Parameter'
        
        if hasattr(self, '_plot_dir') == False or self._plot_dir['param_name'] != param_name or self._plot_dir['score_name'] != score_name:
            self._plot_dir = {'param_name' : param_name,
                              'score_name' : score_name,
                              'param_value': [],
                              'score_value': [],
                              'score_std'  : []
                              }
        
        self.do_fit()
          
        self._plot_dir['param_value'].append(param_value)
        self._plot_dir['score_value'].append(self.cv_search.cv_results_[score_name][self.cv_search.best_index_])
        
        if score_std:
            self._plot_dir['score_std'].append(self.cv_search.cv_results_[score_std][self.cv_search.best_index_])
        
        