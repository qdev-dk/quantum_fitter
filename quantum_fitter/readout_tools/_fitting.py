import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform
from sklearn.utils.fixes import loguniform
from sklearn import svm

from ._import import DataImport

class Fitting(DataImport):
    def __init__(self, filePath=None, channelName=None, entries=None, state_entries=None, labels=None, size=None, 
                 scalar=True, pca=True, cv_params=None, verbose=1, kfolds=10):
        super().__init__(filePath=filePath, channelName=channelName, entries=entries, 
                            state_entries=state_entries, labels=labels, size=size, kfolds=kfolds)  
        
        self.verbose = verbose
        
        self.set_classifier()  
        self.set_pipeline(scalar, pca)
        self.set_cv_params(cv_params)
        self.set_cv_search()
      
    def set_pipeline(self, scalar=True, pca=True):
        """Generates a pipeline with a specific class a fire a scaler function and a PCA.

        Args:
            classifier (bool): The classifier to use.
            scalar (bool, optional): If a standard scarler should be enabled. Defaults to True.
            pca (bool, optional): If a PCA should be enabled. Defaults to True.

        Returns:
            pipeline: The pipeline.
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
        if classifier:
            self.classifier = classifier
        else:
            self.classifier = svm.SVC(kernel='rbf', probability=True)
    
    def set_cv_params(self, params=None, remove=None): 
        # Estimate of function
        from sklearn import svm
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    
        import copy
        
        if params:
            self._int_params = params
        
        if remove:
            self._int_remove = remove
        
        self.cv_params_all = {
                    # SVM
                    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
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
    
    def set_cv_search(self, mode='random', aggressive_elimination=False):
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
                        error_score=np.NaN, verbose=self.verbose)
    
        elif self._int_mode == 'grid':
            self.cv_search = GridSearchCV(self.pipeline, self.cv_params, scoring='average_precision', 
                                             cv=self._kfold-1, refit=True, verbose=self.verbose)
            
        else:
            self.cv_search = None
            print("No search type added. You can change the search type be calling: self._serch_type or by setting the mode = random or grid")
    
    def _reduce_params(self, verbose):
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
        if X == None:
            X, y, weights = self.X_train, self.y_train, self.weights

        self.set_pipeline()
        self.set_cv_params()
        self.set_cv_search()
        self._reduce_params(verbose=self.verbose)
        
        self.cv_search.fit(X, y, classifier__sample_weight=weights)
        
    # oscillation fitting
    def oscillations_guess(self, x, y):
        # Adapted from QDev wrappers, `qdev_fitter`
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
        return (A * np.sin(omega * x + phi) + c)
    
    def do_fit_oscillation(self, x=None, y=None, label=None, ax=None, color=None):
        import quantum_fitter as qf
        from lmfit import Model
        
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
        
        x_eval = np.linspace(min(x), max(x), 100)
        y_eval = t1.eval(x=x_eval)
        
        fit_params, error_params = t1.result.best_values, t1._params_stderr()
        
        ax.plot(x_eval, y_eval, label=label, c=color)
        ax.scatter(x, y, c=color)
        
        for key in fit_params.keys():
            ax.plot(x_eval[0], y_eval[0], 'o', markersize=0,
                    label='{}: {:4.4}±{:4.4}'.format(key, fit_params[key], str_none_if_none(error_params[key])))
    
    def _try_fit(self, classifier=None):
        if classifier is None:
            classifier = self.cv_search
        try:
            classifier.predict([[0,0]])
        except:
            print('Classifier not fitted yet. Fitting classifier now.')
            self.do_fit()
            
        

