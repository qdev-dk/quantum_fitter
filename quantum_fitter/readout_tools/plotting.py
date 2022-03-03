import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from quantum_fitter.readout_tools import *

class Plotting(Fitting):
    """This class contains all plotting functions.
    """
    def __init__(self, filePath=None, channelName=None, entries=None, state_entries=None, labels=None, 
                 size=None, scalar=True, pca=True, cv_params=None, verbose=1, kfolds=10, figsize=(10, 6),
                 alpha=0.70):
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
        super().__init__(filePath, channelName, entries, state_entries, labels, size, scalar, pca, cv_params, verbose, kfolds)

        self.figsize = figsize
        self.alpha = alpha
    
    def plot_classifier_decision_function(self, resolution=350, ax=None, plot_support=True):
        """Plots the decision function for a 2D decision function

        Args:
            resolution (int, optional): Resolution of the plotted decision function. Defaults to 350.
            ax (string, optional): The plot ax to use, if ax=None then new ax is generated. Defaults to None.
            plot_support (bool, optional): Plots exstra data, just nicense. Defaults to True.
        """
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate classifiermodel
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        Y, X = np.meshgrid(y, x)
        xy = np.c_[X.ravel(), Y.ravel()]    #xy = np.array([X.ravel(), Y.ravel()]).T

        self._try_fit()
        
        if ((hasattr(self.cv_search, 'decision_function')) and (len(self._states_labels) < 3)):
            P = self.cv_search.decision_function(xy)
        else:
            P = self.cv_search.predict(xy)

        self.decision_function = P.reshape(X.shape)

        if ((hasattr(self.cv_search, 'decision_function')) and (len(self._states_labels) < 3)):
            levels = [-1, 0, 1]
            linestyles = ['--', '-', '--']
        else:
            list, j = [], 0
            for i in range(len(self.state_entries)):
                num = i-j
                if (i % 2) == 0:
                    list.append(0.5 + num*0.5)
                else:
                    j += 1
                    list.append(0.5 - num*0.5)
               
            levels = list.sort()
            linestyles = ['-'] * len(self.state_entries)

        ax.contour(X, Y, self.decision_function, colors='k',
                   alpha=0.5, levels=levels,
                   linestyles=linestyles)

        # plot support vectors
        if hasattr(self.cv_search, 'support_vectors_'):
            if plot_support:
                ax.scatter(self.cv_search.support_vectors_[:, 0],
                           self.cv_search.support_vectors_[:, 1],
                           s=300,
                           alpha=0.7, linewidth=1,
                           facecolors='None', edgecolors="k")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    def plot_classifier(self, X=None, y=None, sample_weight=None, class_plot=True, save_fig=False, title=None):
        """Plotting fuction for SVM fits

        Args:
            X (list, optional): The X-data to use. If None the Initial states are used: self.X_train. Defaults to None.
            y (list, optional): The y-data to use. If None the Initial states are used: self.y_train. Defaults to None.
            sample_weight (list, optional): The calculated weights. Defaults to None.
            class_plot (bool, optional): If True the classifiers plotted. Defaults to True.
            save_fig (bool, optional): If True the figure is saved. Defaults to False.
            title (_type_, optional): Title of the figure. If None the data file name is used. Defaults to None.

        Returns:
            ax (object): The figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if X is None:
            X = np.array(self.X_train)
            
        if y is None:
            y = np.array(self.y_train)
        
        if isinstance(sample_weight, int) == False:
            sample_weight = self.weights
        
        for i in np.unique(y):   
            plt.scatter(X[:, 0][y==i], X[:, 1][y==i], s=50 * sample_weight[y==i], alpha=self.alpha, cmap='Spectral', label=f'State {int(i)}')
        
        if class_plot == True:
            self.plot_classifier_decision_function()

        ax.set_xlabel("I (V)"), ax.set_ylabel("Q (V)")
        
        if title == None:
            kernel = self.cv_search.best_estimator_[-1].kernel
            title = self._get_file_name_from_path(self._filePath)
        
        ax.set_title(f'Classifiter training plot, kernel: {kernel}\n' + title)
        
        plt.legend()
        plt.tight_layout()
        
        if save_fig == True:
            self.save_fig(self.cv_search, name='classifier_plot',  format='.svg', dpi=600)
        plt.show()
        
        return ax
 
    def plot_testing(self, X=None, save_fig=False, title=None):
        """A function for plotting the testing of a dataset.

        Args:
            X (list, optional): The X-data to use. If None the Initial states are used: self.X_test. Defaults to None.
            save_fig (bool, optional): If True the figure is saved. Defaults to False.
            title (_type_, optional): Title of the figure. If None the data file name is used. Defaults to None.

        Returns:
            ax (object): The figure object
        """
    
        if X is None:
            X = self.X_test
        
        predcition = np.array(self.cv_search.predict(X))
        unique, counts = np.unique(predcition, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in np.unique(predcition):   
            plt.scatter(X[:, 0][predcition==i], X[:, 1][predcition==i], s=25, alpha=self.alpha, cmap='Spectral', label=f'State {int(i)}, {counts/len(predcition):.3}:.%')
        
        
        self.plot_classifier_decision_function(plot_support=False)

        ax.set_xlabel("I"), ax.set_ylabel("Q")
        
        if title == None:
            kernel = self.cv_search.best_estimator_[-1].kernel
            title = self._get_file_name_from_path(self._filePath)
        
        ax.set_title(f'Classifiter testing plot, kernel: {kernel}\n' + title )
        
        plt.legend()
        plt.tight_layout()

        if save_fig == True:
            self.save_fig(self.cv_search, name='testing_plot',  format='.svg', dpi=600)

        return ax
    
    def plot_ROC(self, X=None, y=None, save_fig=False, title=None):
        """Make ROC plots of FPR / TP.

        Args:
            X (list, optional): The X-data to use. If None the Initial states are used: self.X_test. Defaults to None.
            y (list, optional): The y-data to use. If None the Initial states are used: self.y_test. Defaults to None.
            save_fig (bool, optional): If True the figure is saved. Defaults to False.
            title (_type_, optional): Title of the figure. If None the data file name is used. Defaults to None.

        Returns:
            ax (object): The figure object
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if X is None:
            X = self.X_test
            
        if y is None:
            y = self.y_test
        
        # compute y prediction probabilities:
        y_predicted_proba = self.cv_search.predict_proba(X)
        
        n_classes = y_predicted_proba.shape[1]
        
        FPR_list, TPR_list, y_list, y_predicted_proba_list = [], [], [], []
        for i in range(n_classes):
            y_i = np.where((y == i), 1, 0)
            
            # Compute ROC curve and ROC area
            FPR, TPR, _ = roc_curve(y_i, y_predicted_proba[:, i])
            roc_auc = auc(FPR, TPR)

            FPR_list.append(FPR), TPR_list.append(TPR)
            y_list.append(y_i), y_predicted_proba_list.append(y_predicted_proba[:, i])
            
            # plot the ROC curve
            ax.plot(FPR, TPR, label=f'ROC curve state {i} (area = %0.3f)' % roc_auc)
            
            
            
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate(FPR_list))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, FPR_list[i], TPR_list[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        FPR_macro = all_fpr
        TPR_macro = mean_tpr
        roc_auc_macro = auc(FPR_macro, TPR_macro)   
                    
        # plot the macro ROC curve
        ax.plot(FPR_macro, TPR_macro, label=f'macro-average ROC curve (area = %0.3f)' % roc_auc_macro, linestyle="--")
             
        # plot the micro ROC curve
        FPR_micro, TPR_micro, _ = roc_curve(np.array(y_list).flatten(), np.array(y_predicted_proba_list).flatten())
        #FPR_micro, TPR_micro = np.array(FPR_list).flatten(), np.array(TPR_list).flatten()
        roc_auc_micro = auc(FPR_micro, TPR_micro)  
        
        ax.plot(FPR_micro, TPR_micro, label=f'micro-average ROC curve (area = %0.3f)' % roc_auc_micro, linestyle="--")
          
            
            
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.legend(loc="lower right")
        
        if title == None:
            kernel = self.cv_search.best_estimator_[-1].kernel
            title = self._get_file_name_from_path(self._filePath)
        
        ax.set_title(f'ROC plot, kernel: {kernel}\n' + title )
        
        
        plt.tight_layout()
        
        if save_fig == True:
            self.save_fig(self.cv_search, name='roc_plot',  format='.svg', dpi=600)

        return ax
    
    def plot_cv_iterations(self, score_value="mean_test_score", save_fig=False, title=None):
        """The function the plots the performance of each iteration of cross validation.

        Args:
            score_value (list, optional): The score value to plot on the y axis. If None "mean_test_score" is used. Defaults to None.
            save_fig (bool, optional): If True the figure is saved. Defaults to False.
            title (_type_, optional): Title of the figure. If None the data file name is used. Defaults to None.

        Returns:
            ax (object): The figure object
        """
        results = pd.DataFrame(self.cv_search.cv_results_)
        results["params_str"] = results.params.apply(str)
        results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
        
        mean_scores = results.pivot(index="iter", columns="params_str", values=score_value)
        ax = mean_scores.plot(legend=False, alpha=0.6, figsize=(10, 6))

        labels = [
            f"iter={i}\nn_samples={self.cv_search.n_resources_[i]}\nn_candidates={self.cv_search.n_candidates_[i]}"
            for i in range(self.cv_search.n_iterations_)
        ]

        ax.set_xticks(range(self.cv_search.n_iterations_))
        ax.set_xticklabels(labels, rotation=45, multialignment="left")
        ax.set_title("Scores of candidates over iterations")
        
        if title == None:
            kernel = self.cv_search.best_estimator_[-1].kernel
            title = self._get_file_name_from_path(self._filePath)
        
        ax.set_title(f'Scores of candidates over iterations, kernel: {kernel} \n' + title)
        
        ax.set_ylabel(score_value.replace("_"," "))
        ax.set_xlabel("iterations")
        plt.tight_layout()
        
        if save_fig == True:
            self.save_fig(self.cv_search, name='cv_iterations',  format='.svg', dpi=600)
        
        return ax
    
    def plot_oscillation(self, x=None, y=None, X=None, size=None, mode='probability', title=None, state=1):
        """Function for oscillation plot. For more information to see example "quick_run".

        Args:
            x (_type_, optional): If None: self.probability_values or self.expectation_values. Defaults to None.
            y (_type_, optional): If None: self.h5data_log['axis']. Defaults to None.
            X (list, optional): The X-data to use. If None the Initial states are used: self.X_test. Defaults to None.
            size (int, optional): The size of the data set used. Must be integer. Defaults to None.
            mode (str, optional): Can be ['probability','expectation']. Defaults to 'probability'.
            title (_type_, optional): Title of the figure. If None the data file name is used. Defaults to None.
            state (int, optional): The state to be calculated. Defaults to 1.

        Returns:
            ax (object): The figure object
        """
        if title == None:
            title = self._get_file_name_from_path(self._filePath)
            
        if size == None:
            size = self._int_states.shape[1]

        if hasattr(self, '_osc_state') == False:
            self._osc_state = None
        
        if hasattr(self, 'expectation_values') == False or size != self.size or self._osc_state != state:
            self.cal_expectation_values(X, size, state=state)

        self._osc_state = state
       
        fig, ax = plt.subplots(figsize=self.figsize)

        if y is None:
            if mode == 'expectation':
                y = self.expectation_values
                ax.set_ylabel('Expetaction Value')
                
            if mode == 'probability':
                y = self.probability_values
                if self._osc_state == 'all':
                    ax.set_ylabel(f'Probability')
                else:
                    ax.set_ylabel(f'Probability of state {int(self._osc_state)}')
        
        for i, key in enumerate(y.keys()): 
            if x is None:
                x = self.h5data_log['axis']
            
            if mode == 'probability' and self._osc_state == 'all':
                y = np.array(self.probability_values[key])
                for j in range(y.shape[1]):
                    self.do_fit_oscillation(x=x, y=y[:,j], label=key + f', state {j}', ax=ax, color=list(mcolors.TABLEAU_COLORS.values())[j])
            else:
                self.do_fit_oscillation(x=x, y=y[key], label=key, ax=ax, color=list(mcolors.TABLEAU_COLORS.values())[i])
        
        ax.set_xlabel(self.h5data_log['name'])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        fit_type = r'$A \times sin(\omega x + \varphi) + c$'
        ax.set_title(title + '\n Fit type: ' + fit_type)
        
        plt.tight_layout()
        
        return ax
     
    def plot_param_effect(self, plot_dir=None, title=None):
        """A function for plotting the effects on a score of a parameter.

        Args:
            plot_dir (dir, optional): A dir containing the score and parameter values. If None self._plot_dir is used Defaults to None.
            title (_type_, optional): Title of the figure. If None the data file name is used. Defaults to None.

        Returns:
            ax (object): The figure object
        """
        if plot_dir == None:
            try:
                plot_dir = self._plot_dir
            except:
                print('self._plot_dir is not defined. Define using self.set_plot_dir()')
        
        try:
            std = plot_dir['score_std']
        except:
            std = None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.errorbar(plot_dir['param_value'], plot_dir['score_value'], yerr=std)
      
    
        if title == None:
            title = self._get_file_name_from_path(self._filePath)
        
        ax.set_title(f'Scores of candidates over parameter\n' + title)
        
        ax.set_ylabel(plot_dir['score_name'].replace("_"," "))
        ax.set_xlabel(plot_dir['param_name'].replace("_"," "))
        plt.tight_layout()
        
        return ax
        