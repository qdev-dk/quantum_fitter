import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from ._fitting import Fitting

class Plotting(Fitting):
    def __init__(self, filePath=None, channelName=None, entries=None, state_entries=None, labels=None, 
                 size=None, scalar=True, pca=True, cv_params=None, verbose=1, kfolds=10, figsize=(10, 6),
                 alpha=0.70):
        super().__init__(filePath, channelName, entries, state_entries, labels, size, scalar, pca, cv_params, verbose, kfolds)

        self.figsize = figsize
        self.alpha = alpha
    
    def plot_classifier_decision_function(self, resolution=350, ax=None, plot_support=True):
        """Plots the decision function for a 2D decision function

        Args:
            classifier (dis): The classifier to use. Can be a pipline.
            resolution (int, optional): Resolution of the plotted decision function. Defaults to 100.
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
        """Plotting fuction for SVM fits"""
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
        
        return fig
    
    def plot_testing(self, X=None, save_fig=False, title=None):
        """Plotting new dataset to determind states"""
    
        if X is None:
            X = self.X_test
        
        predcition = np.array(self.cv_search.predict(X))
        unique, counts = np.unique(predcition, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in np.unique(predcition):   
            plt.scatter(X[:, 0][predcition==i], X[:, 1][predcition==i], s=25, alpha=self.alpha, cmap='Spectral', label=f'State {int(i)}')
        
        
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

        plt.show()

        return dict(zip(unique, counts/len(predcition)))
    
    def plot_ROC(self, X=None, y=None, save_fig=False, title=None):
        """Make ROC plots of FPR / TPR"""
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

        plt.show()

        return fig, roc_auc
    
    def plot_cv_iterations(self, score_value="mean_test_score", save_fig=False, title=None):
        """the function the plots the performance of each iteration of cross validation.

        Args:
            classifier (bool): The classifier to use.
            save_fig (bool, optional): If save_fig=True the plot is saved. Defaults to False.
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
        
        ax.set_ylabel(score_value.replace("_"," "), fontsize=15)
        ax.set_xlabel("iterations", fontsize=15)
        plt.tight_layout()
        
        if save_fig == True:
            self.save_fig(self.cv_search, name='cv_iterations',  format='.svg', dpi=600)
        
        plt.show()
    
    def plot_oscillation(self, x=None, y=None, X=None, size=None, mode='probability', title=None, state=1):
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
        
        return fig, ax
        
    def _get_file_name_from_path(self, path, part='tail'):
        import os
        head, tail = os.path.split(path)
        
        if part == 'head':
            return head
        else:
            return tail
    