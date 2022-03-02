import matplotlib.pyplot as plt

import Labber as lab
import numpy as np

class DataImport:
    def __init__(self, filePath=None, channelName=None, entries=None, state_entries=None, 
                 labels=None, size=None, kfolds=10):
        
        self._filePath = filePath
        self.h5file = lab.LogFile(self._filePath)
        self.channeldict = self.h5file.getChannelValuesAsDict()
        
        self.size = size
        self.kfolds = kfolds
        
        self.kernel = None
        self.classifer = None
        self.set_data(channelName=channelName, entries=entries)
    
        # define states
        self.data_mean = self._min_max_index()
    
        if state_entries:
            self.state_entries = state_entries
        else:
            self.state_entries = self.data_mean[1]
          
        self.set_states(state_entries=self.state_entries, labels=labels)  
            
    def set_data(self, channelName=None, entries=None, channelName_log='Pulse Generator - Amplitude', unit_log='V'):
        import h5py
        
        h5data = self.h5file.getData(channelName, entry=entries)
        self.h5data = self._reformate(h5data)
        
        self.h5data_index = {'name' : 'Index', 
                             'axis' : range(self.h5file.getNumberOfEntries())}
        
        try:
            h5data_temp = h5py.File(self._filePath, 'r')
            h5datalog = h5data_temp['Data']['Data'][:]
            self.h5data_log = {'name' : f'{channelName_log} [{unit_log}]', 
                               'axis' : h5datalog[:,0,0]}
        except:
            print('channelName_log not in h5data. Setting Pulse Generator Amplitude to index')
            self.h5data_log = self.h5data_index
             
    def set_states(self, state_entries=None, labels=None, offset=None):
        if state_entries:
            self.state_entries = state_entries
        
        self._int_states = self.h5data[np.array(self.state_entries)]
        
        if offset:
            self._int_states[-1] = self._int_states[-1] + offset
        
        if labels != None: 
                self._states_labels = labels 
        else: 
            self._states_labels = range(len(self.state_entries))
        
        self.set_dataset_size()
                 
    def set_dataset_size(self, size=None, X=None):
        if size:
            self.size = size
         
        X_temp = False    
        if X is None:
            X_temp = True
            X = self._int_states
            
        state_list, label_list  = [], []
        for i, j in enumerate(X):
            state_list.append(j[:self.size])
            if X_temp == True: 
                label_list.append(np.ones(state_list[i].shape[0]) * self._states_labels[i])
            
        if X_temp == True:  
            self._states_X = np.concatenate(state_list)
            self._states_target = np.concatenate(label_list)
        else:
            return state_list

        self._split_data()
    
    def _split_data(self):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._states_X, self._states_target, 
                                                            test_size=(1/self.kfolds), random_state=0)
        
        self._matrix_weights()
        
    def _reformate(self, X): 
        h5data_reformated = []
        for i in range(len(X)):
            h5data_reformated.append(np.column_stack((X[i].real, X[i].imag)))
        return np.array(h5data_reformated)
        
    def _min_max_index(self, X=None):
        if X is None:
            X = self.h5data
            
        data_mean = np.linalg.norm(X, axis=2)
        data_mean = np.sum(data_mean, axis=1)
        
        min_, max_ = np.argmin(data_mean), np.argmax(data_mean)
        return data_mean, [min_, max_]
    
    def _matrix_weights(self, X=None, y=None, A=0.1, B=1, bins=100, r=5, plot=False):
        """Creating the weighting array"""
        def matrix_avg(M, r=1, A=0, B=1): 
            M_sum = np.full(M.shape, 0.0)
           
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    n0 = n1 = r
                        
                    if (i-n0) <= -1: 
                        n0 = i
                        
                    if (j-n1) <= -1: 
                        n1 = j  
                
                    Mi = M[i-n0:i+r+1, j-n1:j+r+1]
                
                    M_sum[i,j] = np.sum(Mi) / Mi.size
            M_sum = np.interp(M_sum, (M_sum.min(), M_sum.max()), (A, B))
            return M_sum 
        
        if (X is None):
            X = self.X_train
        
        if (y is None):
            y = self.y_train
        
        xy = [X[y == label] for label in np.unique(y)]
        
        x, y = [], []
        for i in range(len(xy)):
            x.append(xy[i][:, 0])
            y.append(xy[i][:, 1])

        x_con, y_con = np.concatenate(x), np.concatenate(y)
        lim = [[min(x_con), max(x_con)], [min(y_con), max(y_con)]]

        data_matrix = []
            
        for i in range(len(xy)):
            data_matrix_i, xaxis, yaxis = np.histogram2d(x[i], y[i], bins=bins, range=lim)
            data_matrix_i = matrix_avg(data_matrix_i, r=r)
            data_matrix.append(data_matrix_i)
        
        background_matrix = np.minimum.reduce(data_matrix)
        
        if plot == True:
            plt.matshow(background_matrix, cmap=plt.cm.viridis)
            plt.show()
            
        weights = np.empty(0)
        for i in range(len(xy)):
            data_matrix[i] = data_matrix[i] - background_matrix
            if plot == True:
                plt.matshow(data_matrix[i], cmap=plt.cm.viridis)
                plt.show()
                
            weights_i = np.empty(0)
            for j in range(len(xy[i])):
                    index_x = np.absolute(xaxis - xy[i][j, 0]).argmin()
                    index_y = np.absolute(yaxis - xy[i][j, 1]).argmin()

                    weights_i = np.append(weights_i, data_matrix[i][index_x - 1, index_y - 1])

            weights = np.hstack([weights, weights_i])
        self.weights = np.interp(weights, (weights.min(), weights.max()), (A, B))

    def export_classifier(self, filePath=None, fileName=None, fileSave='all', overwrite=True):
        import pickle
        import os.path

        # save type
        if fileSave == 'classifier':
            fileSave = self.cv_search
            
        elif fileSave == 'all' or fileSave == None:
            fileSave = self

        else:
            fileSave = self.fileSave
        
        # save path
        if filePath == None:
            filePath = self._get_file_name_from_path(self._filePath, part='head')
            
        if filePath.endswith('/') == False:
            filePath += '/'
    
        # save name
        if fileName == None:
            fileName = self._get_file_name_from_path(self._filePath).replace('.hdf5','')
    
        # combine
        file = filePath + fileName
        
        # adds number
        if overwrite == False:
            i = 0
            file_temp = file
            while os.path.exists(file + '.pickle') == True:
                file = file_temp + f'_{str(i).zfill(3)}'
                i += 1
            
        with open(filePath + fileName + '.pickle', 'wb') as handle:
            pickle.dump(fileSave, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Saved the pickle')

    
    def import_classifier(self, filePath=None):
        import pickle
        import os.path
        
        try: 
            filePath = self._get_file_name_from_path(self._filePath, part='head')
            fileName = self._get_file_name_from_path(self._filePath).replace('.hdf5','')
            
            if filePath.endswith('/') == False:
                filePath += '/'
            
            # combine
            file = filePath + fileName
            
            i = 999
            file_temp = file
            while os.path.exists(file + '.pickle') == False and i>0:
                
                file = file_temp + f'_{str(i).zfill(3)}'
                i -= 1
            
        
            with open(file + '.pickle', 'rb') as handle:
                classifier = pickle.load(handle)
            
            print('Got your pickle!')
            return classifier
        
        except:
            print('Unable to locate your pickle')

def import_classifier(filePath):
    import pickle
    
    try: 
        with open(filePath, 'rb') as handle:
            classifier = pickle.load(handle)
        
        print('Got your pickle!')
        return classifier
    
    except:
        print('Unable to locate your pickle')
        
     