import matplotlib.pyplot as plt
import Labber as lab
import numpy as np

class DataImport:
    """This class contains all importing and reformatting functions. Furthermore it controls the data size used and the import and export of classifiers.
    """
    def __init__(self, filePath=None, fileName=None, channelName=None, entries=None, state_entries=None, 
                 labels=None, size=None, kfolds=10, data=None):
        """Initializes a instance of the class. If no entries are given it defines the best entries for classification from the mean. 

        Args:
            filePath (string, optional): File path for the Labber (h5file) file containing the IQ data. Defaults to None.
            channelName (string, optional): Channel name contained IQ data. If None the first trace in the file is used. Defaults to None.
            state_entries (list, optional): A list containing the wanted entries for classification. If None the two entries with the largest and smallest mean are used. Defaults to None.
            labels (list, optional): A list containing the labels for the states. Labels can be integers or a strings. If None numbers from 0 to len(number of states) is used. Defaults to None.
            size (int, optional): The size of the data set used. Must be integer. Defaults to None.
            kfolds (int, optional): Number of splits in the dataset. Used for crossvalidation. Defaults to 10.
        """
        
        self._filePath = filePath
        
        if fileName == None:
            self._fileName = self._get_file_name_from_path(self._filePath)
        else:
            self._fileName = fileName
            
        self._data = np.array(data)
        
        if self._filePath != None:
            self.h5file = lab.LogFile(self._filePath)
            self.channeldict = self.h5file.getChannelValuesAsDict()
            
        self.size = size
        self.kfolds = kfolds
        
        self.kernel = None
        self.classifer = None
        self.set_data(channelName=channelName)
        
        if self._filePath != None:
            # define states
            self.data_mean = self._min_max_index()
        
            if state_entries:
                self.state_entries = state_entries
            else:
                self.state_entries = self.data_mean[1]
                
                
        self.set_states(state_entries=entries, labels=labels)  
            
    def set_data(self, channelName=None, state_entries=None, channelName_log='Pulse Generator - Amplitude', unit_log='V'):
        """Selects the data set which is gonna be used in the further calculations.

        Args:
            channelName (string, optional): Channel name contained IQ data. If None the first trace in the file is used. Defaults to None.
            state_entries (list, optional): A list containing the wanted entries for classification. If None the two entries with the largest and smallest mean are used. Defaults to None.
            channelName_log (str, optional): The units scale to be used instead of index. Defaults to 'Pulse Generator - Amplitude'.
            unit_log (str, optional): The units of the unit scale. Defaults to 'V'.
        """
        import h5py
        if self._filePath != None:
            h5data = self.h5file.getData(channelName, entry=state_entries)
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
        """Sets the entries of the "cleanest" states of the data set to be used in further calculations.

        Args:
            state_entries (list, optional): A list containing the wanted entries for classification. If None the two entries with the largest and smallest mean are used. Defaults to None.
            labels (list, optional): A list containing the labels for the states. Labels can be integers or a strings. If None numbers from 0 to len(number of states) is used. Defaults to None.
            offset (float), optional): The offset value offsets the last state in the array. This argument is added to simulate three state data. Defaults to None.
        """
        if state_entries:
            self.state_entries = state_entries
        
        if self._data.any() != None:
            self._int_states = self._data
        else:
            self._int_states = self.h5data[np.array(self.state_entries)]
        
        if offset:
            self._int_states[-1] = self._int_states[-1] + offset
        
        if labels != None: 
                self._states_labels = labels 
        else: 
            self._states_labels = range(len(self._int_states))
        
        self.set_dataset_size()
                 
    def set_dataset_size(self, size=None, X=None):
        """Sets the size of all entries in the data set.

        Args:
            size (int, optional): The size of the data set used. Must be integer. Defaults to None.
            X (list, optional): A list of data. If None the selected initial states are used. Defaults to None.

        Returns:
            List: Dataset shortened by size. (X[:size])
        """
        if size:
            self.size = int(size)
         
        X_temp = False    
        if X is None:
            X_temp = True
            X = self._int_states
            
        state_list, label_list  = [], []
        for i, j in enumerate(X):
            state_list.append(j[:self.size])
            if X_temp == True: 
                label_list.append(np.ones(np.array(state_list[i]).shape[0]) * self._states_labels[i])
            
        if X_temp == True:  
            self._states_X = np.concatenate(state_list)
            self._states_target = np.concatenate(label_list)
        else:
            return state_list

        self._split_data()
    
    def _split_data(self):
        """Split dataset of initial states it into training and testing parts. The size of the test sets will be 1/k_folds, where k_folds = 10 as default.
        """
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._states_X, self._states_target, 
                                                            test_size=(1/self.kfolds), random_state=0)
        
        self._matrix_weights()
        
    def _min_max_index(self, X=None):
        """Max/min function used to determine the initial states, by calculating the mean of the data in the IQ plane for all entries and Finding the index for the min and max value.

        Args:
            X (list, optional): The IQ data to examinen. If None the default data is used: self.h5data. Defaults to None.

        Returns:
            Data_mean (float): The meaning of all the data entries.
            Max/min (list): A list containing the min and max index. Formate: [min, max]
        """
        if X is None:
            X = self.h5data
            
        data_mean = np.linalg.norm(X, axis=2)
        data_mean = np.sum(data_mean, axis=1)
        
        min_, max_ = np.argmin(data_mean), np.argmax(data_mean)
        return data_mean, [min_, max_]
    
    def _matrix_weights(self, X=None, y=None, A=0.1, B=1, bins=100, r=5, plot=False):
        """Calculate a waited matrix for the initial states. This is used to optimize the wall clock time of the calculation.

        Args:
            X (list, optional): The X-data to use. If None the Initial states are used: self.X_train. Defaults to None.
            y (list, optional): The y-data to use. If None the Initial states are used: self.y_train. Defaults to None.
            A (float, optional): The minimum of the final normalization of the array. Defaults to 0.1.
            B (float, optional): The maximum of the final normalization of the array. Defaults to 1.
            bins (int, optional): Number of bins in the generated matrix. This is a measure of resolution. Defaults to 100.
            r (int, optional): Number of beans to some in the calculation. This helps to smooth loud the matrix. Defaults to 5.
            plot (bool, optional): If True matrices are plotted. Defaults to False.
        """
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
        """A function to export the classifier after being fitted or defined. It uses the pickle format, and prints a statement if successfully saved: "Saved the pickle".

        Args:
            filePath (string, optional): The file path to safe to pickle file in. If None the datafile path is used. Defaults to None.
            fileName (string, optional): The filename of the exported file. If None the datafilename is used, but as a .pickle file. Defaults to None.
            fileSave (string, optional): The part to save. If 'all' the entire self saved. Defaults to 'all'.
            overwrite (bool, optional): If False And a pickle file with the same name is in the folder a counter (_000) is added to the name. . Defaults to True.
        """
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
        """A function to import the classifier. It uses the pickle format, and prints a statement if successfully imported: "Got your pickle!". It's not successful: "Unable to locate your pickle".

        Args:
            filePath (string, optional): The file path for the file to be imported. If None the folder of the data is being searched. Defaults to None.

        Returns:
            Classifier (self): The classifier object
        """
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
    """A function to import the classifier. It uses the pickle format, and prints a statement if successfully imported: "Got your pickle!". It's not successful: "Unable to locate your pickle". This can be used without a instance.

        Args:
            filePath (string, optional): The file path for the file to be imported. If None the folder of the data is being searched. Defaults to None.

        Returns:
            Classifier (self): The classifier object
        """
    import pickle
    
    try: 
        with open(filePath, 'rb') as handle:
            classifier = pickle.load(handle)
        
        print('Got your pickle!')
        return classifier
    
    except:
        print('Unable to locate your pickle')
        
     