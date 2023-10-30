from sklearn.svm import SVR
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SVRImputer:
    data: None
    
    def __post_init__(self):
        self._data = self.translation(self.data)
        self.complete_rows, self.incomplete_rows = self._extract_rows()
        self.complete_data = np.array([self._data[x] for x in self.complete_rows])
    def translation(self, x):
        if type(x) == pd.DataFrame:
            return x.to_numpy()
        else:
            return x
    def impute(self):
        # the params of SVR need to be tuned by optuna        
        
        inferenced_rows = []
        # impute incomplete rows based on existing features
        for incomplete_row in self.incomplete_rows:
            
            # make copy to prevent overwriting
            _incomplete_row = self._data[incomplete_row].copy()
            missing_features_ids = np.where(np.isnan(_incomplete_row))[0]
            
            # datarow with only existing features
            valid_data = np.delete(_incomplete_row, missing_features_ids)
            
            x_train = np.delete(self.complete_data, missing_features_ids, axis=1)
            
            # predict missing features
            for missing_feature_id in missing_features_ids:
                
                y_train = self.complete_data[:, missing_feature_id].reshape(-1,1)
                
                svrModel = SVR(kernel='rbf', C=10)
                
                # .ravel will convert that array shape to (n, ) (i.e. flatten it)
                svrModel.fit(x_train, y_train.ravel())
                
                # reshape to make single sample [1,2,3,...] be [[1,2,3,...]]
                inferenced_feature = svrModel.predict(valid_data.reshape(1, -1))

                # add predicted features back to _incomplete_row
                _incomplete_row[missing_feature_id] = inferenced_feature
            
            inferenced_rows.append(_incomplete_row)

        # merge all rows            
        imputed_data = self.merge_inferenced_rows(inferenced_rows)
        
        return imputed_data  
        
    
    def merge_inferenced_rows(self, inferenced_rows):
        '''
        Merged imputed data rows and complete data rows
        '''
        
        merged_data = self._data.copy()
        for i in range(self._data.shape[0]):
            if i in self.incomplete_rows:
                merged_data[i] = inferenced_rows.pop(0)
        return merged_data
        
    
        
        
    def _extract_rows(self):
        '''
        Extract rows with missing features and complete ones
        eliminate rows with all of features missing
        '''
        
        all_nan_rows = np.where(np.isnan(self._data).all(axis=1))[0]
        if len(all_nan_rows) != 0:
            print(f' There are {len(all_nan_rows)} rows in data with all Nan entries, the imputed data wouldn\'t contain the rows')
            self._data = np.delete(self._data, all_nan_rows, axis=0) 
        
        complete_rows, incomplete_rows = [], []
        incomplete_rows = np.where(np.isnan(self._data).any(axis=1))[0]
        print(f' There are {len(incomplete_rows)} imcomplete rows in data')
        
        
        complete_rows = np.where(~np.isnan(self._data).any(axis=1))[0]
        return np.array(complete_rows), np.array(incomplete_rows)



def random_data(seed = 42, upperbound = 0.5, num = 100, features = 2):   
    '''
    Generate random data
    '''
    np.random.seed(seed)
    data = np.random.rand(num, features)
    # data[data < upperbound] = np.nan
    return data

    
    
if __name__ == '__main__':
    data = random_data(42, 0.1, 5, 8)
    data[0:2, [1, 3, 6]] = np.nan
    print('============================================================================================')
    print(data)
    
    
    svmImputer = SVRImputer(data = data)
    # a = data[~np.isnan(data).all(axis=1)]
    # # print(len(np.where(np.isnan(a).any(axis=1))[0]))
    after = svmImputer.impute()
    print('============================================================================================')
    
    print(after)
    print(len(np.where(np.isnan(data).any(axis=1))[0]))
    print(len(np.where(np.isnan(after).any(axis=1))[0]))