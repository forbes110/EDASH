from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class FCMeans:
    # input parameters
    data: np.ndarray
    num_clusters: int
    m: float = 2.0
    epsilon: float = 0.0001
    max_iter: int = 1000
    
    def __post_init__(self):
        self.num_samples = self.data.shape[0]
        self.num_features = self.data.shape[1]
        
        # clusters * each feature
        self.centroids = np.random.rand(self.num_clusters, self.num_features)
        
        # membership matrix U
        self.membership_matrix = np.random.rand(self.num_samples, self.num_clusters)
        self.previous_membership_matrix = np.zeros((self.num_samples, self.num_clusters))
        
    def fit(self):
        for i in range(self.max_iter):
            self.update_membership_matrix()
            self.update_centroids()
            
            # converge to epsilon, stop iterations
            if np.allclose(self.membership_matrix, self.previous_membership_matrix, atol=self.epsilon):
                break
            
            self.previous_membership_matrix = self.membership_matrix.copy()
            
            
    def update_membership_matrix(self):
        for i in range(self.num_samples):
            for j in range(self.num_clusters):
                
                # use L2 norm as distance
                numerator = np.linalg.norm(self.data[i] - self.centroids[j])
                dummy = 0.0
                
                for k in range(self.num_clusters):
                    
                    denominator = (np.linalg.norm(self.data[i] - self.centroids[k]))
                    # dummy += (numerator / denominator)** (2 / (self.m - 1))
                    if denominator == 0:
                        dummy += (numerator / (denominator+1e-5) )** (2 / (self.m - 1))
                
                        # print('dummy:', dummy)

                    else:
                        dummy += (numerator / denominator)** (2 / (self.m - 1))
                        # print('dummy:', dummy)
                        
                # self.membership_matrix[i][j] = 1.0 / dummy 
                if dummy == 0:
                    self.membership_matrix[i][j] = 1.0 / (dummy+1e-5) 
                else:
                    self.membership_matrix[i][j] = 1.0 / dummy 
                
    def update_centroids(self):
        for i in range(self.num_clusters):
            numerator = 0.0
            denominator = 0.0
            for j in range(self.num_samples):
                numerator += (self.membership_matrix[j][i]) * self.data[j]
                denominator += (self.membership_matrix[j][i])
            self.centroids[i] = numerator / denominator
            
    def get_centroids(self):
        return self.centroids
    
    def get_membership_matrix(self):
        return self.membership_matrix
    
    def inference(self, data, centroids, feature_id):   
        inferenced_data = 0
        for j in range(self.num_clusters):
            numerator = np.linalg.norm(data - centroids[j])
            dummy = 0.0
                
            for k in range(self.num_clusters):
                dummy += (numerator / np.linalg.norm(data - centroids[k]))** (2 / (self.m - 1))
                
            membership_value = 1.0 / dummy
            inferenced_data += membership_value* self.centroids[j][feature_id]
            
        
        return inferenced_data
            
@dataclass
class FCMImputer:
    '''            
        input data and parameters of fuzzy cmeans
        output imputed data
        TODO
            no complete data control
    '''
    data: None
    num_clusters: int
    
    # weiting factor
    m: float = 2.0
    
    # acceptable error
    epsilon: float = 1e-4
    max_iter: int = 100
    detection = None
    
    
    
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
        '''
        Let the centriods of fuzzy cmeans result be c_j for j... n_clusters, 
        membership_function u_ij be data points x_i to c_j for i ... n_data points 
        the estimated result is estimated_result of  x_i's incomplete feature be sum of u_ij *(c_j's incomplete) feature forall j
        '''
        fcm = FCMeans(
            data = self.complete_data,
            num_clusters = self.num_clusters,
            m = self.m,
            epsilon = self.epsilon,
            max_iter = self.max_iter
        )
        
        # get fuzzy c means results
        fcm.fit()
        centroids = fcm.get_centroids()
        
        inferenced_rows = []
        
        # with trained fcm, inference the implete data without the missing features
        for incomplete_row in self.incomplete_rows:
            
            _incomplete_row = self._data[incomplete_row].copy()
            missing_features_ids = np.where(np.isnan(_incomplete_row))[0]
            
            valid_data = np.delete(_incomplete_row, missing_features_ids)
            valid_centroids = self.processing_centroids(centroids, missing_features_ids)
            
            for missing_feature_id in missing_features_ids:
                inferenced_feature = fcm.inference(valid_data, valid_centroids, missing_feature_id)
                _incomplete_row[missing_feature_id] = inferenced_feature
            
            inferenced_rows.append(_incomplete_row)
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
    def processing_centroids(self, centroids, missing_features_ids)->list:
        '''
        elimiate all missing features from each centroids
        '''
        valid_centroids = []
        for centroid in centroids:
            valid_centroids.append(np.delete(centroid, missing_features_ids)[0])
        return valid_centroids

    def _extract_rows(self):
        '''
        Extract rows with missing features and complete ones
        eliminate rows with all of features missing
        '''
        
        # rows with all nan's need to be removed.
        all_nan_rows = np.where(np.isnan(self._data).all(axis=1))[0]
        if len(all_nan_rows) != 0:
            print(f' There are {len(all_nan_rows)} rows in data with all Nan entries, the imputed data wouldn\'t contain the rows')
            self._data = np.delete(self._data, all_nan_rows, axis=0) 
            
        
        complete_rows, incomplete_rows = [], []
        incomplete_rows = np.where(np.isnan(self._data).any(axis=1))[0]
        if self.detection == True:
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
    data = random_data(42, 0.1, 100, 8)
    data[0:70, [1, 3, 6]] = np.nan
    print('============================================================================================')
    print(data)
    
    
    fcmImputer = FCMImputer(data = data, num_clusters = 3)
    # a = data[~np.isnan(data).all(axis=1)]
    # # print(len(np.where(np.isnan(a).any(axis=1))[0]))
    after = fcmImputer.impute()
    print('============================================================================================')
    
    print(after)
    print(len(np.where(np.isnan(data).any(axis=1))[0]))
    print(len(np.where(np.isnan(after).any(axis=1))[0])) 

    # print(data)
    # print(after)