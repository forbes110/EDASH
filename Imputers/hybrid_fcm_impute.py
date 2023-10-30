from dataclasses import dataclass
import numpy as np
import random
from typing import *
from svr_impute import SVRImputer
from fcm_impute import FCMImputer
from ga import GeneticAlgorithm



@dataclass
class HybridFCMImputer:
    data: None
    def impute(self):
        print('------------------------------------in1------------------------------------')
        
        fcmImputer = FCMImputer(data = self.data, num_clusters = 3, m = 3)
        
        print('------------------------------------in2------------------------------------')
        svmImputer = SVRImputer(data = self.data)
        
        svm_output = svmImputer.impute()
        iter = 0
        while True or iter > 100:
            ga = GeneticAlgorithm(fcmImputer, svmImputer)
            num_clusters, m = ga.run()
            fcmImputer = FCMImputer(data = self.data, num_clusters = num_clusters, m = m)
            
            fcm_output = fcmImputer.impute()
            
            # MSE
            error = np.power(fcm_output - svm_output, 2).sum()
            if error < 1:
                final_output = fcm_output
                break
            iter += 1
            print(iter)
        return final_output
    
    
    
def random_data(seed = 42, upperbound = 0.5, num = 100, features = 2):   
    '''
    Generate random data
    '''
    np.random.seed(seed)
    data = np.random.rand(num, features)
    # data[data < upperbound] = np.nan
    return data
    
    
    
if __name__ == '__main__':
    data = random_data(42, 0.1, 200, 8)
    data[0:177, [1, 3, 6]] = np.nan
    print('============================================================================================')
    print(data)
    
    
    hfcmImputer = HybridFCMImputer(data = data)
    # a = data[~np.isnan(data).all(axis=1)]
    # # print(len(np.where(np.isnan(a).any(axis=1))[0]))
    after = hfcmImputer.impute()
    print('============================================================================================')
    
    print(after)
    print(len(np.where(np.isnan(data).any(axis=1))[0]))
    print(len(np.where(np.isnan(after).any(axis=1))[0]))