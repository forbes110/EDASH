import pandas as pd
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import dblquad
from dataclasses import dataclass
from scipy.linalg import det
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import r2_score
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# TODO: return modules

@dataclass
class categorical:
    pass

    def entropy(self, data_list):
        return stats.entropy(data_list)

    def kl_divergence(self, p, q):
        return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))

    def js_divergence(self, p, q):
        return 0.5* self.kl_divergence(p,q) + 0,5 * self.kl_divergence(q,p)
    
    def missing_rate(self, df):
        missing = df.isnull().sum()
        missing_rate = missing / len(df)
        total_missing_rate = missing.sum() / (len(df) * len(df.columns))
        
        result = pd.DataFrame({'missing_rate': missing_rate})
        display(result)
        print(total_missing_rate)