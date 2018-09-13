# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:25:55 2018

@author: Saurabh
"""

from sklearn import decomposition
import seaborn as sns
import pandas as pd

#Highly correlated columns (X1, X2)
df1 = pd.DataFrame({'Age':[10, 20, 30, 40],'Fare':[15, 25, 35, 45], })
sns.jointplot('Age','Fare',df1)

#Standardize data
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# X_std = StandardScaler().fit_transform(df1)
# =============================================================================


#==============================================================================
#Finding Co-Variance Matrix with user defined function
import numpy as np
mean_vec = np.mean(df1, axis=0)
cov_mat = (df1 - mean_vec).T.dot((df1 - mean_vec)) / (df1.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

#Or We can try this
#Finding Co-Variance Matrix with built in function
print('NumPy covariance matrix: \n%s' %np.cov(df1.T))
#==============================================================================

#Next, we perform an eigendecomposition on the covariance matrix
#==============================================================================
cov_mat = np.cov(df1.T)

#
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print(eig_vals)# Eigen Value
print(eig_vecs)# Eigen Vector

#Selecting Principal Components
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
