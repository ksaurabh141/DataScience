# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 22:08:14 2018

@author: Saurabh
"""

import pandas as pd
import os
from sklearn import tree
import io
import pydot
from sklearn import model_selection
os.chdir("C:/Users/Saurabh/Desktop/Ghosts")
os.environ['PATH'] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"
ghost_train = pd.read_csv("C:/Users/Saurabh/Desktop/Ghosts/train.csv")

ghost_train.shape
ghost_train.info()
ghost_train.describe

ghost_train1 = pd.get_dummies(ghost_train, columns = ['color'])
ghost_train1.shape 
ghost_train1.info()
ghost_train1.describe

x_ghost = ghost_train1.drop(['id','type'],1)
y_ghost = ghost_train[['type']]

dt = tree.DecisionTreeClassifier()
#param_grid = {'max_depth' :[15,200], 'min_samples_split':[2,6], 'criterion':['gini','entropy']}
param_grid = {'max_depth' :[10,15]}

dt.fit(x_ghost,y_ghost)
dt_grid = model_selection.GridSearchCV(dt,param_grid,cv=15,n_jobs= 9)
dt_grid.fit(x_ghost,y_ghost)
dot_data = io.StringIO()
tree.export_graphviz(dt_grid, out_file = dot_data, feature_names = x_ghost.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) [0]
graph.write_pdf("ghost1.pdf")


dt_grid.grid_scores_
dt_grid.best_params_
dt_grid.best_score_
dt_grid.score(x_ghost,y_ghost)

ghost_test = pd.read_csv("C:/Users/Saurabh/Desktop/Ghosts/test.csv")
ghost_test.shape
ghost_test.info()
# =============================================================================
# ghost_test.bone_length[ghost_test['bone_length'].isnull()] = ghost_test['bone_length'].mean()
# ghost_test.rotting_flesh[ghost_test['rotting_flesh'].isnull()] = ghost_test['rotting_flesh'].mean()
# ghost_test.hair_length[ghost_test['hair_length'].isnull()] = ghost_test['hair_length'].mean()
# ghost_test.has_soul[ghost_test['has_soul'].isnull()] = ghost_test['has_soul'].mean()
# ghost_test.color[ghost_test['color'].isnull()] = ghost_test['color'].mode()
# =============================================================================

ghost_test1 = pd.get_dummies(ghost_test, columns = ['color'])
ghost_test1.shape
ghost_test1.info()
x_test= ghost_test1
x_test.shape
ghost_test['type'] = dt_grid.predict(x_test)
ghost_test.to_csv("Submission_ghost1.csv", columns=['id','type'], index= False)





