import pandas as pd
from sklearn import preprocessing
from Cascade_Forest.Layer import layer
from Base_module import Base_estimators as Be
from sklearn.model_selection import train_test_split
import os
le = preprocessing.LabelEncoder()

"Get the data"
data = pd.read_csv(os.getcwd()+"/yeast_new.csv")
train_data = data.iloc[:, 0:-14].to_numpy()
label = le.fit_transform(data['Class1'].to_numpy())
x = train_data
y = label

"Spilt the data"
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=None, stratify=y)


"Generate 4 random forest classifier and 4 completely random forest classifier"
clf1 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=-1)
clf2 = Be.RFC(n_estimators=25, max_depth=None, max_features=1, random_state=None,
              n_jobs=-1)
clf3 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=-1)
clf4 = Be.RFC(n_estimators=25, max_depth=None, max_features=1, random_state=None,
              n_jobs=-1)
clf5 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=-1)
clf6 = Be.RFC(n_estimators=25, max_depth=None, max_features=1, random_state=None,
              n_jobs=-1)
clf7 = Be.RFC(n_estimators=25, max_depth=None, random_state=None, n_jobs=-1)
clf8 = Be.RFC(n_estimators=25, max_depth=None, max_features=1, random_state=None,
              n_jobs=-1)

"Generate 10 layer and filled with classifer's set"
c = {"crf1": clf1, "crf2": clf3, "crf3": clf5, "crf4": clf7, "crf5": clf2, "crf6": clf4, "crf7": clf6, "crf8": clf8}
layer1 = layer()
layer1.add(**c)
layer2 = layer()
layer2.add(**c)
layer3 = layer()
layer3.add(**c)
layer4 = layer()
layer4.add(**c)
layer5 = layer()
layer5.add(**c)
layer6 = layer()
layer6.add(**c)
layer7 = layer()
layer7.add(**c)
layer8 = layer()
layer8.add(**c)
layer9 = layer()
layer9.add(**c)
layer10 = layer()
layer10.add(**c)

from Cascade_Forest.Cascade_Forest import cascade_forest



"Initialize cascade forest structure, you want save the model generated in the validation step into the 'yeast' directory"

cs1 = cascade_forest(random_state=None, n_jobs=-1, directory='yeast4', metrics='accuracy')
cs2 = cascade_forest(random_state=None, n_jobs=-1, directory='yeast4', metrics='accuracy')
cs3 = cascade_forest(random_state=None, n_jobs=-1, directory='yeast4', metrics='accuracy')

"Add each layer to cascade forest structure"
cs1.add(layer1)
cs1.add(layer2)
cs1.add(layer3)
cs1.add(layer4)
cs1.add(layer5)
cs1.add(layer6)
cs1.add(layer7)
cs1.add(layer8)
cs1.add(layer9)
cs1.add(layer10)

cs2.add(layer1)
cs2.add(layer2)
cs2.add(layer3)
cs2.add(layer4)
cs2.add(layer5)
cs2.add(layer6)
cs2.add(layer7)
cs2.add(layer8)
cs2.add(layer9)
cs2.add(layer10)

cs3.add(layer1)
cs3.add(layer2)
cs3.add(layer3)
cs3.add(layer4)
cs3.add(layer5)
cs3.add(layer6)
cs3.add(layer7)
cs3.add(layer8)
cs3.add(layer9)
cs3.add(layer10)






#################  Cascade Forest   ###################
cs3.fit([X_train], y_train)
cs3.predict([X_test])
score = cs3.score(y_test)
print("cascade forest's F1 :{:.2f} %".format(score * 100))
#######################################################3


import networkx as nx

"Read the graph input"
G1 = nx.read_edgelist('yeast_edge_bnlearn.txt', create_using=nx.Graph(), nodetype=None, data=[('weight', int)])

from Cascade_Forest.Multi_grained_scanning import scanner

"Generate the scanner"
sc1 = scanner(stratify=True, clf_set=(clf1, clf2), n_splits=3, random_state=None,
              walk_length=(24, 48, 96), num_walks=1, p=100, q=100, scale=(80, 56, 8))

sc2 = scanner(window_size=(24, 48, 96), stratify=True, clf_set=(clf1, clf2), n_splits=3, random_state=None)

################## graph-based gcForest ########################

"Using the graph-based approach to scan the data"
transformed_train1, transformed_test1 = sc1.graph_embedding(G1, X_train, y_train, X_test)
"Training"
cs1.fit(transformed_train1, y_train)
"Predicting"
cs1.predict(transformed_test1)
"The accuracy score"
score = cs1.score(y_test)
print("Graph_based gcForest's accuracy :{:.2f} %".format(score * 100))
#################################################################

###############   original gcForest  ############################
"Using the window sliding approach to scan the data"
transformed_train2, transformed_test2 = sc2.transform_feature(X_train, y_train, X_test)
"Training"
cs2.fit(transformed_train2, y_train)
"Predicting"
cs2.predict(transformed_test2)
"The accuracy score"
score = cs2.score(y_test)
print("window sliding gcForest's accuracy :{:.2f} %".format(score * 100))