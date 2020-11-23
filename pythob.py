import pandas as pd
import matplotlib.pyplot as plt
import nltk
import sklearn
import wordcloud
import chardet
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from sklearn import metrics
df = pd.read_csv(r"C:\Users\aniru\Desktop\parkinsons.csv")

x = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]].values
y = df.iloc[:,7].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=0)


#featurescaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test= pca.transform(x_test)
variance = pca.explained_variance_ratio_
#applying Knearest Neighbours

from sklearn.neighbors import KNeighborsClassifier
classfi1 = KNeighborsClassifier(n_neighbors=8,p=2,)
classfi1.fit(x_train,y_train)

#predicting results
y1_pred = classfi1.predict(x_test)

#accuracy metrics

from sklearn.metrics import  accuracy_score
accuracy_score(y_test,y1_pred)

#fitting into svm

from sklearn.svm import SVC
classifi2 = SVC()
classifi2.fit(x_train,y_train)

#prediction for svm

y2_pred = classifi2.predict(x_test)

#accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y2_pred)

#random forest classifier

from sklearn.ensemble import RandomForestClassifier
classifi3 = RandomForestClassifier(n_estimators=16, criterion="entropy", random_state=0)
classifi3.fit(x_train,y_train)

#predicting results

y3_pred = classifi3.predict(x_test)

#metrics

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y3_pred)
cm = confusion_matrix(y_test,y3_pred)







