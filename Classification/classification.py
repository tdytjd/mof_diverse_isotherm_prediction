import pandas as pd
import numpy as np
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix


#classify
#train on set1

DataMatrix_set1 = pd.read_csv('Classification_set1.csv')
DataMatrix_set2 = pd.read_csv('Classification_set2.csv')




#generate training+Val/test
train_valid, test = train_test_split(DataMatrix_set1, test_size = .2, stratify = DataMatrix_set1.molecule, random_state = 0)
descriptor_columns=['LCD','PLD','VF','Tc','Pc','w','logK']


#training+validation
#X = train_valid.iloc[:,2:-2].values
X = train_valid.loc[:,descriptor_columns].values

y = train_valid.iloc[:,-1].values

#testset
#X_test = test.iloc[:,2:-2].values
X_test = test.loc[:,descriptor_columns].values

y_test = test.iloc[:,-1].values

gammas = np.array([1e-1,1e-2,1e-3,1e-4,1e-5])
Cs = np.array([1e1,1e2,1e3,1e4,1e5])
parameter_ranges = {'gamma':gammas,'C':Cs}
svc = SVC(kernel='rbf')
svc_search = GridSearchCV(svc, parameter_ranges, cv=3)
svc_search.fit(X,y)
print('gridsearchcv round 1 best ',svc_search.best_estimator_, svc_search.best_score_)


scale=[0.1,0.25,0.5,0.75,1,2.5,5,7.5,10]
gammas_rd2 = [i * svc_search.best_estimator_.gamma for i in scale]
Cs_rd2 = [i * svc_search.best_estimator_.C for i in scale]


parameter_ranges = {'gamma':gammas_rd2,'C':Cs_rd2}
svc_rd2 = SVC(kernel='rbf')
svc_search_rd2 = GridSearchCV(svc_rd2, parameter_ranges, cv=3, refit = True)
svc_search_rd2.fit(X,y)
print('gridsearchcv round 2 best ',svc_search_rd2.best_estimator_,svc_search_rd2.best_score_)

print('training+validation confusion matrix:\n',confusion_matrix(y, svc_search_rd2.best_estimator_.predict(X)))
print('testset confusion matrix:\n',confusion_matrix(y_test, svc_search_rd2.best_estimator_.predict(X_test)))


#set_2
#X_set_2 = DataMatrix_set2.iloc[:,2:-2].values
X_set_2 = DataMatrix_set2.loc[:,descriptor_columns].values

y_set_2 = DataMatrix_set2.iloc[:,-1].values
print('set2 confusion matrix:\n',confusion_matrix(y_set_2, svc_search_rd2.best_estimator_.predict(X_set_2)))


after_prediction_DataMatrix_set1=DataMatrix_set1.copy()
after_prediction_DataMatrix_set2=DataMatrix_set2.copy()

after_prediction_DataMatrix_set1['CatPredicted'] = None
after_prediction_DataMatrix_set1.iloc[:,-1]=svc_search_rd2.best_estimator_.predict(after_prediction_DataMatrix_set1[descriptor_columns])

after_prediction_DataMatrix_set2['CatPredicted'] = None
after_prediction_DataMatrix_set2.iloc[:,-1]=svc_search_rd2.best_estimator_.predict(X_set_2)

after_prediction_DataMatrix_set1.to_csv('Classification_set1.csv',index=False)
after_prediction_DataMatrix_set2.to_csv('Classification_set2.csv',index=False)
