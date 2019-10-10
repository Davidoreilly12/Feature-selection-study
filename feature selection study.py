import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report as class_report
from feature_selection_ga import FeatureSelectionGA
from skrebate import ReliefF as RFF
#Import required files as pandas dataframe
X=pd.read_excel(r"C:\fileaddress.xlsx", sheet_name='Features',index_cols=0)
Y=pd.read_excel(r"C:\targetvariableaddress.xlsx", sheet_name='Y',index_cols=0)



#ReliefF feature selection (15 rounds with 20% of the sample randomly selected in each)
df=pd.DataFrame()
i=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for x in i:
    X_train_rel, X_test_rel, Y_train_rel, Y_test_rel = train_test_split(X,
                                                                        Y, 
                                                                        test_size = 0.8,
                                                                         stratify=Y) 
    scaler_rel = MinMaxScaler(feature_range=(0,1))
    scaler_rel.fit(X_train_rel)
    X_imputed_train_rel = scaler_rel.transform(X_train_rel)
    X_train_rel = pd.DataFrame(X_imputed_train_rel, 
                               columns = X_train_rel.columns)
    relief1=RFF(n_neighbors=len(X_train_rel)-1,verbose=True,discrete_threshold=5)
    relief1.fit(X_train_rel.values,Y_train_rel.values)
    feats=relief1.feature_importances_
    feats=pd.DataFrame([feats],columns=X_train_rel.columns)
    df=df.append([feats],ignore_index=True)

df=df.T
avg= df.set_index(np.arange(len(df)) //len(df)).mean(level=0)
avg=avg.T
avg.columns=['Score']
avg=avg.sort_values(by='Score',ascending=False)
avg=avg.T
avg.columns=X.columns[avg.columns]
avg=avg.T.reset_index()
avg=avg.drop([0])
print(avg)
X_reliefF=X[avg['index']]

#SVM-RFE AND Genetic algorithm
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(X)
X_imputed_train = scaler.transform(X)
#X_imputed_test = scaler.transform(X)
X = pd.DataFrame(X_imputed_train, columns = X.columns)

cv=StratifiedKFold(10)
svm=SVC(kernel='linear',C=1.0,class_weight='balanced',verbose=True,probability=True,
        random_state=42)
viz1=RFECV(svm,cv=cv,scoring='f1',verbose=True)
viz1.fit(X,Y)
X=X[X.columns[viz1.support_]]
print("Optimal number of features : %d" % viz1.n_features_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(viz1.grid_scores_) + 1), viz1.grid_scores_)
plt.show()

model=SVC(kernel='linear',C=1.0,class_weight='balanced')
fsga = FeatureSelectionGA(model,X.values,Y.values,verbose=1,cv_split=10)
pop = fsga.generate(n_pop=870,mutxpb=0.01,cxpb=0.4,ngen=10)
bestfeatsind=np.asarray(fsga.best_ind,dtype=bool)
X_GA=X[X.columns[bestfeatsind]]

X_final=pd.concat(X_GA,X_reliefF,axis=0,ignore_index=True)


#Final MLP classification model 
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size = 0.5,stratify=Y,)


mlp=MLP(hidden_layer_sizes=(100),activation='logistic',solver='adam',random_state=42,
       verbose=True,validation_fraction=0.1, max_iter=200)
mlp.fit(X_train,Y_train)
preds=mlp.predict(X_test)
acc = accuracy_score(preds, Y_test)
for i in Y_test:
    fpr, tpr,_= roc_curve(Y_test, preds)
    roc_auc = auc(fpr, tpr)



fig=plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

report_with_auc = class_report(
    y_true=Y_test, 
    y_pred=preds)

print("Report", report_with_auc)

print("MLP accuracy", acc)
