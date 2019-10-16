from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report as class_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y)



gnb=GaussianNB(var_smoothing=0.5)
gnb.fit(X_train,Y_train)
preds=gnb.predict(X_test)
accgnb = accuracy_score(preds, Y_test)
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
#plt.title('Random Forest classifier Young vs.Old (ROC curve)')
plt.legend(loc="lower right")
plt.show()

report_with_auc = class_report(
    y_true=Y_test, 
    y_pred=preds)

print("Report", report_with_auc)

print("GNB accuracy", accgnb)

svm=SVC(kernel='rbf',class_weight='balanced',C=1.0,verbose=True,max_iter=-1)
svm.fit(X_train,Y_train)
preds=svm.predict(X_test)
accsvm = accuracy_score(preds, Y_test)
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
#plt.title('Random Forest classifier Young vs.Old (ROC curve)')
plt.legend(loc="lower right")
plt.show()

report_with_auc = class_report(
    y_true=Y_test, 
    y_pred=preds)

print("Report", report_with_auc)

print("SVM accuracy", accsvm)


KNN=knn(n_neighbors=5,p=2,)
KNN.fit(X_train,Y_train)
preds=KNN.predict(X_test)
accknn = accuracy_score(preds, Y_test)
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
#plt.title('Random Forest classifier Young vs.Old (ROC curve)')
plt.legend(loc="lower right")
plt.show()

report_with_auc = class_report(
    y_true=Y_test, 
    y_pred=preds)

print("Report", report_with_auc)

print("KNN accuracy", accknn)


mlp=MLP(hidden_layer_sizes=(100),activation='logistic',solver='adam',random_state=42,
       verbose=True, max_iter=200)
mlp.fit(X_train,Y_train)
preds=mlp.predict(X_test)
accmlp = accuracy_score(preds, Y_test)
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
#plt.title('Random Forest classifier Young vs.Old (ROC curve)')
plt.legend(loc="lower right")
plt.show()

report_with_auc = class_report(
    y_true=Y_test, 
    y_pred=preds)

print("Report", report_with_auc)

print("MLP accuracy", accmlp)
