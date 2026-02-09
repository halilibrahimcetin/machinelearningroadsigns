from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pca import pca
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold



import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##normalize ve standartize fonksiyonları
normalize_scaler = MinMaxScaler()
standartize_scaler = StandardScaler()

#dataset import etme
dataset = pd.read_csv("C:/Users/halilibrahim/Desktop/Ders/Machine Learning/Proje/datasetrevize.csv")

#label değişkenini ayırdık 
X = dataset.drop(columns=['label'])
Y = dataset['label']


kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#smote = SMOTE(sampling_strategy=0.25,random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X, Y)

#adasyn = ADASYN(sampling_strategy=0.75,random_state=42)
#X_resampled, y_resampled = adasyn.fit_resample(X, Y)

tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X, Y)

# Veri setini train-test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#X için test ve train standartizasyon ve normalizasyonu
X_train_normalize = normalize_scaler.fit_transform(X_train)
X_test_normalize = normalize_scaler.transform(X_test)

X_train_standartize = standartize_scaler.fit_transform(X_train)
X_test_standartize = standartize_scaler.transform(X_test)

#print(X_test)
#<print("----------------")
#print(X_test_normalize)




param_grid_ann = {'hidden_layer_sizes': [(10, 30, 2), (50, 50, 50),(100,)],'activation': ['logistic','relu'],'alpha': [0.0001, 0.001, 0.01,1],'learning_rate_init': [0.001, 0.01, 0.1,1],'max_iter': [200, 500, 1000,2000]}
ann_model_no_scale = RandomizedSearchCV(estimator=MLPClassifier(random_state=42),param_distributions=param_grid_ann,cv=kf)
ann_model_no_scale.fit(X_train, y_train)



y_pred = ann_model_no_scale.predict(X_test)
print("No scaling Accuracy:", accuracy_score(y_test, y_pred))


## Normalize
ann_model_normalize =RandomizedSearchCV(estimator=MLPClassifier(random_state=42),param_distributions=param_grid_ann,cv=kf)
ann_model_normalize.fit(X_train_normalize, y_train)
y_pred_normalize = ann_model_normalize.predict(X_test_normalize)
print("Normalize Accuracy:", accuracy_score(y_test, y_pred_normalize))
#print(classification_report(y_test, y_pred_normalize))

## Standartize
ann_model_standartize = RandomizedSearchCV(estimator=MLPClassifier(random_state=42),param_distributions=param_grid_ann,cv=kf)
ann_model_standartize.fit(X_train_standartize, y_train)
y_pred_standartize= ann_model_standartize.predict(X_test_standartize)
print("Standartize Accuracy:", accuracy_score(y_test, y_pred_standartize))
#print(classification_report(y_test, y_pred_standartize))

#cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.title("Confusion Matrix")
#plt.xlabel("Predicted")
#plt.ylabel("True")
#plt.show()

#print(Y.value_counts())

#-------------------------------------------------------------
#----------            ANN                --------------------
#-------------------------------------------------------------


##############################################################
##########             NO SCALE           ####################
##############################################################

#PCA 3 no scale
pca3 = PCA(n_components=3)
X_train_PCA3 = pca3.fit_transform(X_train)
X_test_PCA3 = pca3.transform(X_test)
ann_model_no_scale.fit(X_train_PCA3,y_train)
y_pca3_predict = ann_model_no_scale.predict(X_test_PCA3)

#PCA 4 no scale
pca4 = PCA(n_components=4)
X_train_PCA4 = pca4.fit_transform(X_train_standartize)
X_test_PCA4 = pca4.transform(X_test_standartize)
ann_model_no_scale.fit(X_train_PCA4,y_train)
y_pca4_predict = ann_model_no_scale.predict(X_test_PCA4)


##############################################################
##########             NORMALİZE          ####################
##############################################################

#PCA 3 normalize
pca3 = PCA(n_components=3)
X_train_PCA3_normalize = pca3.fit_transform(X_train_normalize)
X_test_PCA3_normalize = pca3.transform(X_test_normalize)
ann_model_normalize.fit(X_train_PCA3_normalize,y_train)
y_pca3_predict_normalize = ann_model_normalize.predict(X_test_PCA3_normalize)

#PCA 4 normalize
pca4 = PCA(n_components=4)
X_train_PCA4_normalize = pca4.fit_transform(X_train_normalize)
X_test_PCA4_normalize = pca4.transform(X_test_normalize)
ann_model_normalize.fit(X_train_PCA4_normalize,y_train)
y_pca4_predict_normalize = ann_model_normalize.predict(X_test_PCA4_normalize)


##############################################################
##########             STANDARTİZE        ####################
##############################################################

#PCA 3 standartize
pca3 = PCA(n_components=3)
X_train_PCA3_standartize = pca3.fit_transform(X_train_standartize)
X_test_PCA3_standartize = pca3.transform(X_test_standartize)
ann_model_standartize.fit(X_train_PCA3_standartize,y_train)
y_pca3_predict_standartize = ann_model_standartize.predict(X_test_PCA3_standartize)

#PCA 4 standartize
pca4 = PCA(n_components=4)
X_train_PCA4_standartize = pca4.fit_transform(X_train_standartize)
X_test_PCA4_standartize = pca4.transform(X_test_standartize)
ann_model_standartize.fit(X_train_PCA4_standartize,y_train)
y_pca4_predict_standartize = ann_model_standartize.predict(X_test_PCA4_standartize)



##SONUÇ PRİNTLERİ

'''print("PCA3 Kullanımı Sonrası Accuracy: (NO SCALE)", accuracy_score(y_test, y_pca3_predict))
print(classification_report(y_test, y_pca3_predict))
print("PCA4 Kullanımı Sonrası Accuracy: (NO SCALE)", accuracy_score(y_test,y_pca4_predict))
print(classification_report(y_test, y_pca4_predict))

print("PCA3 Kullanımı Sonrası Accuracy: (Normalize)", accuracy_score(y_test, y_pca3_predict_normalize))
print(classification_report(y_test, y_pca3_predict_normalize))
print("PCA4 Kullanımı Sonrası Accuracy: (Normalize)", accuracy_score(y_test,y_pca4_predict_normalize))
print(classification_report(y_test, y_pca4_predict_normalize))

print("PCA3 Kullanımı Sonrası Accuracy: (Standartize)"), accuracy_score(y_test, y_pca3_predict_standartize)
print(classification_report(y_test, y_pca3_predict_standartize))
print("PCA4 Kullanımı Sonrası Accuracy: (Standartize)", accuracy_score(y_test,y_pca4_predict_standartize))
print(classification_report(y_test, y_pca4_predict_standartize))
'''



analiz_sonuclari = []
analiz_sonuclari.append(["No Scale", "PCA 0", accuracy_score(y_test, y_pred)])
analiz_sonuclari.append(["No Scale", "PCA 3", accuracy_score(y_test, y_pca3_predict)])
analiz_sonuclari.append(["No Scale", "PCA 4", accuracy_score(y_test, y_pca4_predict)])

analiz_sonuclari.append(["Normalize", "PCA 0", accuracy_score(y_test, y_pred_normalize)])
analiz_sonuclari.append(["Normalize", "PCA 3", accuracy_score(y_test, y_pca3_predict_normalize)])
analiz_sonuclari.append(["Normalize", "PCA 4", accuracy_score(y_test, y_pca4_predict_normalize)])

analiz_sonuclari.append(["Standartize", "PCA 0", accuracy_score(y_test, y_pred_standartize)])
analiz_sonuclari.append(["Standartize", "PCA 3", accuracy_score(y_test, y_pca3_predict_standartize)])
analiz_sonuclari.append(["Standartize", "PCA 4", accuracy_score(y_test, y_pca4_predict_standartize)])


analiz_sonuclari_f1score = []
analiz_sonuclari_f1score.append(["No Scale", "PCA 0",f1_score(y_test,y_pred)])
analiz_sonuclari_f1score.append(["No Scale", "PCA 3",f1_score(y_test,y_pca3_predict)])
analiz_sonuclari_f1score.append(["No Scale", "PCA 4",f1_score(y_test,y_pca4_predict)])

analiz_sonuclari_f1score.append(["Normalize", "PCA 0",f1_score(y_test,y_pred_normalize)])
analiz_sonuclari_f1score.append(["Normalize", "PCA 3",f1_score(y_test,y_pca3_predict_normalize)])
analiz_sonuclari_f1score.append(["Normalize", "PCA 4",f1_score(y_test,y_pca4_predict_normalize)])

analiz_sonuclari_f1score.append(["Standartize", "PCA 0",f1_score(y_test,y_pred_standartize)])
analiz_sonuclari_f1score.append(["Standartize", "PCA 3",f1_score(y_test,y_pca3_predict_standartize)])
analiz_sonuclari_f1score.append(["Standartize", "PCA 4",f1_score(y_test,y_pca4_predict_standartize)])

analiz_sonuclari_precision = []
analiz_sonuclari_precision.append(["No Scale", "PCA 0",precision_score(y_test,y_pred)])
analiz_sonuclari_precision.append(["No Scale", "PCA 3",precision_score(y_test,y_pca3_predict)])
analiz_sonuclari_precision.append(["No Scale", "PCA 4",precision_score(y_test,y_pca4_predict)])

analiz_sonuclari_precision.append(["Normalize", "PCA 0",precision_score(y_test,y_pred_normalize)])
analiz_sonuclari_precision.append(["Normalize", "PCA 3",precision_score(y_test,y_pca3_predict_normalize)])
analiz_sonuclari_precision.append(["Normalize", "PCA 4",precision_score(y_test,y_pca4_predict_normalize)])

analiz_sonuclari_precision.append(["Standartize", "PCA 0",precision_score(y_test,y_pred_normalize)])
analiz_sonuclari_precision.append(["Standartize", "PCA 3",precision_score(y_test,y_pca3_predict_standartize)])
analiz_sonuclari_precision.append(["Standartize", "PCA 4",precision_score(y_test,y_pca4_predict_standartize)])

analiz_sonuclari_recall = []
analiz_sonuclari_recall.append(["No Scale", "PCA 0",recall_score(y_test,y_pred)])
analiz_sonuclari_recall.append(["No Scale", "PCA 3",recall_score(y_test,y_pca3_predict)])
analiz_sonuclari_recall.append(["No Scale", "PCA 4",recall_score(y_test,y_pca4_predict)])

analiz_sonuclari_recall.append(["Normalize", "PCA 0",recall_score(y_test,y_pred_normalize)])
analiz_sonuclari_recall.append(["Normalize", "PCA 3",recall_score(y_test,y_pca3_predict_normalize)])
analiz_sonuclari_recall.append(["Normalize", "PCA 4",recall_score(y_test,y_pca4_predict_normalize)])

analiz_sonuclari_recall.append(["Standartize", "PCA 0",recall_score(y_test,y_pred_standartize)])
analiz_sonuclari_recall.append(["Standartize", "PCA 3",recall_score(y_test,y_pca3_predict_standartize)])
analiz_sonuclari_recall.append(["Standartize", "PCA 4",recall_score(y_test,y_pca4_predict_standartize)])


sonuc_dataframe = pd.DataFrame(analiz_sonuclari,columns=["Scaling", "PCA Components", "Accuracy"])
f1sonuc_dataframe = pd.DataFrame(analiz_sonuclari_f1score,columns=["Scaling", "PCA Components", "F1-Score"])
precision_dataframe = pd.DataFrame(analiz_sonuclari_precision,columns=["Scaling", "PCA Components", "Precision"])
recall_dataframe = pd.DataFrame(analiz_sonuclari_recall,columns=["Scaling", "PCA Components", "Recall"])

print("\n\n Accuracy Sonuçları:")
print(sonuc_dataframe)
print("\n\n F1-Score Sonuçları:")
print(f1sonuc_dataframe)
print("\n\n Precision Sonuçları:")
print(precision_dataframe)
print("\n\n Recall Sonuçları:")
print(recall_dataframe)


'''
fig, ax = plt.subplots(figsize=(10, 6))
scaling_methods = sonuc_dataframe["Scaling"].unique()

for method in scaling_methods:
    method_data = sonuc_dataframe[sonuc_dataframe["Scaling"] == method]
    ax.bar(
        method_data["PCA Components"] + " (" + method + ")",
        method_data["Accuracy"],
        label=method
    )

 #Grafik ayarları
plt.title("ANN Model Performans Karşılaştırması")
plt.ylabel("Accuracy")
plt.xlabel("PCA ve Ölçeklendirme Yöntemi")
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Accuracy aralığı [0, 1]
plt.legend(title="Scaling Method")
plt.tight_layout()
plt.show()
'''






#-------------------------------------------------------------
#----------            SVM                --------------------
#-------------------------------------------------------------

svm_results = []

##############################################################
##########             NO SCALE           ####################
##############################################################
param_grid_svm = {'kernel': ['rbf'],'C': [0.1,1,10],'gamma': [0.001,0.01,0.1,1],'max_iter': [500, 1000, 2000]}


svm_no_scale = RandomizedSearchCV(estimator=SVC(random_state=42) , param_distributions=param_grid_svm,cv=kf)
svm_no_scale.fit(X_train_PCA3, y_train)
y_svm_pca3_predict = svm_no_scale.predict(X_test_PCA3)
svm_results.append(["No Scale", "PCA 3", accuracy_score(y_test, y_svm_pca3_predict)])



# PCA 3 No Scale
pca3 = PCA(n_components=3)
X_train_PCA3 = pca3.fit_transform(X_train)
X_test_PCA3 = pca3.transform(X_test)

svm_no_scale.fit(X_train_PCA3, y_train)
y_svm_pca3_predict = svm_no_scale.predict(X_test_PCA3)
svm_results.append(["No Scale", "PCA 3", accuracy_score(y_test, y_svm_pca3_predict)])

# PCA 4 No Scale
pca4 = PCA(n_components=4)
X_train_PCA4 = pca4.fit_transform(X_train)
X_test_PCA4 = pca4.transform(X_test)

svm_no_scale.fit(X_train_PCA4, y_train)
y_svm_pca4_predict = svm_no_scale.predict(X_test_PCA4)
svm_results.append(["No Scale", "PCA 4", accuracy_score(y_test, y_svm_pca4_predict)])


##############################################################
##########             NORMALİZE          ####################
##############################################################


# PCA 3 Normalize
pca3 = PCA(n_components=3)
X_train_PCA3_normalize = pca3.fit_transform(X_train_normalize)
X_test_PCA3_normalize = pca3.transform(X_test_normalize)

svm_normalize = svm_no_scale
svm_normalize.fit(X_train_PCA3_normalize, y_train)
y_svm_pca3_predict_normalize = svm_normalize.predict(X_test_PCA3_normalize)
svm_results.append(["Normalize", "PCA 3", accuracy_score(y_test, y_svm_pca3_predict_normalize)])

# PCA 4 Normalize
pca4 = PCA(n_components=4)
X_train_PCA4_normalize = pca4.fit_transform(X_train_normalize)
X_test_PCA4_normalize = pca4.transform(X_test_normalize)

svm_normalize.fit(X_train_PCA4_normalize, y_train)
y_svm_pca4_predict_normalize = svm_normalize.predict(X_test_PCA4_normalize)
svm_results.append(["Normalize", "PCA 4", accuracy_score(y_test, y_svm_pca4_predict_normalize)])


##############################################################
##########             STANDARTİZE        ####################
##############################################################


# PCA 3 Standartize
pca3 = PCA(n_components=3)
X_train_PCA3_standartize = pca3.fit_transform(X_train_standartize)
X_test_PCA3_standartize = pca3.transform(X_test_standartize)

svm_standartize = svm_no_scale
svm_standartize.fit(X_train_PCA3_standartize, y_train)
y_svm_pca3_predict_standartize = svm_standartize.predict(X_test_PCA3_standartize)
svm_results.append(["Standartize", "PCA 3", accuracy_score(y_test, y_svm_pca3_predict_standartize)])

# PCA 4 Standartize
pca4 = PCA(n_components=4)
X_train_PCA4_standartize = pca4.fit_transform(X_train_standartize)
X_test_PCA4_standartize = pca4.transform(X_test_standartize)

svm_standartize.fit(X_train_PCA4_standartize, y_train)
y_svm_pca4_predict_standartize = svm_standartize.predict(X_test_PCA4_standartize)
svm_results.append(["Standartize", "PCA 4", accuracy_score(y_test, y_svm_pca4_predict_standartize)])

svm_sonuclari = []
svm_sonuclari.append(["No Scale", "PCA 0", accuracy_score(y_test, y_pred)])
svm_sonuclari.append(["No Scale", "PCA 3", accuracy_score(y_test, y_svm_pca3_predict)])
svm_sonuclari.append(["No Scale", "PCA 4", accuracy_score(y_test, y_svm_pca4_predict)])

svm_sonuclari.append(["Normalize", "PCA 0", accuracy_score(y_test, y_pred_normalize)])
svm_sonuclari.append(["Normalize", "PCA 3", accuracy_score(y_test, y_svm_pca3_predict_normalize)])
svm_sonuclari.append(["Normalize", "PCA 4", accuracy_score(y_test, y_svm_pca4_predict_normalize)])

svm_sonuclari.append(["Standartize", "PCA 0", accuracy_score(y_test, y_pred_standartize)])
svm_sonuclari.append(["Standartize", "PCA 3", accuracy_score(y_test, y_svm_pca3_predict_standartize)])
svm_sonuclari.append(["Standartize", "PCA 4", accuracy_score(y_test, y_svm_pca4_predict_standartize)])

svm_sonuclari_f1score = []
svm_sonuclari_f1score.append(["No Scale", "PCA 0", f1_score(y_test, y_pred)])
svm_sonuclari_f1score.append(["No Scale", "PCA 3", f1_score(y_test, y_svm_pca3_predict)])
svm_sonuclari_f1score.append(["No Scale", "PCA 4", f1_score(y_test, y_svm_pca4_predict)])

svm_sonuclari_f1score.append(["Normalize", "PCA 0", f1_score(y_test, y_pred)])
svm_sonuclari_f1score.append(["Normalize", "PCA 3", f1_score(y_test, y_svm_pca3_predict_normalize)])
svm_sonuclari_f1score.append(["Normalize", "PCA 4", f1_score(y_test, y_svm_pca4_predict_normalize)])

svm_sonuclari_f1score.append(["Standartize", "PCA 0", f1_score(y_test, y_pred)])
svm_sonuclari_f1score.append(["Standartize", "PCA 3", f1_score(y_test, y_svm_pca3_predict_standartize)])
svm_sonuclari_f1score.append(["Standartize", "PCA 4", f1_score(y_test, y_svm_pca4_predict_standartize)])

svm_sonuclari_recall = []
svm_sonuclari_recall.append(["No Scale", "PCA 0", recall_score(y_test, y_pred)])
svm_sonuclari_recall.append(["No Scale", "PCA 3", recall_score(y_test, y_svm_pca3_predict)])
svm_sonuclari_recall.append(["No Scale", "PCA 4", recall_score(y_test, y_svm_pca4_predict)])

svm_sonuclari_recall.append(["Normalize", "PCA 0", recall_score(y_test, y_pred)])
svm_sonuclari_recall.append(["Normalize", "PCA 3", recall_score(y_test, y_svm_pca3_predict_normalize)])
svm_sonuclari_recall.append(["Normalize", "PCA 4", recall_score(y_test, y_svm_pca4_predict_normalize)])

svm_sonuclari_recall.append(["Standartize", "PCA 0", recall_score(y_test, y_pred)])
svm_sonuclari_recall.append(["Standartize", "PCA 3", recall_score(y_test, y_svm_pca3_predict_standartize)])
svm_sonuclari_recall.append(["Standartize", "PCA 4", recall_score(y_test, y_svm_pca4_predict_standartize)])

svm_sonuclari_precision = []
svm_sonuclari_precision.append(["No Scale", "PCA 0", precision_score(y_test, y_pred)])
svm_sonuclari_precision.append(["No Scale", "PCA 3", precision_score(y_test, y_svm_pca3_predict)])
svm_sonuclari_precision.append(["No Scale", "PCA 4", precision_score(y_test, y_svm_pca4_predict)])

svm_sonuclari_precision.append(["Normalize", "PCA 0", precision_score(y_test, y_pred)])
svm_sonuclari_precision.append(["Normalize", "PCA 3", precision_score(y_test, y_svm_pca3_predict_normalize)])
svm_sonuclari_precision.append(["Normalize", "PCA 4", precision_score(y_test, y_svm_pca4_predict_normalize)])

svm_sonuclari_precision.append(["Standartize", "PCA 0", precision_score(y_test, y_pred)])
svm_sonuclari_precision.append(["Standartize", "PCA 3", precision_score(y_test, y_svm_pca3_predict_standartize)])
svm_sonuclari_precision.append(["Standartize", "PCA 4", precision_score(y_test, y_svm_pca4_predict_standartize)])




svm_accuracy_dataframe = pd.DataFrame(svm_sonuclari,columns=["Scaling", "PCA Components", "Accuracy"])
svm_f1score_dataframe = pd.DataFrame(svm_sonuclari_f1score,columns=["Scaling", "PCA Components", "F1-Score"])
svm_recall_dataframe = pd.DataFrame(svm_sonuclari_recall,columns=["Scaling", "PCA Components", "Recall"])
svm_precision_dataframe = pd.DataFrame(svm_sonuclari_precision,columns=["Scaling", "PCA Components", "Precision"])


print("\n\n ----------SVM------------")

print("\n\n Accuracy Sonuçları:")
print(svm_accuracy_dataframe)
print("\n\n F1-Score Sonuçları:")
print(svm_f1score_dataframe)
print("\n\n Recall Sonuçları:")
print(svm_recall_dataframe)
print("\n\n Precision Sonuçları:")
print(svm_precision_dataframe)








#-------------------------------------------------------------
#----------            RFC                --------------------
#-------------------------------------------------------------

rfc_results = []

rf_grid={'n_estimators' : [100,200,500] , 'max_depth': [5,10,25]}
rf_model = RandomizedSearchCV(RandomForestClassifier(random_state=42),param_distributions=rf_grid,cv=kf)
rf_model.fit(X_train, y_train)
y_rf_predict = rf_model.predict(X_test)


##############################################################
##########             NO SCALE           ####################
######################### #####################################


# PCA 3 No Scale
pca3 = PCA(n_components=3)
X_train_PCA3 = pca3.fit_transform(X_train)
X_test_PCA3 = pca3.transform(X_test)

rf_model.fit(X_train_PCA3, y_train)
y_rfc_pca3_predict = rf_model.predict(X_test_PCA3)
rfc_results.append(["No Scale", "PCA 3", accuracy_score(y_test, y_rfc_pca3_predict)])

# PCA 4 No Scale
pca4 = PCA(n_components=4)
X_train_PCA4 = pca4.fit_transform(X_train)
X_test_PCA4 = pca4.transform(X_test)

rf_model.fit(X_train_PCA4, y_train)
y_rfc_pca4_predict = rf_model.predict(X_test_PCA4)
rfc_results.append(["No Scale", "PCA 3", accuracy_score(y_test, y_rfc_pca4_predict)])



##############################################################
##########             NORMALİZE          ####################
##############################################################

# PCA 3 Normalize
pca3 = PCA(n_components=3)
X_train_PCA3_normalize = pca3.fit_transform(X_train_normalize)
X_test_PCA3_normalize = pca3.transform(X_test_normalize)

rf_model.fit(X_train_PCA3_normalize, y_train)
y_rfc_pca3_predict_normalize = rf_model.predict(X_test_PCA3_normalize)
rfc_results.append(["Normalize", "PCA 3", accuracy_score(y_test, y_rfc_pca3_predict_normalize)])

# PCA 4 Normalize
pca4 = PCA(n_components=4)
X_train_PCA4_normalize = pca4.fit_transform(X_train_normalize)
X_test_PCA4_normalize = pca4.transform(X_test_normalize)

rf_model.fit(X_train_PCA4_normalize, y_train)
y_rfc_pca4_predict_normalize = rf_model.predict(X_test_PCA4_normalize)
rfc_results.append(["Normalize", "PCA 3", accuracy_score(y_test, y_rfc_pca4_predict_normalize)])


##############################################################
##########             STANDARTİZE        ####################
##############################################################


# PCA 3 Normalize
pca3 = PCA(n_components=3)
X_train_PCA3_standartize = pca3.fit_transform(X_train_standartize)
X_test_PCA3_standartize = pca3.transform(X_test_standartize)

rf_model.fit(X_train_PCA3_standartize, y_train)
y_rfc_pca3_predict_standartize = rf_model.predict(X_test_PCA3_standartize)
rfc_results.append(["Standartize", "PCA 3", accuracy_score(y_test, y_rfc_pca3_predict_standartize)])

# PCA 4 Normalize
pca4 = PCA(n_components=4)
X_train_PCA4_standartize = pca4.fit_transform(X_train_standartize)
X_test_PCA4_standartize = pca4.transform(X_test_standartize)

rf_model.fit(X_train_PCA4_standartize, y_train)
y_rfc_pca4_predict_standartize = rf_model.predict(X_test_PCA4_standartize)
rfc_results.append(["Normalize", "PCA 3", accuracy_score(y_test, y_rfc_pca4_predict_standartize)])


rfc_sonuclari = []
rfc_sonuclari.append(["No Scale", "PCA 0", accuracy_score(y_test, y_rf_predict)])
rfc_sonuclari.append(["No Scale", "PCA 3", accuracy_score(y_test, y_rfc_pca3_predict)])
rfc_sonuclari.append(["No Scale", "PCA 4", accuracy_score(y_test, y_rfc_pca4_predict)])

rfc_sonuclari.append(["Normalize", "PCA 0", accuracy_score(y_test, y_rf_predict)])
rfc_sonuclari.append(["Normalize", "PCA 3", accuracy_score(y_test, y_rfc_pca3_predict_normalize)])
rfc_sonuclari.append(["Normalize", "PCA 4", accuracy_score(y_test, y_rfc_pca4_predict_normalize)])

rfc_sonuclari.append(["Standartize", "PCA 0", accuracy_score(y_test, y_rf_predict)])
rfc_sonuclari.append(["Standartize", "PCA 3", accuracy_score(y_test, y_rfc_pca3_predict_standartize)])
rfc_sonuclari.append(["Standartize", "PCA 4", accuracy_score(y_test, y_rfc_pca4_predict_standartize)])



rfc_sonuclari_f1 = []
rfc_sonuclari_f1.append(["No Scale", "PCA 0", f1_score(y_test, y_rf_predict)])
rfc_sonuclari_f1.append(["No Scale", "PCA 3", f1_score(y_test, y_rfc_pca3_predict)])
rfc_sonuclari_f1.append(["No Scale", "PCA 4", f1_score(y_test, y_rfc_pca4_predict)])

rfc_sonuclari_f1.append(["Normalize", "PCA 0", f1_score(y_test, y_rf_predict)])
rfc_sonuclari_f1.append(["Normalize", "PCA 3", f1_score(y_test, y_rfc_pca3_predict_normalize)])
rfc_sonuclari_f1.append(["Normalize", "PCA 4", f1_score(y_test, y_rfc_pca4_predict_normalize)])

rfc_sonuclari_f1.append(["Standartize", "PCA 0", f1_score(y_test, y_rf_predict)])
rfc_sonuclari_f1.append(["Standartize", "PCA 3", f1_score(y_test, y_rfc_pca3_predict_standartize)])
rfc_sonuclari_f1.append(["Standartize", "PCA 4", f1_score(y_test, y_rfc_pca4_predict_standartize)])


rfc_sonuclari_recall = []
rfc_sonuclari_recall.append(["No Scale", "PCA 0", recall_score(y_test, y_rf_predict)])
rfc_sonuclari_recall.append(["No Scale", "PCA 3", recall_score(y_test, y_rfc_pca3_predict)])
rfc_sonuclari_recall.append(["No Scale", "PCA 4", recall_score(y_test, y_rfc_pca4_predict)])

rfc_sonuclari_recall.append(["Normalize", "PCA 0", recall_score(y_test, y_rf_predict)])
rfc_sonuclari_recall.append(["Normalize", "PCA 3", recall_score(y_test, y_rfc_pca3_predict_normalize)])
rfc_sonuclari_recall.append(["Normalize", "PCA 4", recall_score(y_test, y_rfc_pca4_predict_normalize)])

rfc_sonuclari_recall.append(["Standartize", "PCA 0", recall_score(y_test, y_rf_predict)])
rfc_sonuclari_recall.append(["Standartize", "PCA 3", recall_score(y_test, y_rfc_pca3_predict_standartize)])
rfc_sonuclari_recall.append(["Standartize", "PCA 4", recall_score(y_test, y_rfc_pca4_predict_standartize)])

'''
rfc_sonuclari_precision = []
rfc_sonuclari_precision.append(["No Scale", "PCA 0", precision_score(y_test, y_rf_predict)])
rfc_sonuclari_precision.append(["No Scale", "PCA 3", precision_score(y_test, y_rfc_pca3_predict)])
rfc_sonuclari_precision.append(["No Scale", "PCA 4", precision_score(y_test, y_rfc_pca4_predict)])

rfc_sonuclari_precision.append(["Normalize", "PCA 0", precision_score(y_test, y_rf_predict)])
rfc_sonuclari_precision.append(["Normalize", "PCA 3", precision_score(y_test, y_rfc_pca3_predict_normalize)])
rfc_sonuclari_precision.append(["Normalize", "PCA 4", precision_score(y_test, y_rfc_pca4_predict_normalize)])

rfc_sonuclari_precision.append(["Standartize", "PCA 0", precision_score(y_test, y_rf_predict)])
rfc_sonuclari_precision.append(["Standartize", "PCA 3", precision_score(y_test, y_rfc_pca3_predict_standartize)])
rfc_sonuclari_precision.append(["Standartize", "PCA 4", precision_score(y_test, y_rfc_pca4_predict_standartize)])'''


rfc_accuracy_dataframe = pd.DataFrame(rfc_sonuclari,columns=["Scaling", "PCA Components", "Accuracy"])
rfc_f1score_dataframe = pd.DataFrame(rfc_sonuclari_f1,columns=["Scaling", "PCA Components", "F1-Score"])
rfc_recall_dataframe = pd.DataFrame(rfc_sonuclari_recall,columns=["Scaling", "PCA Components", "Recall"])
'''rfc_precision_dataframe = pd.DataFrame(rfc_sonuclari_precision,columns=["Scaling", "PCA Components", "Precision"])'''


print("\n\n ----------RFC------------")

print("\n\n Accuracy Sonuçları:")
print(rfc_accuracy_dataframe)
print("\n\n F1-Score Sonuçları:")
print(rfc_f1score_dataframe)
print("\n\n Recall Sonuçları:")
print(rfc_recall_dataframe)
'''print("\n\n Precision Sonuçları:")
print(rfc_precision_dataframe)'''

#######  -CM--- #############

#X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
'''
def plot_confusion_matrix(y_true, y_pred, model_name, scaling, pca_comp):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {model_name} ({scaling}, PCA {pca_comp})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ANN
plot_confusion_matrix(y_test, y_pred, "ANN", "No Scale", "0")
plot_confusion_matrix(y_test, y_pca3_predict, "ANN", "No Scale", "3")
plot_confusion_matrix(y_test, y_pca4_predict, "ANN", "No Scale", "4")

plot_confusion_matrix(y_test, y_pred_normalize, "ANN", "Normalize", "0")
plot_confusion_matrix(y_test, y_pca3_predict_normalize, "ANN", "Normalize", "3")
plot_confusion_matrix(y_test, y_pca4_predict_normalize, "ANN", "Normalize", "4")

plot_confusion_matrix(y_test, y_pred_standartize, "ANN", "Standartize", "0")
plot_confusion_matrix(y_test, y_pca3_predict_standartize, "ANN", "Standartize", "3")
plot_confusion_matrix(y_test, y_pca4_predict_standartize, "ANN", "Standartize", "4")

# SVM 
plot_confusion_matrix(y_test, y_svm_pca3_predict, "SVM", "No Scale", "3")
plot_confusion_matrix(y_test, y_svm_pca4_predict, "SVM", "No Scale", "4")
plot_confusion_matrix(y_test, y_svm_pca3_predict_normalize, "SVM", "Normalize", "3")
plot_confusion_matrix(y_test, y_svm_pca4_predict_normalize, "SVM", "Normalize", "4")
plot_confusion_matrix(y_test, y_svm_pca3_predict_standartize, "SVM", "Standartize", "3")
plot_confusion_matrix(y_test, y_svm_pca4_predict_standartize, "SVM", "Standartize", "4")

# RFC 
plot_confusion_matrix(y_test, y_rf_predict, "RFC", "No Scale", "0")
plot_confusion_matrix(y_test, y_rfc_pca3_predict, "RFC", "No Scale", "3")
plot_confusion_matrix(y_test, y_rfc_pca4_predict, "RFC", "No Scale", "4")
plot_confusion_matrix(y_test, y_rfc_pca3_predict_normalize, "RFC", "Normalize", "3")
plot_confusion_matrix(y_test, y_rfc_pca4_predict_normalize, "RFC", "Normalize", "4")
plot_confusion_matrix(y_test, y_rfc_pca3_predict_standartize, "RFC", "Standartize", "3")
plot_confusion_matrix(y_test, y_rfc_pca4_predict_standartize, "RFC", "Standartize", "4")

'''

korelasyon_matris = dataset.corr()
plt.figure(figsize=(15,15))
sns.heatmap(korelasyon_matris, annot=True, cmap="viridis", center=0, cbar_kws={'shrink': 0.8})
plt.title("Dataset Korelasyon Analizi")
plt.show()


'''

##########    K-FOLD     ##############



### ---- ANN ---- ###

ann_accuracy_scores = []
ann_precision_scores = []
ann_recall_scores = []
ann_f1_scores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    X_train_normalize = normalize_scaler.fit_transform(X_train)
    X_test_normalize = normalize_scaler.transform(X_test)

    ann_model = MLPClassifier(hidden_layer_sizes=(10, 30, 2),learning_rate="constant",alpha=0.05, activation='logistic', max_iter=2000, random_state=42)
    ann_model.fit(X_train_normalize, y_train)

    y_pred = ann_model.predict(X_test_normalize)

    ann_accuracy_scores.append(accuracy_score(y_test, y_pred))
    ann_precision_scores.append(precision_score(y_test, y_pred))
    ann_recall_scores.append(recall_score(y_test, y_pred))
    ann_f1_scores.append(f1_score(y_test, y_pred))

print("\n\n----K-FOLD----")
print("\n\nANN K-Fold Sonuçları:")
print(f"Accuracy: {np.mean(ann_accuracy_scores):.4f}")
print(f"Precision: {np.mean(ann_precision_scores):.4f}")
print(f"Recall: {np.mean(ann_recall_scores):.4f}")
print(f"F1-Score: {np.mean(ann_f1_scores):.4f}")



### ---- SVM ---- ###

svm_accuracy_scores = []
svm_precision_scores = []
svm_recall_scores = []
svm_f1_scores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    X_train_standartize = standartize_scaler.fit_transform(X_train)
    X_test_standartize = standartize_scaler.transform(X_test)

    svm_model = SVC(kernel='rbf', C=2, gamma='auto', random_state=42)
    svm_model.fit(X_train_standartize, y_train)
    y_pred = svm_model.predict(X_test_standartize)

    svm_accuracy_scores.append(accuracy_score(y_test, y_pred))
    svm_precision_scores.append(precision_score(y_test, y_pred))
    svm_recall_scores.append(recall_score(y_test, y_pred))
    svm_f1_scores.append(f1_score(y_test, y_pred))

print("\nSVM K-Fold Sonuçları:")
print(f"Accuracy: {np.mean(svm_accuracy_scores):.4f}")
print(f"Precision: {np.mean(svm_precision_scores):.4f}")
print(f"Recall: {np.mean(svm_recall_scores):.4f}")
print(f"F1-Score: {np.mean(svm_f1_scores):.4f}")



### ---- RFC ---- ###


rf_accuracy_scores = []
rf_precision_scores = []
rf_recall_scores = []
rf_f1_scores = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    pca3 = PCA(n_components=3)
    X_train_pca = pca3.fit_transform(X_train)
    X_test_pca = pca3.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf_model.fit(X_train_pca, y_train)

    y_pred = rf_model.predict(X_test_pca)

    rf_accuracy_scores.append(accuracy_score(y_test, y_pred))
    rf_precision_scores.append(precision_score(y_test, y_pred))
    rf_recall_scores.append(recall_score(y_test, y_pred))
    rf_f1_scores.append(f1_score(y_test, y_pred))

print("\nRandom Forest K-Fold Sonuçları:")
print(f"Accuracy: {np.mean(rf_accuracy_scores):.4f}")
print(f"Precision: {np.mean(rf_precision_scores):.4f}")
print(f"Recall: {np.mean(rf_recall_scores):.4f}")
print(f"F1-Score: {np.mean(rf_f1_scores):.4f}")




'''

##############################################################
##########          SONUÇLAR GRAFİK       ####################
##############################################################


methods = [
    
    "No Scale - PCA 3", "No Scale - PCA 4",
    "Normalize - PCA 3", "Normalize - PCA 4",
    "Standardize - PCA 3", "Standardize - PCA 4"
]

# ANN, SVM ve RFC için doğruluk değerleri


ann_values = [
    accuracy_score(y_test, y_pca3_predict), accuracy_score(y_test, y_pca4_predict),
    accuracy_score(y_test, y_pca3_predict_normalize), accuracy_score(y_test, y_pca4_predict_normalize),
    accuracy_score(y_test, y_pca3_predict_standartize), accuracy_score(y_test, y_pca4_predict_standartize)
]

svm_values = [
    accuracy_score(y_test, y_svm_pca3_predict), accuracy_score(y_test, y_svm_pca4_predict),
    accuracy_score(y_test, y_svm_pca3_predict_normalize), accuracy_score(y_test, y_svm_pca4_predict_normalize),
    accuracy_score(y_test, y_svm_pca3_predict_standartize), accuracy_score(y_test, y_svm_pca4_predict_standartize)
]

rfc_values = [
    accuracy_score(y_test, y_rfc_pca3_predict), accuracy_score(y_test, y_rfc_pca4_predict),
    accuracy_score(y_test, y_rfc_pca3_predict_normalize), accuracy_score(y_test, y_rfc_pca4_predict_normalize),
    accuracy_score(y_test, y_rfc_pca3_predict_standartize), accuracy_score(y_test, y_rfc_pca4_predict_standartize)
]


x = np.arange(len(methods))  # X ekseni için sütun konumları
bar_width = 0.25  # Sütun genişliği

fig, ax = plt.subplots(figsize=(14, 8))

# ANN
bars1 = ax.bar(x - bar_width, ann_values, bar_width, label='ANN', color='royalblue')

# SVM
bars2 = ax.bar(x, svm_values, bar_width, label='SVM', color='darkorange')

# RFC
bars3 = ax.bar(x + bar_width, rfc_values, bar_width, label='Random Forest', color='seagreen')

ax.set_title("ANN, SVM ve RFC Accuracy Değerleri", fontsize=16)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.legend(title="Models")

# Y ekseni sınırları
plt.ylim(0, 1)

# Görsel düzenleme
plt.tight_layout()
plt.show()
