import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Veri Setini Yükleme
dataset = pd.read_csv("C:/Users/halilibrahim/Desktop/Ders/Machine Learning/Proje/dataset.csv")

# Veri Analizi ve Eksik Verilerin Temizlenmesi
print("Veri seti ön bilgisi:")
print(dataset.info())
print("Eksik değer sayıları:")
print(dataset.isnull().sum())

# Eksik değerleri temizleme (gerekirse):
dataset.dropna(inplace=True)

# Label ve Özelliklerin Ayrılması
X = dataset.drop(columns=['label'])
Y = dataset['label']

# Eğitim ve Test Verilerinin Ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizasyon ve Standardizasyon
normalize_scaler = MinMaxScaler()
standartize_scaler = StandardScaler()

X_train_normalize = normalize_scaler.fit_transform(X_train)
X_test_normalize = normalize_scaler.transform(X_test)

X_train_standartize = standartize_scaler.fit_transform(X_train)
X_test_standartize = standartize_scaler.transform(X_test)

# PCA ile Özellik Azaltma
pca = PCA()
pca.fit(X_train_standartize)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Grafik: PCA Açıklanan Varyans
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('PCA Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# PCA ile 3 ve 4 bileşen için veri setleri oluşturma
pca_3 = PCA(n_components=3)
X_train_pca3 = pca_3.fit_transform(X_train_standartize)
X_test_pca3 = pca_3.transform(X_test_standartize)

pca_4 = PCA(n_components=4)
X_train_pca4 = pca_4.fit_transform(X_train_standartize)
X_test_pca4 = pca_4.transform(X_test_standartize)

# Model Tanımları ve Hiperparametre Optimizasyonu
models = {
    "ANN": MLPClassifier(max_iter=1000, random_state=42),
    "SVM": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

params = {
    "ANN": {
        'hidden_layer_sizes': [(10, 30, 2), (30, 15), (50,)],
        'activation': ['logistic', 'relu'],
        'learning_rate_init': [0.01, 0.001]
    },
    "SVM": {
        'kernel': ['rbf'],
        'C': [1, 2, 5],
        'gamma': ['scale', 'auto']
    },
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
}

results = {}
for model_name, model in models.items():
    print(f"\n--- {model_name} için Grid Search ---")
    grid = GridSearchCV(model, params[model_name], cv=3, scoring='accuracy')
    grid.fit(X_train_standartize, y_train)
    best_model = grid.best_estimator_
    print(f"En İyi Parametreler: {grid.best_params_}")

    # Test verisinde değerlendirme
    y_pred = best_model.predict(X_test_standartize)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[model_name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }

# Sonuçları Görselleştirme
results_df = pd.DataFrame(results).T
print("\nSonuçlar:")
print(results_df)

results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performans Karşılaştırması')
plt.ylabel('Değer')
plt.xlabel('Model')
plt.legend(title='Metrikler')
plt.grid()
plt.tight_layout()
plt.show()
