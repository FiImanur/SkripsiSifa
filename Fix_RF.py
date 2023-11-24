# Import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

tic=time.time()

# Baca data dari file CSV
data = pd.read_csv('D:\sifa\skripsi\ES3.csv')

# Pisahkan fitur (X) dan label (y)
X = data.drop('Label', axis=1)  # Ubah 'label' menjadi nama kolom label sesuai dengan data Anda
y = data['Label']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih model pada data latih
rf_model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = rf_model.predict(X_test)

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 

# tabel akurasi
print (classification_report(y_test, y_pred))

# feature selection supervised -> filter method -> feature importances using random forest
importance = rf_model.feature_importances_
columns = X.columns
i = 0

while i < len(columns):
    print(f" The importance of feature '{columns[i]}' is {round(importance[i] * 100, 2)}%. ")
    i += 1

# heatmap
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()

# save
joblib.dump(rf_model, "random_forest3.joblib")

toc=time.time()
print("\nComputation time is {} second".format((toc-tic)))
