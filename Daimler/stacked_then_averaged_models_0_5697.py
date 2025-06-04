import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import scipy.stats as stats

# === 1. Daten laden und vorbereiten ===
train = pd.read_csv('input/train.csv')

# Entferne ID-Spalte
if 'ID' in train.columns:
    train.drop(columns=['ID'], inplace=True)

# Zielvariable und Features
X = train.drop(columns=['y'])
y = train['y']

# Spalten automatisch in numerisch und kategorisch trennen
cat_features = X.select_dtypes(include=['object']).columns.tolist()
X[cat_features] = X[cat_features].astype(str)  # falls z. B. int+str gemischt sind

# Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. CatBoost Modell ===
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    loss_function='Quantile:alpha=0.5',
    eval_metric='MAE',
    cat_features=cat_features,
    verbose=100
)
model.fit(X_train, y_train)

# Vorhersage
y_pred = model.predict(X_val)

# === 3. Residuen-Analyse ===
residuals = y_val - y_pred
sns.histplot(residuals, bins=30, kde=True)
plt.title('Residuenverteilung')
plt.xlabel('y - ŷ')
plt.grid(True)
plt.show()

# Q-Q-Plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q-Plot der Residuen")
plt.grid(True)
plt.show()

# === 4. Outlier-Detektion ===
iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
threshold = 1.5 * iqr
outliers = residuals[np.abs(residuals) > threshold]
print(f"Anzahl Outlier (|resid| > 1.5*IQR): {len(outliers)}")

# === 5. Feature-Bedeutung: Mutual Information ===
X_mi = pd.get_dummies(X)  # MI braucht numerisch
mi = mutual_info_regression(X_mi, y)
mi_series = pd.Series(mi, index=X_mi.columns).sort_values(ascending=True)
mi_series.tail(20).plot(kind='barh', figsize=(8, 6))
plt.title('Feature-Bedeutung (Mutual Information)')
plt.xlabel('MI Score')
plt.grid(True)
plt.show()

# === 6. PCA für Visualisierung ===
X_numeric = X.select_dtypes(include=[np.number])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_numeric)

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.colorbar(sc, label='y')
plt.xlabel('PCA Komponente 1')
plt.ylabel('PCA Komponente 2')
plt.title('PCA Projektion (farblich: y)')
plt.grid(True)
plt.show()

# === 7. Modellbewertung ===
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'RMSE auf Validierungsdaten: {rmse:.4f}')
