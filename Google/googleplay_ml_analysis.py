
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Daten laden
df = pd.read_csv("input/googleplaystore.csv")
reviews = pd.read_csv("input/googleplaystore_user_reviews.csv")

# Vorverarbeitung
df = df[df["Installs"].notnull()]
df = df[~df["Installs"].str.contains("Free")]  # Entferne fehlerhafte Einträge
df["Installs"] = df["Installs"].str.replace(",", "").str.replace("+", "").astype(int)

# Zielvariable visualisieren
plt.figure(figsize=(10,6))
sns.histplot(df["Installs"], bins=50, kde=True)
plt.title("Verteilung der Installationen")
plt.xscale("log")
plt.xlabel("Installationen (log)")
plt.ylabel("Anzahl")
plt.savefig("01_distribution_installs.png")
plt.close()

# Reviews als numerisch
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

# Größe bereinigen
df["Size"] = df["Size"].replace("Varies with device", np.nan)
df["Size"] = df["Size"].str.replace("M", "e6").str.replace("k", "e3")
df["Size"] = df["Size"].str.replace(" ", "")
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

# Preis bereinigen
df["Price"] = df["Price"].str.replace("$", "")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Imputation fehlender Werte
num_cols = ["Rating", "Reviews", "Size", "Price"]
imp = SimpleImputer(strategy="median")
df[num_cols] = imp.fit_transform(df[num_cols])

# Kodierung kategorischer Daten
cat_cols = ["Category", "Type", "Content Rating", "Genres"]
df[cat_cols] = df[cat_cols].fillna("Unknown")
df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

# Features zusammenführen
features = pd.concat([df[num_cols], df_encoded], axis=1)
target = df["Installs"]

# Skalierung
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Heatmap der Korrelationsmatrix
plt.figure(figsize=(12,10))
corr_matrix = pd.DataFrame(features_scaled, columns=features.columns).corr()
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Korrelationsmatrix der Features")
plt.savefig("02_correlation_matrix.png")
plt.close()

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Entscheidungsbaum
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Baum visualisieren
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=features.columns, filled=True, fontsize=8)
plt.savefig("03_decision_tree.png")
plt.close()

# Feature Importance
importances = tree.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10,6))
plt.title("Top 10 wichtigste Features")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features.columns[i] for i in indices])
plt.savefig("04_feature_importance.png")
plt.close()

# Fehlerkennzahlen
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Validierungsmetriken:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# KMeans-Clustering (z. B. auf skalierte Features)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

plt.figure(figsize=(10,6))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.title("KMeans-Cluster auf Apps")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("05_kmeans_clusters.png")
plt.close()
