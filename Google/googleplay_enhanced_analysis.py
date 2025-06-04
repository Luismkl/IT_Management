import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Daten laden
df = pd.read_csv("input/googleplaystore.csv")
reviews = pd.read_csv("input/googleplaystore_user_reviews.csv")

# Zielvariable vorbereiten
df = df[df["Installs"].notnull()]
df = df[~df["Installs"].str.contains("Free")]
df["Installs"] = df["Installs"].str.replace(",", "").str.replace("+", "").astype(int)

# Datenbereinigung
df["Size"] = df["Size"].replace("Varies with device", np.nan)
df["Size"] = df["Size"].str.replace("M", "e6").str.replace("k", "e3").str.replace(" ", "")
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
df["Price"] = df["Price"].str.replace("$", "")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Nur numerische Features ohne "Reviews"
num_cols = ["Rating", "Size", "Price"]
imp = SimpleImputer(strategy="median")
df[num_cols] = imp.fit_transform(df[num_cols])

# Kategorische Features
cat_cols = ["Category", "Type", "Content Rating", "Genres"]
df[cat_cols] = df[cat_cols].fillna("Unknown")
df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

# Feature-Matrix und Ziel
features = pd.concat([df[num_cols], df_encoded], axis=1)
target = df["Installs"]

# Skalierung
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df["Cluster"] = clusters

# Cluster-Visualisierung
plt.figure(figsize=(10,6))
plt.scatter(features_scaled[:,0], features_scaled[:,1], c=clusters, cmap="Set1", alpha=0.6)
plt.title("App Cluster basierend auf Features (ohne Reviews)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("03_kmeans_clusters.png")
plt.close()

# Clusterbeschreibung
df["Cluster"] = clusters
df.groupby("Cluster")[["Size", "Price", "Rating", "Installs"]].mean().to_csv("cluster_summary.csv")

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Einfacher Entscheidungsbaum
simple_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
simple_tree.fit(X_train, y_train)

plt.figure(figsize=(16,8))
plot_tree(simple_tree, feature_names=features.columns, filled=True, fontsize=8)
plt.title("Einfacher Entscheidungsbaum ohne 'Reviews'")
plt.savefig("04_simple_tree_no_reviews.png")
plt.close()

# Komplexer Baum
complex_tree = DecisionTreeRegressor(max_depth=15, min_samples_split=5, random_state=42)
complex_tree.fit(X_train, y_train)
y_pred_complex = complex_tree.predict(X_test)

# Feature Importance
importances = complex_tree.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10,6))
plt.title("Top 10 wichtigste Features (ohne 'Reviews')")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features.columns[i] for i in indices])
plt.savefig("06_feature_importance_no_reviews.png")
plt.close()

# Evaluation-Funktion
def evaluate_model(y_true, y_pred, title):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(6,4))
    sns.barplot(x=["RMSE", "MAE", "R2"], y=[rmse, mae, r2])
    plt.title(f"Validierung: {title}")
    plt.savefig(f"07_validation_{title}.png")
    plt.close()
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

metrics_simple = evaluate_model(y_test, simple_tree.predict(X_test), "Einfacher_Baum")
metrics_complex = evaluate_model(y_test, y_pred_complex, "Komplexer_Baum")

# Cross-Validation
scoring = {'R2': 'r2', 'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error'}
cv_results = cross_validate(complex_tree, features_scaled, target, cv=5, scoring=scoring)

cv_metrics = {
    'R2': np.mean(cv_results['test_R2']),
    'MAE': -np.mean(cv_results['test_MAE']),
    'RMSE': -np.mean(cv_results['test_RMSE']),
}

plt.figure(figsize=(6,4))
sns.barplot(x=list(cv_metrics.keys()), y=list(cv_metrics.values()))
plt.title("Cross-Validation (ohne 'Reviews')")
plt.savefig("08_cross_validation_no_reviews.png")
plt.close()

print("Einfacher Baum:", metrics_simple)
print("Komplexer Baum:", metrics_complex)
print("Cross-Validation:", cv_metrics)
