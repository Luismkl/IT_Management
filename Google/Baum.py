# --------------------------------------------------------
# 📦 Komplettanalyse für Google Play Store App-Daten
# Autor: [Dein Name] – basierend auf IT-Management Skript
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

# 📁 Ordner für Ausgabe
os.makedirs("output", exist_ok=True)
sns.set(style="whitegrid")

# -----------------------------
# 🔹 Schritt 1: Daten einlesen
# -----------------------------
df = pd.read_csv("input/googleplaystore.csv")
reviews = pd.read_csv("input/googleplaystore_user_reviews.csv")

# 🔸 Zielvariable vorbereiten
df = df[df["Installs"].notnull()]
df = df[~df["Installs"].str.contains("Free")]
df["Installs"] = df["Installs"].str.replace(",", "").str.replace("+", "").astype(int)

# -----------------------------
# 🔹 Schritt 2: Bereinigung
# -----------------------------
df["Size"] = df["Size"].replace("Varies with device", np.nan)
df["Size"] = df["Size"].str.replace("M", "e6").str.replace("k", "e3").str.replace(" ", "")
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

df["Price"] = df["Price"].str.replace("$", "")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Visualisierung: Verteilung vor Imputation
plt.figure(figsize=(10, 4))
sns.histplot(df["Size"], bins=40, color="skyblue", kde=True)
plt.title("Größe vor Imputation")
plt.savefig("output/size_vor_imputation.png")
plt.close()

# -----------------------------
# 🔹 Schritt 3: Imputation & Kodierung
# -----------------------------
num_cols = ["Rating", "Size", "Price"]
imp = SimpleImputer(strategy="median")
df[num_cols] = imp.fit_transform(df[num_cols])

# Kategorisch kodieren
cat_cols = ["Category", "Type", "Content Rating", "Genres"]
df[cat_cols] = df[cat_cols].fillna("Unknown")
df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

# Visualisierung: Größe nach Imputation
plt.figure(figsize=(10, 4))
sns.histplot(df["Size"], bins=40, color="orange", kde=True)
plt.title("Größe nach Imputation")
plt.savefig("output/size_nach_imputation.png")
plt.close()

# -----------------------------
# 🔹 Schritt 4: Skalierung
# -----------------------------
features = pd.concat([df[num_cols], df_encoded], axis=1)
target = df["Installs"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Vergleich vor/nach Skalierung
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
sns.histplot(features["Size"], bins=30)
plt.title("Originalgröße")
plt.subplot(1,2,2)
sns.histplot(features_scaled[:,1], bins=30)
plt.title("Skaliert (Size)")
plt.tight_layout()
plt.savefig("output/skaliert_size_vergleich.png")
plt.close()

# -----------------------------
# 🔹 Schritt 5: Clustering
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df["Cluster"] = clusters

plt.figure(figsize=(8,6))
sns.scatterplot(x=features_scaled[:,1], y=features_scaled[:,2], hue=clusters, palette="Set2")
plt.xlabel("Size (skaliert)")
plt.ylabel("Price (skaliert)")
plt.title("Cluster nach Größe & Preis")
plt.savefig("output/cluster_groesse_preis.png")
plt.close()

df.groupby("Cluster")[["Size", "Price", "Rating", "Installs"]].mean().to_csv("output/cluster_summary.csv")

# -----------------------------
# 🔹 Schritt 6: Entscheidungsbäume
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Einfacher Baum für Präsentation
simple_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
simple_tree.fit(X_train, y_train)

plt.figure(figsize=(16,8))
plot_tree(simple_tree, feature_names=features.columns, filled=True, rounded=True, fontsize=8)
plt.title("Einfacher Entscheidungsbaum (max_depth=3)")
plt.savefig("output/baum_einfach.png")
plt.close()

# Komplexer Baum
complex_tree = DecisionTreeRegressor(max_depth=15, min_samples_split=5, random_state=42)
complex_tree.fit(X_train, y_train)
y_pred_complex = complex_tree.predict(X_test)

# -----------------------------
# 🔹 Schritt 7: Feature Importance
# -----------------------------
importances = complex_tree.feature_importances_
top10_idx = np.argsort(importances)[-10:]
plt.figure(figsize=(10,6))
plt.barh(range(10), importances[top10_idx])
plt.yticks(range(10), [features.columns[i] for i in top10_idx])
plt.title("Top 10 wichtigste Features")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.close()

# -----------------------------
# 🔹 Schritt 8: Validierung
# -----------------------------
def evaluate_model(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.barplot(x=["RMSE", "MAE", "R²"], y=[rmse, mae, r2])
    plt.title(f"Validierung: {name}")
    plt.savefig(f"output/validation_{name}.png")
    plt.close()
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

metrics_simple = evaluate_model(y_test, simple_tree.predict(X_test), "einfach")
metrics_complex = evaluate_model(y_test, y_pred_complex, "komplex")

# -----------------------------
# 🔹 Schritt 9: Cross-Validation
# -----------------------------
from sklearn.model_selection import cross_validate

cv_scores = cross_validate(complex_tree, features_scaled, target, cv=5,
    scoring={"R2": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"})

cv_metrics = {
    "R2": np.mean(cv_scores["test_R2"]),
    "MAE": -np.mean(cv_scores["test_MAE"]),
    "RMSE": -np.mean(cv_scores["test_RMSE"]),
}

# -----------------------------
# 🔹 Schritt 10: Modellvergleich
# -----------------------------
labels = ["Einfacher Baum", "Komplexer Baum", "Cross-Validation"]
rmse_vals = [metrics_simple["RMSE"], metrics_complex["RMSE"], cv_metrics["RMSE"]]
mae_vals = [metrics_simple["MAE"], metrics_complex["MAE"], cv_metrics["MAE"]]
r2_vals  = [metrics_simple["R2"],  metrics_complex["R2"],  cv_metrics["R2"]]

x = np.arange(len(labels))
bar_width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x, rmse_vals, width=bar_width, label="RMSE")
plt.bar(x + bar_width, mae_vals, width=bar_width, label="MAE")
plt.bar(x + 2*bar_width, r2_vals, width=bar_width, label="R²")
plt.xticks(x + bar_width, labels)
plt.title("Vergleich der Modelle")
plt.legend()
plt.tight_layout()
plt.savefig("output/modellvergleich.png")
plt.close()

print("✅ Analyse abgeschlossen. Diagramme & CSV liegen im Ordner 'output/'.")

# -----------------------------
# 🔁 NEU: Cross-Validation für beide Modelle
# -----------------------------
def crossval_scores(model, X, y, name):
    scores = cross_validate(model, X, y, cv=5,
        scoring={"R2": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"})
    result = {
        "R2": np.mean(scores["test_R2"]),
        "MAE": -np.mean(scores["test_MAE"]),
        "RMSE": -np.mean(scores["test_RMSE"]),
    }
    print(f"Cross-Validation für {name}: {result}")
    return result

cv_simple = crossval_scores(simple_tree, features_scaled, target, "Einfacher Baum")
cv_complex = crossval_scores(complex_tree, features_scaled, target, "Komplexer Baum")

# -----------------------------
# 📊 Vergleich der Cross-Validation beider Modelle
# -----------------------------
labels = ["Einfach (CV)", "Komplex (CV)"]
x = np.arange(len(labels))
bar_width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x, [cv_simple["RMSE"], cv_complex["RMSE"]], width=bar_width, label="RMSE")
plt.bar(x + bar_width, [cv_simple["MAE"], cv_complex["MAE"]], width=bar_width, label="MAE")
plt.bar(x + 2 * bar_width, [cv_simple["R2"], cv_complex["R2"]], width=bar_width, label="R²")
plt.xticks(x + bar_width, labels)
plt.title("📉 Cross-Validation Vergleich: Einfach vs. Komplex")
plt.legend()
plt.tight_layout()
plt.savefig("output/crossval_vergleich.png")
plt.close()

# -----------------------------
# 🔍 Cluster-Analyse nach Vorlesung (3 große Gruppen)
# -----------------------------
# Cluster neu mit 3 Gruppen
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(features_scaled)

# Visuelle Klassifikation
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Size", y="Price", hue="Cluster", style="Type", palette="Set1")
plt.title("🔍 Cluster: Preis vs. Größe + App-Typ")
plt.xlabel("Größe (MB)")
plt.ylabel("Preis ($)")
plt.legend(title="Cluster & Type")
plt.tight_layout()
plt.savefig("output/cluster_typ_groesse_preis.png")
plt.close()

# -----------------------------
# 📝 Beschreibung der Cluster (qualitativ)
# -----------------------------
print("\n🧠 Cluster-Interpretation:")
for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]
    print(f"\nCluster {c}:")
    print(f"Anzahl Apps: {len(sub)}")
    print(f"∅ Größe: {sub['Size'].mean():.2f} MB")
    print(f"∅ Preis: {sub['Price'].mean():.2f} $")
    print(f"∅ Rating: {sub['Rating'].mean():.2f}")
    print(f"Kostenlos-Anteil: {(sub['Type'] == 'Free').mean():.2%}")
    if sub['Price'].mean() > 2 and sub['Size'].mean() > 40:
        print("➡️ Beschreibung: Teure, große Apps – evtl. Premium-Spiele")
    elif sub['Price'].mean() < 0.5 and sub['Size'].mean() > 30:
        print("➡️ Beschreibung: Große Freeware – evtl. Produktivitäts-Apps")
    else:
        print("➡️ Beschreibung: Kleine Tools oder günstige Apps")
