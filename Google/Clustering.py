import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Daten laden
df = pd.read_csv("input/googleplaystore_bereinigt.csv")

# 2. Datum verarbeiten


# 3. Log-Transformation
df["Installs_log"] = np.log1p(df["Installs"])
df["Price_log"] = np.log1p(df["Price"])

# 4. Relevante Features
numerical_cols = ["Rating", "Size", "Installs_log", "Price_log", "Last Updated"]
df_numerical = df[numerical_cols].copy()

# 5. Extremwerte entfernen (mean ± 3*std)
for col in numerical_cols:
    mean = df_numerical[col].mean()
    std = df_numerical[col].std()
    df_numerical = df_numerical[(df_numerical[col] >= mean - 3 * std) & (df_numerical[col] <= mean + 3 * std)]

# 6. Index synchronisieren
df = df.loc[df_numerical.index]

# 7. Standardisierung
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numerical)

# 8. KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)

# 9. Cluster-Beschreibungen automatisch anhand Quantilen
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
quantiles = df[numerical_cols].quantile([0.33, 0.66])

print("\nCluster-Beschreibungen:")
for cid, row in cluster_summary.iterrows():
    beschreibung = []
    for col in numerical_cols:
        val = row[col]
        q33, q66 = quantiles.loc[0.33, col], quantiles.loc[0.66, col]
        if val >= q66:
            level = "hoch"
        elif val >= q33:
            level = "mittel"
        else:
            level = "niedrig"
        name_map = {
            "Rating": "Bewertung",
            "Size": "Größe",
            "Installs_log": "Nutzung",
            "Price_log": "Preis",
            "Last Updated": "Aktualität"
        }
        beschreibung.append(f"{level}e {name_map[col]}")
    print(f"Cluster {cid}: {', '.join(beschreibung)}")


# 10. PCA zur Visualisierung
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)
df["PCA1"] = pca_components[:, 0]
df["PCA2"] = pca_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=70)
plt.title("PCA-Visualisierung der Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Boxplots: Nutzung, Preis, Bewertung, Aktualität, Größe
plt.figure(figsize=(20, 4))
plot_cols = ["Installs_log", "Price_log", "Rating", "Last Updated", "Size"]
for i, col in enumerate(plot_cols):
    plt.subplot(1, len(plot_cols), i+1)
    sns.boxplot(data=df, x="Cluster", y=col, palette="tab10")
    plt.title(f"{col} nach Cluster")
plt.tight_layout()
plt.show()

# 12. KDE für Nutzung
plt.figure(figsize=(10, 5))
for cluster_id in df["Cluster"].unique():
    sns.kdeplot(df[df["Cluster"] == cluster_id]["Installs_log"], label=f"Cluster {cluster_id}", fill=True)
plt.title("Dichteverteilung der Nutzung (log)")
plt.xlabel("log(Installs)")
plt.legend()
plt.tight_layout()
plt.show()

# 13. Balkenplot – durchschnittliche log(Nutzung)
cluster_installs_log = df.groupby("Cluster")["Installs_log"].mean().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(
    x=cluster_installs_log.index.astype(str),
    height=cluster_installs_log.values,
    color=sns.color_palette("tab10")
)
plt.title("Durchschnittliche Nutzung (log) pro Cluster")
plt.ylabel("log(Installs)")
plt.xlabel("Cluster")
plt.tight_layout()
plt.show()

# 14. Beispielhafte Apps je Cluster
print("\nBeispielhafte Apps pro Cluster:")
for c in sorted(df["Cluster"].unique()):
    print(f"\nCluster {c}:")
    beispiele = df[df["Cluster"] == c].head(3)
    print(beispiele[["App", "Category", "Rating", "Size", "Installs", "Price", "Last Updated"]])
