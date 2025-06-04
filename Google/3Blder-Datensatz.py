import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Schritt 1: Daten einlesen
# -------------------------------
df = pd.read_csv("input/googleplaystore.csv")

# -------------------------------
# Schritt 2: Beispieldaten festlegen
# -------------------------------
sample_indices = df[df['Rating'].isna() | df['Price'].str.contains('\$', na=False) | df['Installs'].str.contains('\+', na=False)].sample(5, random_state=42).index
df_sample_orig = df.loc[sample_indices].copy()

# -------------------------------
# Schritt 3: Tabelle 1 – Originaldaten
# -------------------------------
cols_to_show = ['App', 'Category', 'Rating', 'Reviews', 'Installs', 'Price']

fig1, ax1 = plt.subplots(figsize=(12, 3))
ax1.axis('off')
table1 = ax1.table(cellText=df_sample_orig[cols_to_show].astype(str).values,
                   colLabels=cols_to_show,
                   cellLoc='center',
                   loc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1.2, 1.5)
plt.title("Tabelle 1 – Originaldaten (vor Bereinigung)", fontsize=14)
plt.savefig("tabelle_1_originaldaten.png", bbox_inches='tight')
plt.close()

# -------------------------------
# Schritt 4: Imputation + Kodierung
# -------------------------------
df_clean = df.copy()

# Spalten mit zu vielen NaNs entfernen
missing_percent = df_clean.isnull().mean() * 100
to_drop = missing_percent[missing_percent > 20].index.tolist()
df_clean.drop(columns=to_drop, inplace=True)

# Imputation: numerisch -> mean, kategorisch -> mode
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype == 'object':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

# Bereinigung und Umwandlung
df_clean['Price'] = df_clean['Price'].astype(str).str.replace('$', '', regex=False)
df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce').fillna(0)

df_clean['Installs'] = df_clean['Installs'].astype(str).str.replace('[+,]', '', regex=True)
df_clean['Installs'] = pd.to_numeric(df_clean['Installs'], errors='coerce').fillna(0)

df_clean['Reviews'] = pd.to_numeric(df_clean['Reviews'], errors='coerce').fillna(0)
df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce').fillna(df_clean['Rating'].mean())

# Kodierung von Kategorien
categorical_cols = df_clean.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype('category').cat.codes

# Tabelle 2 – Nach Imputation & Kodierung (aber vor Standardisierung)
df_sample_imputed = df_clean.loc[sample_indices, cols_to_show]

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.axis('off')
table2 = ax2.table(cellText=df_sample_imputed.astype(str).values,
                   colLabels=cols_to_show,
                   cellLoc='center',
                   loc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1.2, 1.5)
plt.title("Tabelle 2 – Nach Imputation & Kodierung", fontsize=14)
plt.savefig("tabelle_2_imputiert_kodiert.png", bbox_inches='tight')
plt.close()

# -------------------------------
# Schritt 5: Standardisierung
# -------------------------------
numerical_cols = ['App', 'Category','Rating', 'Reviews', 'Installs', 'Price']
scaler = StandardScaler()
df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

df_sample_scaled = df_clean.loc[sample_indices, cols_to_show]

# Tabelle 3 – Nach Standardisierung
fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.axis('off')
table3 = ax3.table(cellText=df_sample_scaled.astype(str).values,
                   colLabels=cols_to_show,
                   cellLoc='center',
                   loc='center')
table3.auto_set_font_size(False)
table3.set_fontsize(10)
table3.scale(1.2, 1.5)
plt.title("Tabelle 3 – Final (Standardisiert)", fontsize=14)
plt.savefig("tabelle_3_standardisiert.png", bbox_inches='tight')
plt.close()

df_clean.to_csv("input/googleplaystore_bereinigt.csv", index=False)