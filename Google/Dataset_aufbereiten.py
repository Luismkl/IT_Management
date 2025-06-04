import pandas as pd
import numpy as np
from tabulate import tabulate
import plotly.graph_objs as go
import plotly.offline as py

# Lade CSV-Datei
df = pd.read_csv("input/googleplaystore.csv")

# --- SCHRITT 1: Repräsentative Vorschau vor der Bereinigung ---
sample_df = df.sample(n=5, random_state=42)
sample_df.to_csv("preview_original.csv", index=False)

# --- SCHRITT 2: Datenbereinigung ---
df_cleaned = df.copy()

# Entferne offensichtliche Duplikate
df_cleaned.drop_duplicates(inplace=True)

# Bereinige 'Installs'-Spalte (entferne '+' und ',')
if 'Installs' in df_cleaned.columns:
    df_cleaned['Installs'] = df_cleaned['Installs'].str.replace('+', '', regex=False)
    df_cleaned['Installs'] = df_cleaned['Installs'].str.replace(',', '', regex=False)
    df_cleaned['Installs'] = pd.to_numeric(df_cleaned['Installs'], errors='coerce')

# Bereinige 'Price'-Spalte (entferne '$')
if 'Price' in df_cleaned.columns:
    df_cleaned['Price'] = df_cleaned['Price'].str.replace('$', '', regex=False)
    df_cleaned['Price'] = pd.to_numeric(df_cleaned['Price'], errors='coerce')

# Korrigiere 'Varies with device' mit Modus bei bestimmten Spalten
for col in ['Current Ver', 'Android Ver', 'Last Updated']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].replace('Varies with device', np.nan)
        if df_cleaned[col].dropna().size > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

# Ersetze NaNs in numerischen Spalten durch Median, in Strings durch Modus
for col in df_cleaned.columns:
    if df_cleaned[col].dtype in ['float64', 'int64']:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    else:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

# Korrigiere fehlerhafte Werte (z.B. 'Size' Spalte mit 'Varies with device')
if 'Size' in df_cleaned.columns:
    df_cleaned['Size'] = df_cleaned['Size'].replace('Varies with device', np.nan)
    df_cleaned['Size'] = df_cleaned['Size'].str.replace('M', '').str.replace('k', '')
    df_cleaned['Size'] = pd.to_numeric(df_cleaned['Size'], errors='coerce')
    df_cleaned['Size'] = df_cleaned['Size'].fillna(df_cleaned['Size'].median())

# --- SCHRITT 3: KEINE Feature-Encoding mehr (Strings bleiben erhalten) ---
# Überspringt bewusste Kodierung

# Gleiche Zeilen aus cleaned DataFrame extrahieren
sample_cleaned_df = df_cleaned.loc[sample_df.index]
sample_cleaned_df.to_csv("preview_cleaned.csv", index=False)

# --- SCHRITT 4: Visualisierung als HTML-Datei ---
def render_table(df):
    return go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in df.columns],
            fill_color='firebrick',
            font=dict(color='white', family='Arial', size=14),
            align='left',
            line_color='black'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='white',
            font=dict(color='black', family='Arial', size=12),
            align='left',
            line_color='firebrick'
        )
    )])

fig_orig = render_table(sample_df)
fig_clean = render_table(sample_cleaned_df)

# Speichere HTML-Datei mit beiden Tabellen
with open("vergleich_vor_nachher.html", "w") as f:
    f.write(py.plot(fig_orig, include_plotlyjs='cdn', output_type='div'))
    f.write("<br><br>")
    f.write(py.plot(fig_clean, include_plotlyjs=False, output_type='div'))

print("\nHTML-Datei 'vergleich_vor_nachher.html' wurde erstellt.")
print("Bereinigter Datensatz gespeichert als 'googleplaystore_cleaned.csv'.")

# Bereinigten gesamten Datensatz speichern
df_cleaned.to_csv("googleplaystore_cleaned.csv", index=False)