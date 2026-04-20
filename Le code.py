import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Traitement_Eaux_GE.csv")

print(f"Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(f"Total valeurs manquantes : {df.isnull().sum().sum()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(df['DCO_entree'], df['taux_elimination'], alpha=0.6, color='steelblue')
axes[0].set_xlabel("DCO en entrée (mg/L)")
axes[0].set_ylabel("Taux d'élimination (%)")
axes[0].set_title("DCO_entrée vs Taux d'élimination")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df['debit_entree'], df['taux_elimination'], alpha=0.6, color='coral')
axes[1].set_xlabel("Débit journalier (m³/j)")
axes[1].set_ylabel("Taux d'élimination (%)")
axes[1].set_title("Débit journalier vs Taux d'élimination")
axes[1].grid(True, alpha=0.3)

procedes = ['Decantation', 'Lagunage', 'Filtre_biologique', 'Boues_activees']
data_box = [df[df['type_procede'] == p]['taux_elimination'].values for p in procedes]
bp = axes[2].boxplot(data_box, tick_labels=['Décantation', 'Lagunage', 'Filtre bio.', 'Boues act.'], patch_artist=True)
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[2].set_xlabel("Type de procédé")
axes[2].set_ylabel("Taux d'élimination (%)")
axes[2].set_title("Performances par type de procédé")
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("visualisation_donnees.png", dpi=150, bbox_inches='tight')
plt.show()

encodage = {'Decantation': 1, 'Lagunage': 2, 'Filtre_biologique': 3, 'Boues_activees': 4}
df['type_procede_encoded'] = df['type_procede'].map(encodage)

df_num = df.drop(columns=['type_procede'])
corr_matrix = df_num.corr()
print(corr_matrix['taux_elimination'].sort_values(ascending=False))

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title("Heatmap de la matrice de corrélation")
plt.tight_layout()
plt.savefig("heatmap_correlation.png", dpi=150, bbox_inches='tight')
plt.show()

corr_target = corr_matrix['taux_elimination'].drop('taux_elimination').abs().sort_values(ascending=False)
features_selectionnees = corr_target[corr_target >= 0.1].index.tolist()

X_all = df_num[features_selectionnees]
y = df['taux_elimination']

X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

X_train_m1 = X_train[['DCO_entree']]
X_test_m1  = X_test[['DCO_entree']]
model1 = LinearRegression()
model1.fit(X_train_m1, y_train)

model2 = LinearRegression()
model2.fit(X_train, y_train)

def evaluer_modele(modele, X_test, y_test, nom):
    y_pred = modele.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n{nom} — MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return y_pred, mse, mae, r2

y_pred1, mse1, mae1, r2_1 = evaluer_modele(model1, X_test_m1, y_test, "Modèle 1 - Simple")
y_pred2, mse2, mae2, r2_2 = evaluer_modele(model2, X_test,    y_test, "Modèle 2 - Multiple")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, y_pred, titre, r2 in zip(axes,
    [y_pred1, y_pred2],
    ["Modèle 1 - Simple", "Modèle 2 - Multiple"],
    [r2_1, r2_2]):
    ax.scatter(y_test, y_pred, alpha=0.6, color='royalblue')
    lim = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lim, lim, 'r--', linewidth=2, label="Prédiction parfaite")
    ax.set_xlabel("Valeurs réelles (%)")
    ax.set_ylabel("Valeurs prédites (%)")
    ax.set_title(f"{titre} | R² = {r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prediction_vs_reel.png", dpi=150, bbox_inches='tight')
plt.show()

new_sample = pd.DataFrame({
    'debit_entree': [1500], 'DCO_entree': [300], 'MES_entree': [200],
    'temperature_eau': [20], 'pH_entree': [7.5], 'type_procede_encoded': [4]
})
print(f"\nPrédiction Modèle 1 : {model1.predict(new_sample[['DCO_entree']])[0]:.2f} %")
print(f"Prédiction Modèle 2 : {model2.predict(new_sample[features_selectionnees])[0]:.2f} %")
