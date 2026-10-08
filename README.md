# Fraud Detection - Déploiement (VS Code)

Projet PFE — Hamza Chaabene — TEK-UP University × Capgemini

## 📁 Structure

```
fraud-detection/
├── models/               ← produit par la CELL 18 du notebook (voir ci-dessous)
│   ├── catboost_model.cbm       ← CatBoost (modèle seul + base du stacking)
│   ├── stack_lgb_model.pkl      ← base LightGBM du stacking
│   ├── stack_xgb_model.pkl      ← base XGBoost du stacking
│   ├── stack_meta_model.pkl     ← méta-modèle (régression logistique)
│   ├── stacking_config.json     ← fichiers + ordre des entrées du méta-modèle
│   ├── preprocessing.pkl        ← médianes, encodeurs (complétion des features)
│   ├── feature_names.json       ← 512 features attendues par l'API
│   ├── thresholds.json          ← seuils + "production_model"
│   ├── dashboard_metrics.json   ← chiffres affichés par le dashboard (CELL 18)
│   ├── example_transactions.json← vraies transactions de test (1 par décision, CELL 18)
│   ├── lgb_model.pkl / xgb_model.pkl / autoencoder.keras / scaler.pkl
│   │                            ← modèles de comparaison (non utilisés par l'API)
│   └── feature_names_lgb_xgb.json
├── api/
│   └── main.py           ← FastAPI (backend de prédiction)
├── dashboard/
│   └── app.py             ← Streamlit (interface)
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Étape 1 : Copier les modèles

Copiez le contenu du dossier `models/` produit par la CELL 18 du
notebook dans le dossier `models/` à la racine de ce projet.

### Modèle de production

Le modèle utilisé est lu dans `thresholds.json` (clé
`production_model`), et peut être forcé avec la variable
d'environnement `FRAUD_MODEL` :

| Valeur | Modèle | Résultat test (alerte ≤ 5 %) |
|---|---|---|
| `stacking` | LightGBM + XGBoost + CatBoost → régression logistique | AUC 0.913 — bénéfice net $1,227,320 |
| `catboost` | CatBoost seul | AUC 0.897 — bénéfice net $1,158,410 |

Les seuils de décision (`<modèle>_profit_optimized` → revue manuelle,
`<modèle>` (F1) → blocage) sont choisis par le notebook sur la
validation, pour le modèle concerné.

```powershell
$env:FRAUD_MODEL="catboost"   # optionnel : forcer CatBoost seul
```

## 🔐 Étape 2 : Définir la clé API (sécurité)

Les endpoints `/predict`, `/predict/batch` et `/model/info` sont
protégés par une clé API. Définissez-la avant de lancer le serveur
(remplacez par une valeur secrète de votre choix) :

```powershell
# Windows PowerShell
$env:FRAUD_API_KEY="votre-cle-secrete"
```
```bash
# Linux / Mac
export FRAUD_API_KEY="votre-cle-secrete"
```

⚠️ Si cette variable n'est pas définie, l'API démarre quand même
(pour faciliter le développement) mais affiche un avertissement et
utilise une clé par défaut non sécurisée (`dev-only-change-me`) — à
ne jamais utiliser hors développement local.

Vous devrez saisir la même clé dans la barre latérale du dashboard
Streamlit (champ "Clé API").

## 🚀 Étape 3 : Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

Vérifiez : http://localhost:8000/health et
http://localhost:8000/docs (documentation interactive Swagger).

## 🖥️ Étape 4 : Lancer le Dashboard

Dans un second terminal :

```bash
streamlit run dashboard/app.py
```

Le dashboard s'ouvre sur http://localhost:8501 et se connecte
automatiquement à l'API sur `http://localhost:8000` (modifiable
dans la barre latérale).

Tous les chiffres affichés (AUC, bénéfice net, seuils…) sont lus dans
`models/dashboard_metrics.json`, écrit par la CELL 18 du notebook :
après un ré-entraînement, il suffit de recopier `models/`, sans
modifier `dashboard/app.py`. Le thème est défini dans
`.streamlit/config.toml` (lancer Streamlit depuis la racine du projet).

## 🧪 Tests automatisés

Une suite de tests `pytest` valide le comportement réel de l'API
(authentification, validation des entrées, système de décision à
3 niveaux, endpoint d'explicabilité) contre le **vrai modèle
CatBoost** — pas de mock, pour un test fidèle au comportement réel.

```bash
pytest tests/ -v
```

⚠️ Prérequis : le dossier `models/` doit être rempli (voir Étape 1
ci-dessus), sinon les tests échoueront au chargement du modèle,
exactement comme l'API elle-même.

## ⚠️ Limitation assumée

L'API attend des **features déjà prétraitées** (mêmes colonnes que
`feature_names.json`). Les features absentes d'une requête sont
complétées avec les valeurs d'entraînement de `preprocessing.pkl`
(médiane, ou taux de fraude global pour les colonnes target-encodées).
Attention : `card1`, `card2`, `addr1`, `P_emaildomain` et
`uid_target_enc` sont des valeurs **target-encodées** (taux de fraude
lissé entre 0 et 1), pas les identifiants bruts. En mode `stacking`,
`/predict/explain` renvoie la probabilité du stacking mais les valeurs
SHAP du modèle de base CatBoost.

Le pipeline complet de feature engineering du
notebook (UID reconstruction, target encoding, agrégations,
lag/rolling) nécessite l'historique complet des transactions et
n'est pas reproduit en temps réel dans cette version — ce choix est
documenté comme limitation dans le rapport (Chapitre 5 : Déploiement).
