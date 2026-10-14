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
│   ├── online_state.pkl         ← historique en ligne au début du test (scripts/build_online_state.py)
│   ├── example_raw_transactions.json ← les 3 transactions de test sous forme BRUTE
│   ├── online_parity.json / replay_report.json / feedback_experiment.json
│   │                            ← résultats des vérifications temps réel
│   ├── lgb_model.pkl / xgb_model.pkl / autoencoder.keras / scaler.pkl
│   │                            ← modèles de comparaison (non utilisés par l'API)
│   └── feature_names_lgb_xgb.json
├── api/
│   └── main.py           ← FastAPI (backend de prédiction)
├── realtime/
│   ├── engine.py          ← moteur de features en ligne (historique cartes / clients)
│   └── scoring.py         ← chargement des modèles pour les scripts
├── scripts/
│   ├── build_online_state.py   ← construit l'état en ligne + vérifie la parité
│   ├── replay_stream.py        ← rejoue la période de test via l'API (latence)
│   ├── feedback_experiment.py  ← effet des labels confirmés (/feedback)
│   └── export_stream_sample.py ← data/stream_sample.csv.gz pour l'onglet Live
├── dashboard/
│   └── app.py             ← Streamlit (interface)
├── data/stream_sample.csv.gz ← 3 000 transactions brutes de test (onglet Live)
├── tests/                 ← tests pytest (API + temps réel)
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
| `stacking` | LightGBM + XGBoost + CatBoost → régression logistique | AUC 0.918 — bénéfice net $1,229,000 |
| `catboost` | CatBoost seul | AUC 0.907 — bénéfice net $1,175,880 |

Les seuils de décision (`<modèle>_profit_optimized` → revue manuelle,
`<modèle>` (F1) → blocage) sont choisis par le notebook sur la
validation, pour le modèle concerné.

```powershell
$env:FRAUD_MODEL="catboost"   # optionnel : forcer CatBoost seul
```

### Historique en ligne (temps réel)

L'endpoint `/predict/raw` calcule lui-même les 512 features d'une
transaction **brute** à partir d'un historique des cartes et des
clients. Cet historique est construit une fois, à partir des données
brutes du notebook (`train_transaction.csv` et `train_identity.csv`,
dans le dossier parent par défaut) :

```bash
python scripts/build_online_state.py      # ~3 min -> models/online_state.pkl
```

Le script rejoue train + validation (472 432 transactions), sauvegarde
l'état au premier jour du test, puis rejoue tout le test transaction
par transaction et vérifie qu'il retrouve exactement les résultats du
notebook (AUC 0.918, 2 524 fraudes détectées, $1,229,000). Sans ce
fichier, l'API démarre quand même, mais `/predict/raw` répond 503.

## 🔐 Étape 2 : Définir la clé API (sécurité)

Tous les endpoints sauf `/health` (`/predict/raw`, `/feedback`,
`/predict`, `/predict/batch`, `/model/info`, ...) sont protégés par
une clé API. Définissez-la avant de lancer le serveur
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

Le dashboard Streamlit lit automatiquement la même clé dans `.env`
(côté serveur : elle n'est jamais envoyée au navigateur), il n'y a
donc rien à saisir. Une autre clé peut être utilisée via « Use a
different key » dans la barre latérale. L'adresse de l'API peut
aussi être définie dans `.env` avec `FRAUD_API_URL`.

## 🚀 Étape 3 : Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

Vérifiez : http://localhost:8000/health et
http://localhost:8000/docs (documentation interactive Swagger).

### Endpoints temps réel

| Endpoint | Rôle |
|---|---|
| `POST /predict/raw` | score une transaction **brute** (`{"transaction": {...colonnes du CSV...}}`) ; met à jour l'historique |
| `POST /feedback` | label confirmé a posteriori (`{"transaction_id": ..., "is_fraud": 0/1}`) |
| `GET /online/status` | taille de l'historique en ligne |

Une transaction déjà scorée est refusée (409). La limite de débit de
ces endpoints (`FRAUD_STREAM_RATE_LIMIT`, 6000/minute par défaut) est
celle d'un flux de paiements, pas d'un utilisateur.

Simulation d'un flux réel (API démarrée depuis l'état sauvegardé —
redémarrez-la avant chaque rejeu) :

```bash
python scripts/replay_stream.py --limit 5000                       # parité + latence
python scripts/feedback_experiment.py                              # effet des labels (7 / 30 jours)
```

Résultats mesurés : 13 ms par transaction (médiane aller-retour),
scores identiques au notebook ; avec des labels confirmés après 7 jours,
AUC 0.941 et +9,6 % de bénéfice net à charge de revue égale.

## 🖥️ Étape 4 : Lancer le Dashboard

Dans un second terminal :

```bash
streamlit run dashboard/app.py
```

Le dashboard s'ouvre sur http://localhost:8501 et se connecte
automatiquement à l'API sur `http://localhost:8000` (modifiable
dans la barre latérale).

L'onglet **Live stream** envoie les transactions de
`data/stream_sample.csv.gz` (début de la période de test, dans
l'ordre) à `/predict/raw`, une par une, et affiche les décisions, la
part de transactions signalées et le temps de réponse.

Tous les chiffres affichés (AUC, bénéfice net, seuils…) sont lus dans
`models/dashboard_metrics.json`, écrit par la CELL 18 du notebook :
après un ré-entraînement, il suffit de recopier `models/`, sans
modifier `dashboard/app.py`. Le thème est défini dans
`.streamlit/config.toml` (lancer Streamlit depuis la racine du projet).

## 🧪 Tests automatisés

Une suite de 31 tests `pytest` valide le comportement réel de l'API
(authentification, validation des entrées, système de décision à
3 niveaux, explicabilité, scoring temps réel et feedback) contre les
**vrais modèles** — pas de mock, pour un test fidèle au comportement
réel. Les tests temps réel (`tests/test_realtime.py`) sont ignorés si
`models/online_state.pkl` n'a pas encore été construit.

```bash
pytest tests/ -v
```

⚠️ Prérequis : le dossier `models/` doit être rempli (voir Étape 1
ci-dessus), sinon les tests échoueront au chargement du modèle,
exactement comme l'API elle-même.

## ℹ️ Deux façons d'appeler l'API

- **`/predict/raw` (temps réel)** : la transaction brute ; l'API
  calcule toutes les features à partir de son historique en ligne.
- **`/predict`, `/predict/batch`, `/predict/explain`** : un vecteur de
  features déjà calculé (utilisé par les exemples du dashboard). Les
  features absentes sont complétées avec les valeurs d'entraînement de
  `preprocessing.pkl`. Attention : `card1`, `card2`, `addr1`,
  `P_emaildomain` et `uid_target_enc` y sont des valeurs
  **target-encodées** (taux de fraude lissé entre 0 et 1), pas les
  identifiants bruts. En mode `stacking`, `/predict/explain` renvoie la
  probabilité du stacking mais les valeurs SHAP du modèle de base
  CatBoost.

Limite connue : l'historique en ligne vit dans la mémoire d'un seul
processus et n'est pas sauvegardé à l'arrêt (un redémarrage repart de
`online_state.pkl`). Un déploiement à plusieurs instances le placerait
dans un stockage partagé (ex. Redis).
