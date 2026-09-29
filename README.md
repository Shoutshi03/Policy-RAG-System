# PolicyRAG & Insight Assistant

Prototype Streamlit pour interroger des rapports de politique publique en français, en anglais et en arabe. Il ingère des PDF, extrait le texte page par page, applique l’OCR Tesseract aux pages scannées si nécessaire, indexe les passages avec leurs références et génère des réponses à partir des passages retrouvés.

> Les trois rapports préchargés sont **entièrement fictifs** et servent uniquement aux tests de l’interface. Ils ne sont pas des publications de l’ONU/du PNUD ; leurs chiffres ne doivent pas être réutilisés comme résultats réels.

## Fonctions livrées

- Import de PDF multiples, contrôle de format/signature, taille (20 Mo), nombre de pages (500), déduplication SHA-256 et nom sécurisé.
- Extraction paginée via PyMuPDF, nettoyage et repérage simple des titres/sections.
- OCR de secours avec Tesseract (français/anglais ou arabe/anglais dans l’interface).
- Chunking conservant document, page, section, langue et identifiant de passage.
- Recherche hybride : BM25 + vecteurs de caractères multilingues par feature hashing ; seuil de confiance et abstention explicite.
- Réécriture FR/EN/AR et réponse fondée sur les passages par **Qwen3.8 27B (free)** via l’API OpenRouter compatible OpenAI.
- Modes : questions/réponses, résumé, explication, comparaison, extraction de recommandations/risques/objectifs/indicateurs/parties prenantes, traduction et insights.
- Citations [S1]… limitées aux sources réellement retrouvées ; panneau avec document, page, section et extrait.
- Conversation courte conservée dans l’état de la session Streamlit.
- Adaptateur Qdrant distant facultatif, séparé par UUID de workspace. Sans QDRANT_URL, le démonstrateur reste autonome.
- API REST FastAPI/Pydantic complémentaire, tests pytest et jeu d’évaluation trilingue ; Dockerfile et Docker Compose.

## Architecture

```text
PDF → validation + SHA-256 → PyMuPDF → OCR si nécessaire → nettoyage/sections
    → chunks paginés → BM25 + vecteurs n-grammes (Qdrant facultatif)
    → reranking → contexte limité → Qwen via OpenRouter → citations filtrées / abstention
```

La recherche locale utilise des vecteurs de caractères par feature hashing pour garder le projet léger et testable. **Ce ne sont pas des embeddings neuronaux BGE-M3** ; pour une instance de production exigeant une recherche sémantique, remplacer l’adaptateur vectoriel par BGE-M3 ou multilingual-e5 et évaluer sur des rapports représentatifs.

## Démarrage rapide

### Prérequis

- Python 3.11+.
- Tesseract et ses données linguistiques français, anglais et arabe pour l’OCR.
- Une clé OpenRouter configurée côté serveur dans le fichier caché `.env` (ou dans les variables de l’hébergeur) pour activer la réécriture/génération Qwen.

### Installation Ubuntu/Linux

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-ara
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.template .env
```

Renseigner `OPENROUTER_API_KEY` dans `.env`. Ce fichier est caché et ignoré par Git ; il est lu côté serveur au démarrage. **Aucun champ de clé API n’existe dans l’interface Streamlit.** Ne jamais committer une clé ni la partager dans un ticket ou une conversation. Pour un hébergement, utiliser son gestionnaire de secrets et injecter la variable `OPENROUTER_API_KEY` dans le processus.

### Clé OpenRouter et quota gratuit

1. Créer une clé sur [OpenRouter Keys](https://openrouter.ai/keys).
2. Le modèle sélectionné par défaut est `qwen/qwen3.8-27b:free`.
3. Selon la FAQ OpenRouter consultée le 27 septembre 2026, les modèles gratuits ont une limite indicative de **50 requêtes/jour au total** sans crédits achetés ; les comptes ayant acheté au moins 10 USD de crédits peuvent relever la limite gratuite indiquée à 1 000/jour. Cette offre peut changer. Les limites par minute et la disponibilité peuvent aussi varier ; confirmer sur la [FAQ OpenRouter](https://openrouter.ai/docs/faq).
4. Le catalogue public OpenRouter consulté à cette date affiche 0 $ en entrée et en sortie pour le variant `:free`. Cela ne garantit pas une disponibilité permanente ou un quota illimité.

La clé reste côté serveur et n’est jamais exposée dans l’interface ni dans le dépôt source ; `.env` est exclu du contrôle de version et de l’archive livrée. Pour des documents confidentiels, exécuter l’application dans un environnement contrôlé et vérifier les règles de traitement du fournisseur avant d’envoyer des extraits. Sans clé, le retrieval, les citations et les réponses extractives restent disponibles.

### Lancer Streamlit

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Ouvrir `http://localhost:8501`.

### API REST FastAPI

Dans un deuxième terminal :

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Documentation interactive : `http://127.0.0.1:8000/docs`. Endpoints livrés : `GET /health`, `POST /api/workspaces`, `GET/POST /api/workspaces/{id}/documents`, `DELETE /api/workspaces/{id}/documents/{document_id}` et `POST /api/ask`. Pour ce prototype, l’UUID aléatoire d’espace sert de jeton de capacité ; ajouter authentification, autorisation, limitation de débit, audit et règles d’expiration avant toute publication de l’API.

## Tests et évaluation

```bash
pytest -q
python -m evaluation.evaluate
```

Le benchmark est déterministe et n’appelle aucun LLM. Il mesure Recall@3, MRR@3 et l’abstention sur un corpus synthétique. C’est un contrôle de câblage, pas une validation sur des politiques publiques réelles.

## Qdrant facultatif

Les documents importés restent par défaut en mémoire dans la session Streamlit. Pour tester l’adaptateur distant, définir `QDRANT_URL`, et `QDRANT_API_KEY` si le cluster l’exige. Les vecteurs et extraits envoyés à Qdrant sont filtrés par UUID de workspace. Sans URL, ou si Qdrant ne répond pas, l’application conserve la recherche locale. Ne pas utiliser un cluster partagé sans contrôler rétention et droits d’accès.

## Docker

```bash
cp env.template .env
# Renseigner OPENROUTER_API_KEY si l’inférence Qwen est souhaitée.
docker compose up --build
```

L’interface est disponible sur `http://localhost:8501`, l’API sur `http://localhost:8000/docs`. Le service Qdrant est optionnel et commenté dans `docker-compose.yml`. L’API de ce prototype ne doit pas être exposée publiquement sans contrôle d’accès.

## Limites et passage en production

- Le stockage documentaire est volatile et propre à la session Streamlit ; les PDF importés ne sont pas persistés localement par défaut.
- Le variant `:free` dépend des limites, du routage et de la disponibilité OpenRouter.
- Le gold set est synthétique ; créer un benchmark métier avant une utilisation institutionnelle.
- Le retrieval par feature hashing n’est pas un modèle d’embeddings multilingue entraîné ; envisager BGE-M3/e5, Qdrant et un reranker dédié.
- Le prototype n’ajoute pas de contrôle d’accès Streamlit, journal d’audit ni base durable multi-utilisateur. Ne pas exposer publiquement des documents sensibles.
- Aucun fine-tuning n’est effectué. Le contenu PDF est considéré comme une source non fiable et ne peut remplacer les instructions système.

## Arborescence

```text
app.py                     Interface Streamlit
api.py                     Endpoints FastAPI/Pydantic
policyrag/ingestion.py     PDF, OCR, sections et chunking
policyrag/retrieval.py     BM25 + vecteurs n-grammes + abstention
policyrag/generation.py    OpenRouter/Qwen et citations
policyrag/service.py       Orchestration des modes assistant
policyrag/qdrant_store.py Adaptateur Qdrant distant facultatif
policyrag/demo.py          Corpus fictif
 tests/                    Tests de l’application et de l’API
evaluation/                Gold set et métriques retrieval
research_sources.md        Références au modèle et au quota gratuit
```
