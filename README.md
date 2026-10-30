# RAG multimodal et agentique

Application qui extrait le texte et les figures de l'article *Attention Is All You Need*, les indexe
dans un même espace vectoriel CLIP, puis les interroge via un agent ReAct construit avec LangGraph.
Tout tourne en local : CLIP sur le CPU, ChromaDB sur le disque, le LLM avec Ollama.

## Installation

Prérequis : Python 3.14 et [Ollama](https://ollama.com).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b
New-Item -ItemType Directory -Force data
Invoke-WebRequest https://arxiv.org/pdf/1706.03762 -OutFile "data\Attention is all you need.pdf"
```

Le PDF n'est pas versionné. Si le téléchargement des poids CLIP reste bloqué, définir
`HF_HUB_DISABLE_XET=1`. Une fois les poids en cache, `HF_HUB_OFFLINE=1` évite les appels réseau.

## Utilisation

```powershell
python -m rag index                                  # indexe le PDF (ignoré s'il n'a pas changé)
python -m rag search "multi-head attention diagram"  # recherche directe, aperçu HTML dans outputs/
python -m rag search --image data\images\page03_xref128.png
python -m rag ask "Quelle figure montre l'architecture du Transformer ?"
python -m rag ask "À quelle figure correspond cette image ?" --image data\images\page04_xref183.png
```

La commande `ask` affiche le cycle de l'agent : `[ACTION]`, `[OBSERVATION]`, puis `[RÉPONSE]`, avec
`[REASONING]` lorsque le LLM écrit un raisonnement avant son action.
Pour changer de modèle : `$env:OLLAMA_MODEL = "llama3.2"`.

## Architecture

| Étape | Module | Rôle |
|---|---|---|
| 1. Extraction | `rag/extraction.py` | Texte découpé en fragments par page, images du PDF, rendu des figures vectorielles |
| 2. Vectorisation | `rag/embeddings.py` | CLIP ViT-B-32 (open_clip), vecteurs normalisés |
| 3. Indexation | `rag/store.py` | ChromaDB (cosinus), identifiant SHA-256, métadonnées |
| 4. Recherche | `rag/retrieval.py` | Requête texte ou image, top k, aperçu HTML |
| 5. Outil | `rag/tools.py` | `retriever_tool` pour le LLM |
| 6. Agent | `rag/agent.py` | Graphe LangGraph : MIND (LLM) ↔ BODY (outil) |
| 7. Validation | `validation/run_validation.py`, `tests/` | Mesures de recherche, traces de l'agent, tests |

## Choix guidés par le document et les mesures

- **Figures vectorielles.** Le PDF ne contient que 3 images matricielles (Figures 1 et 2). Les Figures 3
  à 5 sont vectorielles : elles sont rendues depuis leur zone de page, et leur texte est retiré des fragments.
- **Fragments courts.** CLIP n'accepte que 77 jetons. Le texte est découpé en 166 fenêtres d'au plus
  75 jetons, avec recouvrement, sinon la fin de chaque fragment serait ignorée.
- **Schémas hauts.** Les images sont complétées en carré avant CLIP, dont le recadrage central couperait
  le haut et le bas de la Figure 1.
- **Modalités classées séparément.** Les cosinus texte-texte (environ 0,7) dépassent nettement les
  cosinus texte-image (environ 0,3) : un classement unique éliminerait les images. Le mode `all` alterne donc les deux.
- **Recherche renforcée.** Pour les requêtes texte, CLIP seul trouve mal les faits précis. Les images sont
  reclassées avec leur légende, et les passages avec BM25 (fusion par rangs réciproques).
- **Outil simplifié.** Le nombre de résultats et les modalités sont fixés dans l'outil (5 passages et
  2 figures), car `llama3.2` demandait `k=1` ou des valeurs invalides.

## Résultats

Mesures sur 10 requêtes texte et 7 requêtes image écrites pour ce projet. Ce jeu est petit et a servi à
choisir les améliorations : ces chiffres sont indicatifs, pas une mesure de généralisation.

| Recherche | Rang 1 | Top 3 | Top 5 |
|---|---|---|---|
| Texte, CLIP seul | 4/10 | 7/10 | 8/10 |
| Texte, CLIP + BM25 | 6/10 | 8/10 | 9/10 |
| Image, visuel seul | 4/7 | 5/7 | non mesuré |
| Image, visuel + légende | 6/7 | 7/7 | non mesuré |

| Agent | Contexte trouvé | `llama3.2` | `qwen2.5:7b` |
|---|---|---|---|
| Fait précis : nombre de têtes | non | non | non |
| Valeur du dropout | oui | oui | oui |
| Score BLEU anglais-allemand | oui | non | oui |
| Figure de l'architecture | oui | oui | oui |
| Hybride : schéma et explication | oui | non | oui |
| Requête image | oui | oui | oui |
| Hors document : prix des GPU | oui | oui | oui |
| **Total** | | **4/7** | **6/7** |
| Durée moyenne par question, sur CPU | | environ 60 s | environ 105 s |

`qwen2.5:7b` est le modèle par défaut. Une réponse ne compte comme correcte que si la bonne page a été
récupérée. Le cas du nombre de têtes échoue pour cette raison : les deux modèles répondent 8, mais le passage
`h = 8` de la page 5 n'a pas été récupéré. Les traces sont dans
[`outputs/validation_qwen2_5_7b.md`](outputs/validation_qwen2_5_7b.md) et
[`outputs/validation_llama3_2.md`](outputs/validation_llama3_2.md).

## Limites

- **Comportement de l'agent.** Avec les deux modèles, l'agent appelle l'outil une seule fois par question :
  il ne reformule jamais sa recherche, donc la boucle ReAct ne dépasse pas un tour sur ces 7 cas.
- **Fiabilité.** `llama3.2` confond des valeurs présentes dans le contexte : il donne 41,8 (anglais-français)
  pour le score anglais-allemand, qui est 28,4. `qwen2.5:7b` fait mieux, mais il cite parfois une mauvaise
  page : les « douze heures » d'entraînement sont page 2, pas page 4. Les citations doivent être vérifiées.
- **Requêtes en français.** `qwen2.5:7b` n'écrit pas toujours ses requêtes en anglais malgré la consigne,
  ce qui peut réduire la qualité de la recherche.
- **Pas de vision côté LLM.** Le LLM lit les légendes et les métadonnées, pas les pixels. La partie
  multimodale porte sur la recherche, pas sur l'analyse visuelle des figures.
- **Figure 2.** Ses deux schémas partagent une seule légende, donc la recherche les distingue mal.
- **Extraction.** Les formules sont souvent dégradées dans le texte extrait, et le titre « Attention
  Visualizations » est inclus dans l'image de la Figure 3.

## Tests

```powershell
python -m pytest tests -q                 # 7 tests, sans LLM
python validation\run_validation.py       # recherche et agent, écrit dans outputs/
```
