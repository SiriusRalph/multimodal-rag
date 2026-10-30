# Validation du RAG multimodal

Modèle : qwen2.5:7b. Recherche texte sur 10 requêtes, recherche image sur 7 requêtes.

## Recherche

| Mesure | Résultat |
|---|---|
| Texte, CLIP seul, rang 1 | 4/10 |
| Texte, CLIP seul, top 3 | 7/10 |
| Texte, CLIP seul, top 5 | 8/10 |
| Texte, CLIP + BM25, rang 1 | 6/10 |
| Texte, CLIP + BM25, top 3 | 8/10 |
| Texte, CLIP + BM25, top 5 | 9/10 |
| Image, visuel seul, rang 1 | 4/7 |
| Image, visuel seul, top 3 | 5/7 |
| Image, visuel + légende, rang 1 | 6/7 |
| Image, visuel + légende, top 3 | 7/7 |

## Agent

| Cas | Appels d'outil | Contexte trouvé | Réponse correcte | Durée (s) |
|---|---|---|---|---|
| texte-fait | 1 | non | non | 149.3 |
| texte-regularisation | 1 | oui | oui | 73.0 |
| texte-resultat | 1 | oui | oui | 87.3 |
| figure | 1 | oui | oui | 71.0 |
| hybride-texte-image | 1 | oui | oui | 132.0 |
| requete-image | 1 | oui | oui | 102.9 |
| hors-document | 1 | oui | oui | 110.1 |

## Traces

### texte-fait

**Question :** Combien de têtes d'attention utilise le modèle de base ?

- Action : `{"query": "number of attention heads", "image_path": ""}`
- Pages récupérées : [2, 9, 13, 14, 15]

**Réponse :** Le modèle de base utilise 8 têtes d'attention, comme indiqué sur la page 9 du papier (Table 3).

### texte-regularisation

**Question :** Quelle valeur de dropout est utilisée pendant l'entraînement ?

- Action : `{"query": "dropout training value"}`
- Pages récupérées : [3, 4, 7, 8]

**Réponse :** Le papier indique que pour le modèle "base", le taux de dropout utilisé pendant l'entraînement est de 0.1 (page 8).

### texte-resultat

**Question :** Quel score BLEU le Transformer obtient-il en traduction anglais-allemand ?

- Action : `{"query": "score BLEU anglais-allemand"}`
- Pages récupérées : [1, 3, 8, 15]

**Réponse :** Le papier "Attention Is All You Need" indique que le modèle Transformer a obtenu un score de BLEU de 28.4 sur la tâche de traduction anglais-allemand (page 8).

### figure

**Question :** Quelle figure montre l'architecture complète du Transformer, et à quelle page ?

- Action : `{"query": "architecture complète Transformer", "image_path": ""}`
- Pages récupérées : [2, 3, 4, 8, 10]

**Réponse :** Figure 1 montre l'architecture complète du Transformer et se trouve à la page 3.

### hybride-texte-image

**Question :** Trouve le schéma de la multi-head attention, puis explique à partir du texte pourquoi l'article utilise plusieurs têtes.

- Action : `{"query": "multi-head attention schema", "image_path": ""}`
- Pages récupérées : [1, 4, 5, 14]

**Réponse :** Le schéma de la multi-head attention se trouve dans Figure 2, qui se trouve sur la page 4 du papier. Figure 2 montre que la multi-head attention consiste en plusieurs couches d'attention qui fonctionnent en parallèle.

L'article utilise plusieurs têtes d'attention pour permettre le modèle d'observer l'information à partir de différents espaces de représentation simultanément. Selon le texte de la page 5, "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this." Cela signifie que si le modèle utilisait une seule tête d'attention, la moyennation des poids attribués aux différentes valeurs pourrait inhiber la capacité du modèle à prendre en compte des informations provenant de différents espaces de représentation. La multi-head attention résout ce problème en utilisant plusieurs têtes d'attention, chacune se concentrant sur des informations différentes.

### requete-image

**Question :** À quelle figure de l'article correspond cette image, et quel mécanisme décrit-elle ?

- Action : `{"image_path": "data/images/page04_xref183.png"}`
- Pages récupérées : [3, 4]

**Réponse :** La figure correspondante à l'image fournie est Figure 2, qui se trouve sur la page 4. Cette figure montre (à gauche) l'attention par produit scalaire normalisé et (à droite) l'attention à plusieurs têtes, qui consiste en plusieurs couches d'attention fonctionnant en parallèle.

Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

### hors-document

**Question :** Quel était le prix des GPU utilisés pour l'entraînement ?

- Action : `{"query": "prix GPU entraînement", "image_path": ""}`
- Pages récupérées : [2, 3, 8, 15]

**Réponse :** Le papier "Attention Is All You Need" ne fournit pas d'informations spécifiques sur le prix des GPU utilisés pour l'entraînement. Cependant, il mentionne que l'entraînement a pris 3.5 jours sur 8 P100 GPUs (page 8, Table 2) et que le temps d'entraînement peut varier de douze heures sur huit P100 GPUs (page 4, Figure 1).
