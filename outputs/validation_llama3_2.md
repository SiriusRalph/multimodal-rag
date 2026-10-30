# Validation du RAG multimodal

Modèle : llama3.2. Recherche texte sur 10 requêtes, recherche image sur 7 requêtes.

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
| texte-fait | 1 | non | non | 66.4 |
| texte-regularisation | 1 | oui | oui | 33.7 |
| texte-resultat | 1 | oui | non | 48.8 |
| figure | 1 | oui | oui | 37.7 |
| hybride-texte-image | 1 | oui | non | 94.1 |
| requete-image | 1 | oui | oui | 69.1 |
| hors-document | 1 | oui | oui | 67.7 |

## Traces

### texte-fait

**Question :** Combien de têtes d'attention utilise le modèle de base ?

- Action : `{"query": "number of attention heads in base model", "image_path": ""}`
- Pages récupérées : [9, 13, 14, 15]

**Réponse :** Le modèle de base utilise 8 têtes d'attention.

### texte-regularisation

**Question :** Quelle valeur de dropout est utilisée pendant l'entraînement ?

- Action : `{"query": "dropout value used during training", "image_path": ""}`
- Pages récupérées : [3, 7, 8, 15]

**Réponse :** Le dropout utilisé pendant l'entraînement est de 0.1.

### texte-resultat

**Question :** Quel score BLEU le Transformer obtient-il en traduction anglais-allemand ?

- Action : `{"query": "Transformer BLEU score English-German translation"}`
- Pages récupérées : [1, 3, 8, 15]

**Réponse :** Le Transformer obtient un score BLEU de 41,8 sur la tâche de traduction anglais-allemand, ce qui établit un nouveau record de score BLEU pour la tâche de traduction anglais-allemand. C'est une amélioration significative par rapport aux résultats précédents, qui étaient de 28,4 BLEU.

### figure

**Question :** Quelle figure montre l'architecture complète du Transformer, et à quelle page ?

- Action : `{"query": "Transformer architecture figure page", "image_path": ""}`
- Pages récupérées : [2, 3, 4, 8, 10]

**Réponse :** La figure qui montre l'architecture complète du Transformer est Figure 1, page 3.

### hybride-texte-image

**Question :** Trouve le schéma de la multi-head attention, puis explique à partir du texte pourquoi l'article utilise plusieurs têtes.

- Action : `{"query": "multi-head attention schema", "image_path": ""}`
- Pages récupérées : [1, 4, 5, 14]

**Réponse :** D'après le texte, l'article utilise plusieurs "têtes" (ou "heads") dans la multi-head attention pour permettre au modèle de jointly attendre à différentes positions et à différentes représentations subspatiales. Cela permet de capturer des relations entre les éléments de l'entrée et de les utiliser pour faire des prédictions.

En effet, le texte indique que "Multi-Head(Q, K, V ) = Concat(head1, ..., headn)", où n est le nombre de têtes. Cela signifie que le modèle utilise plusieurs poids pour chaque entrée, ce qui permet de capturer des relations plus complexes.

De plus, le texte mentionne que "Multi-Head(Q, K, V ) = Concat(head1, ..., headn)" permet de réduire la dimensionnalité de l'entrée, ce qui peut aider à éviter les problèmes de surdimensionnalité.

En résumé, les têtes de la multi-head attention permettent au modèle de mieux capturer les relations entre les éléments de l'entrée et de les utiliser pour faire des prédictions plus précises.

### requete-image

**Question :** À quelle figure de l'article correspond cette image, et quel mécanisme décrit-elle ?

- Action : `{"image_path": "data/images/page04_xref183.png", "query": "Figure of the article corresponding to the image and mechanism described"}`
- Pages récupérées : [3, 4]

**Réponse :** La figure correspondant à l'image fournie est la Figure 2, qui décrit le mécanisme de Scaled Dot-Product Attention. Ce mécanisme consiste à prendre en compte les valeurs de manière à attribuer un poids à chaque valeur en fonction de la compatibilité entre la requête et la clé correspondante. La figure montre également la mise en œuvre de Multi-Head Attention, qui consiste en plusieurs couches d'attention qui s'exécutent en parallèle.

### hors-document

**Question :** Quel était le prix des GPU utilisés pour l'entraînement ?

- Action : `{"query": "GPU cost training Transformer"}`
- Pages récupérées : [1, 3, 4, 8, 10]

**Réponse :** Le coût du GPU utilisé pour l'entraînement n'est pas explicitement mentionné dans les observations fournies. Les observations parlent du coût de l'entraînement en général, mais pas du coût spécifique des GPU utilisés. 

La paper ne donne pas d'information sur le prix des GPU utilisés pour l'entraînement.
