# Systeme de recommandation avec contraintes CP

Ce projet implemente un pipeline de recommandation qui combine :

- une matrice utilisateur-item `R` ;
- une matrice de ponderation `W` construite soit par filtrage collaboratif, soit par facteurs latents ;
- une matrice de scores reconstruite `R_hat = R @ W` ;
- un modele de programmation par contraintes avec `CPMpy` pour produire des recommandations faisables sous contraintes.

L'objectif est de ne pas seulement recommander les items ayant les meilleurs scores, mais de produire des listes de recommandations qui respectent des contraintes comme la taille de la liste, la diversite, l'equilibre des categories, l'exposition des fournisseurs et le support explicatif.

## Structure du projet

| Fichier | Role |
| --- | --- |
| `uniform.py` | Contient les fonctions communes pour construire `W` et calculer `R_hat = R @ W`. |
| `algos1.py` | Pipeline principal : construit `W`, `R_hat`, les candidats `Cu`, puis resout le modele CP. |
| `algos.py` | Premiere version du pipeline complet. `algos1.py` est la version reprise et corrigee. |
| `recomm_memoire.py` | Prototype issu du notebook Colab avec un exemple CPMpy et des donnees simulees. |
| `fc_with_cos.py` | Exemple separe de filtrage collaboratif avec similarite cosinus. |
| `facteur_latent.py` | Exemple separe de recommandation par facteurs latents/SVD. |
| `u_data_pivot.csv` | Matrice utilisateur-item pivotee : premiere colonne `user_id`, puis une colonne par item. |
| `recommendations_algos1.csv` | Fichier genere par `algos1.py` avec les recommandations finales. |
| `exercices/` | Exemples d'utilisation de solveurs et de modeles de contraintes. |

## Pipeline general

Le pipeline principal est celui de `algos1.py` :

```text
R -> W -> R_hat -> Cu -> modele CP -> recommandations
```

### 1. Matrice utilisateur-item `R`

`R` est une matrice de taille `n_users x n_items`.

```python
R[u, i] = note donnee par l'utilisateur u a l'item i
R[u, i] = 0 si l'item n'est pas note ou pas observe
```

Exemple :

```python
R = np.array([
    [5, 3, 0, 1, 0, 0],
    [4, 2, 1, 0, 0, 0],
    [1, 0, 5, 4, 0, 0],
    [0, 4, 4, 5, 0, 0],
], dtype=float)
```

### 2. Historique utilisateur `Hu`

`Hu` est une matrice binaire construite a partir de `R`.

```python
Hu[u, i] = 1 si l'utilisateur u a deja consomme ou note l'item i
Hu[u, i] = 0 sinon
```

Dans `algos1.py`, elle est construite par :

```python
Hu = (R > 0).astype(int)
```

### 3. Construction de `W`

La matrice `W` represente les relations entre items. Elle a la taille :

```text
n_items x n_items
```

Deux methodes sont disponibles.

#### Methode `fc` : filtrage collaboratif

Cette methode utilise la co-occurrence item-item :

```text
C = R.T @ R
W = D^-1 C D^-1
```

Elle est disponible avec :

```python
method="fc"
```

#### Methode `fl` ou `svd` : facteurs latents

Cette methode utilise une SVD tronquee pour projeter les items dans un espace latent :

```text
R_centre ~= U Sigma V.T
Q = V_k.T
W = Q @ Q.T
```

Elle est disponible avec :

```python
method="fl"
```

ou :

```python
method="svd"
```

Le parametre `k` controle le nombre de facteurs latents. Il est utilise seulement avec `method="svd"`, `method="fl"` ou `method="facteurs_latents"`. Avec `method="fc"`, `k` est ignore.

Interpretation de `k` :

- petit `k` : representation plus compacte, recommandations plus generales ;
- grand `k` : representation plus detaillee, mais potentiellement plus sensible au bruit ;
- `k` ne change pas le nombre de recommandations produites, il change seulement la maniere dont `W` est construit en SVD.

### 4. Scores reconstruits `R_hat`

Une fois `W` obtenu, les scores de recommandation sont calcules avec :

```python
R_hat = R @ W
```

`R_hat[u, i]` represente le score predit de l'item `i` pour l'utilisateur `u`.

Attention : avec la methode SVD, les scores peuvent etre negatifs, car la matrice est centree avant decomposition. Ce n'est pas forcement une erreur, mais cela peut influencer l'objectif CP.

### 5. Ensembles candidats `Cu`

Pour chaque utilisateur `u`, on construit un ensemble de candidats `Cu[u]` a partir des meilleurs scores de `R_hat`.

La selection se fait dans cet ordre :

1. calculer tous les scores `R_hat[u, i]` ;
2. retirer les items deja consommes si `exclude_consumed=True` ;
3. retirer les items sous `min_score` si ce parametre est donne ;
4. trier les items restants par score decroissant ;
5. garder les `n_candidates` premiers.

Parametres importants :

| Parametre | Role |
| --- | --- |
| `n_candidates` | Nombre maximal de candidats gardes par utilisateur. |
| `exclude_consumed` | Si `True`, les items deja consommes sont retires. |
| `min_score` | Score minimum pour accepter un item candidat. |

`n_candidates` ne correspond donc pas au nombre final de recommandations. Il limite seulement le nombre d'items que le modele CP a le droit de considerer pour chaque utilisateur. Le nombre final est donne par `slate_size`.

Exemple :

```python
Cu = build_candidate_sets(
    R_hat=R_hat,
    Hu=Hu,
    n_candidates=5,
    exclude_consumed=True,
    min_score=0,
)
```

## Modele CP

Le modele CP choisit les variables binaires :

```text
x_ui = 1 si l'item i est recommande a l'utilisateur u
x_ui = 0 sinon
```

L'objectif est de maximiser la somme des scores des items selectionnes :

```text
max sum(R_hat[u, i] * x_ui)
```

Le modele est resolu avec `CPMpy` et le solveur `ortools` par defaut.

## Contraintes disponibles

### Cardinalite

Chaque utilisateur recoit exactement `slate_size` recommandations.

```python
slate_size=3
```

Cela impose :

```text
sum_i x_ui = slate_size
```

### Equilibre des categories

Si les items ont des categories, on peut imposer un minimum et un maximum d'items par categorie.

```python
item_categories = np.array([0, 0, 1, 1, 2, 2])
category_bounds = (0, 1)
```

Ici, chaque categorie peut apparaitre entre 0 et 1 fois dans la liste d'un utilisateur.

On peut aussi donner des bornes par categorie :

```python
category_bounds = {
    0: (1, 2),
    1: (0, 1),
    2: (0, 1),
}
```

### Diversite par paires interdites

Certaines paires d'items peuvent etre interdites ensemble.

```python
forbidden_pairs = [(0, 1), (2, 3)]
```

Cela impose :

```text
x_u0 + x_u1 <= 1
x_u2 + x_u3 <= 1
```

pour chaque utilisateur, si ces items sont dans ses candidats.

### Exposition des fournisseurs

Si les items appartiennent a des fournisseurs, on peut limiter l'exposition globale de chaque fournisseur sur tous les utilisateurs.

```python
item_providers = np.array([0, 1, 0, 1, 0, 1])
provider_bounds = (1, 6)
```

Ici, chaque fournisseur doit apparaitre au moins 1 fois et au plus 6 fois dans l'ensemble des recommandations.

On peut aussi donner des bornes par fournisseur :

```python
provider_bounds = {
    0: (2, 5),
    1: (1, 4),
}
```

### Support explicatif

Le support explicatif force une recommandation a etre justifiee par des items deja consommes.

On cree des variables :

```text
y_uij = 1 si l'item historique j supporte la recommandation de l'item i pour u
```

La relation entre `j` et `i` est acceptee si :

```python
abs(W[j, i]) > support_threshold
```

Parametres :

| Parametre | Role |
| --- | --- |
| `explanation_min` | Nombre minimum d'items historiques qui doivent supporter chaque recommandation. |
| `support_threshold` | Seuil minimum pour considerer que le lien `W[j, i]` est assez fort. |

Exemple :

```python
explanation_min=1
support_threshold=0.05
```

Cela signifie : chaque item recommande doit avoir au moins un item historique lie avec une force superieure a `0.05`.

Si `support_threshold` est trop bas, les explications seront nombreuses mais peu selectives. S'il est trop haut, le modele peut devenir infaisable.

## Exemple complet

```python
import numpy as np
from algos1 import recommend_complete

R = np.array([
    [5, 3, 0, 1, 0, 0],
    [4, 2, 1, 0, 0, 0],
    [1, 0, 5, 4, 0, 0],
    [0, 4, 4, 5, 0, 0],
], dtype=float)

item_categories = np.array([0, 0, 1, 1, 2, 2])
item_providers = np.array([0, 1, 0, 1, 0, 1])

result = recommend_complete(
    R=R,
    method="fc",
    slate_size=2,
    n_candidates=4,
    min_score=None,
    item_categories=item_categories,
    category_bounds=(0, 1),
    item_providers=item_providers,
    provider_bounds=(1, 6),
    forbidden_pairs=[(0, 1), (2, 3)],
    explanation_min=1,
    support_threshold=1e-12,
)

if result["status"]:
    print("Objectif :", result["objective"])
    print("Recommandations :", result["recommendations"])
    print("Explications :", result["explanations"])
else:
    print("Aucune solution faisable avec ces contraintes.")
```

## Tester `fc` et `svd`

Pour utiliser le filtrage collaboratif :

```python
result_fc = recommend_complete(R, method="fc", slate_size=2)
```

Pour utiliser les facteurs latents :

```python
result_svd = recommend_complete(R, method="svd", k=2, slate_size=2)
```

Si l'objectif SVD est negatif, cela peut venir des scores `R_hat` produits par la projection latente. Pour eviter de recommander des items avec des scores negatifs, utiliser :

```python
result_svd = recommend_complete(
    R,
    method="svd",
    k=2,
    slate_size=2,
    min_score=0,
)
```

Attention : si `min_score=0` retire trop de candidats, le modele CP peut devenir infaisable.

## Interface Streamlit

Une interface web permet aux testeurs non développeurs de configurer le pipeline et de comparer les scénarios S1 à S4.

### Lancement

```bash
cd codes
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app/app.py
```

Ouvrir l'URL affichée (par défaut http://localhost:8501).

### Fonctionnalités

- Chargement du pivot MovieLens (`u_data_pivot.csv`) ou upload CSV
- Choix de la méthode **FC** ou **SVD**, hyperparamètres `k`, `slate_size`, `n_candidates`
- Scénarios **S1** (classique), **S2** (heuristique), **S3** (hybride), **S4** (CPMpy)
- Contraintes : quotas catégories, paires interdites, exposition fournisseurs, support explicatif
- Comparaison S1–S4 sur le même utilisateur
- Export CSV des recommandations
- Genres MovieLens via `u.item` (télécharger depuis GroupLens si absent)

```bash
curl -o u.item https://files.grouplens.org/datasets/movielens/ml-100k/u.item
```

## Installation

Le projet utilise principalement :

- `numpy`
- `cpmpy`
- `ortools`, utilise par `cpmpy` comme solveur par defaut

Installation possible dans un environnement virtuel :

```bash
python3 -m venv .venv
.venv/bin/pip install numpy cpmpy ortools
```

Lancement de l'exemple :

```bash
.venv/bin/python algos1.py
```

Verification syntaxique :

```bash
.venv/bin/python -m py_compile algos1.py uniform.py
```

## Interpretation des sorties

Une sortie typique ressemble a :

```python
{
    "status": True,
    "objective": 0.46,
    "recommendations": {0: [2, 4], 1: [3, 4]},
    "explanations": {0: {2: [0, 1], 4: [1]}},
    "R_hat": ...,
    "Cu": ...,
}
```

Signification :

- `status=True` : le solveur a trouve une solution faisable ;
- `objective` : somme optimisee des scores `R_hat[u, i]` selectionnes ;
- `recommendations` : items recommandes par utilisateur ;
- `explanations` : items historiques qui supportent chaque recommandation ;
- `R_hat` : matrice complete des scores predits ;
- `Cu` : candidats consideres par utilisateur.

## Causes frequentes d'infaisabilite

Le modele peut retourner `status=False` si les contraintes sont trop fortes. Les causes courantes sont :

- `slate_size` est superieur au nombre de candidats disponibles ;
- `min_score` retire trop d'items ;
- `category_bounds` exige des categories absentes des candidats ;
- `provider_bounds` impose une exposition impossible ;
- `support_threshold` est trop eleve ;
- `explanation_min` exige trop d'items historiques de support ;
- `forbidden_pairs` interdit trop de combinaisons.

Pour deboguer, commencer avec peu de contraintes, puis les ajouter progressivement.

## Formulation mathematique simplifiee

Ensemble des utilisateurs :

```text
U = {1, ..., m}
```

Ensemble des items :

```text
I = {1, ..., n}
```

Matrice observee :

```text
R in R^(m x n)
```

Scores reconstruits :

```text
R_hat = R W
```

Variable de decision :

```text
x_ui in {0, 1}
```

Objectif :

```text
max sum_u sum_i R_hat[u, i] x_ui
```

Cardinalite :

```text
sum_i x_ui = N
```

Support explicatif :

```text
sum_j y_uij >= K x_ui
```

avec :

```text
y_uij <= x_ui
```

et `j` est accepte comme support seulement si `abs(W[j, i]) > support_threshold`.

## Remarque importante

`recomm_memoire.py` vient d'un notebook Colab et contient des cellules converties, notamment des commandes de type `!pip`. Il sert de prototype et de trace d'experimentation. Pour un usage Python propre, il faut utiliser `algos1.py`.
