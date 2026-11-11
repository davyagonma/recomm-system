# Tests — protocole d’évaluation

Ce dossier compare **deux sélecteurs** sur les mêmes scores (ou la même matrice `R` / `R̂`).

| Méthode | Idée |
| --- | --- |
| **Classique** | Top-N glouton (FC) : parcourir les items par score décroissant, **rejeter** les candidats qui violent les contraintes au fil de la construction. |
| **CSP** | Modèle CPMpy : optimiser la somme des scores sous contraintes **dures** (faisable ou inexistant). |

## Progression des expériences

1. **Simple** — sans contrainte métier (Top-N pur vs CSP à cardinalité seule).  


2. **Avec parité** — paires interdites (+ éventuellement diversité genre selon le script).  
   - `simple_csp_with_parity.py` : **FC + CSP simple** sur une ligne utilisateur (5×5 intégré ou `--csv`).  
   - `fc_with_parity.py` : variante **classique seule** avec les mêmes règles de parité.

3. **Deuxième contrainte** — parité **+** plafond par catégorie/genre.  
   - `simple_csp_with_parity.py` : `--max-items-per-genre` / `--category-max`.  
   - `csp_with_parity.py` : pipeline `R → W → R̂ → Cu → CP` sur MovieLens avec `--category-min` / `--category-max`.

**Indicateurs** (comme dans les `print` des scripts) : **temps d’exécution**, **complétude** (liste de taille `top_n` / `slate_size`), **respect des contraintes** (violations implicites en classique, statut solveur en CSP).

## Commandes types

```bash
# 5×5 intégré : FC vs CSP (paires + genre)
python test/simple_csp_with_parity.py

# MovieLens, utilisateur 196, parité classique
python test/fc_with_parity.py --csv u_data_pivot.csv --user-id 196 --forbidden-pair 50:51 --category-max 2

# MovieLens, CSP + borne catégorie
python test/csp_with_parity.py --input u_data_pivot.csv --category-max 2 --slate-size 5
```

## Lien Streamlit / bibliothèque

La démo `streamlit run streamlit_app/app.py` rejoue le **même protocole en 3 scénarios × 2 méthodes** via `recomm/evaluation.py` (même logique gloutonne que `recommend_fc` dans `simple_csp_with_parity.py`).

