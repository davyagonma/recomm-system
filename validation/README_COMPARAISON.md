# Comparaison des résultats  CSP vs modèles classiques

Ce document synthétise la validation du pipeline CSP (CPMpy / OR-Tools) par rapport à **toutes les façons possibles** d'ajouter des contraintes sur un algorithme classique Top-N.

Scripts modulaires dans `validation/` ; résultats JSON/CSV dans `validation/results/`.

## Structure des scripts

| Script | Objectif |
| --- | --- |
| **`HEURISTICS_S1_S8.md`** | **Documentation détaillée : comment marchent S1–S8 + S4 CSP** |
| `common.py` | Utilitaires partagés |
| `classic_methods.py` | **8 méthodes classiques** (S1, S2, S3, C4–C8) |
| `compare_csp_fc_sans_contraintes.py` | CSP+FC vs FC sans contraintes |
| `compare_csp_svd_sans_contraintes.py` | CSP+SVD vs SVD sans contraintes |
| `compare_contraintes_5x5.py` | Comparaison rapide sur matrice 5×5 |
| `benchmark_efficiency.py` | Temps CSP vs S1 sans contraintes (MovieLens) |
| `benchmark_all_methods.py` | Benchmark complet : S1–S3, C4–C8 vs CSP avec contraintes |
| `benchmark_large_scale.py` | **Grandes matrices** : croisement classique vs CSP (pool 12–35) |

## Lancement

```bash
cd codes
.venv/bin/python validation/benchmark_all_methods.py --scenario both --method fc
.venv/bin/python validation/benchmark_large_scale.py --sizes 1600 6400 16000 36000 64000
.venv/bin/python validation/compare_csp_fc_sans_contraintes.py
.venv/bin/python validation/benchmark_efficiency.py --method both
```

---

## 1. Comment fonctionne l'heuristique S2 (pas à pas)

L'heuristique S2 (`select_heuristic` dans `streamlit_app/selection.py`) est la méthode de référence du mémoire pour corriger a posteriori un Top-N classique.

### Entrées

- `R_hat[u, i]` : scores de recommandation (même matrice que le CSP)
- `Cu[u]` : candidats triés par score décroissant
- `slate_size` : nombre d'items à recommander (ex. 2)
- `constraints` : règles métier (catégories, paires interdites, etc.)
- `pool_size` : taille du pool de remplacement (défaut : `slate_size × 3`)

### Algorithme

```text
1. Construire un pool = top pool_size candidats de Cu[u]
2. Slate initiale = pool[:slate_size]        ← Top-N naïf sur le pool
3. Répéter jusqu'à 2 × |pool| itérations :
   a. Si aucune violation → STOP (succès)
   b. Sinon, identifier l'item le MOINS scoreé de la slate
   c. Chercher dans le reste du pool un remplacement qui rend la slate faisable
   d. Si trouvé → remplacer l'item le moins scoreé
   e. Sinon, chercher un remplacement qui RÉDUIT le nombre de violations
   f. Si aucun remplacement → STOP (échec partiel, slate invalide)
4. Retourner la slate (valide ou non)
```

### Exemple concret (user 4, matrice 5×5, contraintes actives)

```text
Candidats Cu[4] triés par score : [1, 0, 4, ...]
Slate initiale S2 : [1, 0]   ← top 2 scores

Violations détectées :
  - category_max:0  (items 0 et 1 sont tous deux catégorie 0, max = 1)
  - pair:0,1        (paire interdite)

Item le moins scoreé : 0
Replacement testé : 4 → slate [1, 4]
  → catégories {0, 2} OK, paire OK → slate valide
```

### Limites structurelles de S2

| Limite | Conséquence |
| --- | --- |
| **Greedy local** | Remplace un seul item à la fois ; peut rester bloqué |
| **Ordre fixe** | Retire toujours le moins scoreé, pas forcément celui qui cause la violation |
| **Pool borné** | Si la solution est hors du pool, échec |
| **Pas de garantie** | Peut retourner une slate encore invalide |
| **Par utilisateur** | Ne voit pas les contraintes globales (fournisseurs inter-users) |
| **Pas d'explicabilité** | Ne vérifie pas `explanation_min` nativement |

---

## 2. Complications pas à pas pour ajouter des contraintes au classique

Contrairement au CSP où chaque contrainte est une ligne déclarative, le classique exige de **réimplémenter manuellement** chaque règle.

### Étape 0  Pipeline commun (obligatoire)

```text
R → W (FC ou SVD) → R_hat = R @ W → Cu (candidats triés)
```

Sans cette étape, pas de scores comparables au CSP.

### Étape 1  Contrainte de cardinalité (`slate_size`)

| Classique | CSP |
| --- | --- |
| `return candidates[:slate_size]` | `sum(x_ui) == slate_size` |

**Complication** : aucune pour le classique (trivial).

### Étape 2  Exclusion des items consommés

| Classique | CSP |
| --- | --- |
| Filtrer `Cu` avant le Top-N | Variables créées seulement pour `Cu[u]` |

**Complication** : doit être fait en amont dans `build_candidate_sets`. Si oublié, le classique recommande des items déjà notés.

### Étape 3  Quotas par catégorie (`category_bounds`)

| Classique | CSP |
| --- | --- |
| Fonction `_violations()` + boucle de correction S2 | `sum(x_ui for i in cat) <= max` |

**Complications** :
1. Écrire un compteur par catégorie dans la slate
2. Vérifier min ET max par catégorie
3. Décider quel item retirer quand une catégorie est sur-représentée
4. Risque : tous les top candidats sont de la même catégorie → infaisable sans le détecter

### Étape 4  Paires interdites (`forbidden_pairs`)

| Classique | CSP |
| --- | --- |
| Vérifier `(i, j) in slate` pour chaque paire | `x_ui + x_uj <= 1` |

**Complications** :
1. Lister toutes les paires interdites présentes dans la slate
2. Choisir laquelle « casser » lors du remplacement
3. Une paire interdite entre le 1er et 2e item du Top-N → correction obligatoire

### Étape 5  Exposition fournisseur globale (`provider_bounds`)

| Classique | CSP |
| --- | --- |
| Nécessite C7 (traitement séquentiel inter-users) | `sum(x_ui for all u) >= min_provider` |

**Complications** :
1. **Contrainte globale** : impossible avec S2 seul (par user)
2. Maintenir un compteur `global_provider_counts` partagé entre utilisateurs
3. **Ordre de traitement** des users influence le résultat (non déterministe métier)
4. Un user traité en dernier peut se retrouver sans solution
5. Pas de backtracking : si le user 40 « consomme » tout le quota, les users suivants échouent

### Étape 6  Support explicatif (`explanation_min`)

| Classique | CSP |
| --- | --- |
| C8 : filtrer via `W[j,i] > threshold` après coup | Variables `y_uij` + contraintes de lien |

**Complications** :
1. Accès à la matrice `W` et à l'historique `Hu`
2. Pour chaque candidat, compter les items d'historique qui « supportent » la reco
3. Retirer les items sans support, puis remplir la slate → autre boucle
4. Peut rendre la slate incomplète (`< slate_size`)
5. Seuil `support_threshold` difficile à calibrer

### Étape 7  Détection d'infaisabilité

| Classique | CSP |
| --- | --- |
| Retourne une slate partielle ou invalide silencieusement | `status = False` |

**Complication majeure** : le classique n'a pas de signal d'échec standard. Il faut ajouter manuellement un post-check.

### Étape 8  Optimalité

| Classique | CSP |
| --- | --- |
| Optimum local (premier repair trouvé) | Optimum global sur `Cu` |

**Complication** : même faisable, la slate classique peut avoir un score inférieur au CSP.

---

## 3. Catalogue des méthodes classiques implémentées

| ID | Nom | Principe | Contraintes globales | Garantie faisabilité |
| --- | --- | --- | --- | --- |
| **S1** | Top-N seul | Tri par score, ignore les contraintes | Non | Non |
| **S2** | Top-N + repair heuristique | Remplacement local de l'item le moins scoreé | Non | Non |
| **S3** | Glouton hybride | Score + diversité soft (λ × diversité) | Non | Non |
| **C4** | Glouton contraint | Ajoute item par item si faisable | Non | Non |
| **C5** | Filter-then-rank | Meilleur item/catégorie, puis S2 | Non | Non |
| **C6** | Exhaustif sur pool | Énumère C(pool, slate) combinaisons | Non | Oui dans le pool |
| **C7** | Glouton global séquentiel | C4 + compteur fournisseur inter-users | **Oui** | Non |
| **C8** | Filtre explicatif | Retire items sans support, refill | Non | Non |
| **S4** | **CSP / CPMpy** | Optimisation déclarative | **Oui** | **Oui** |

---

## 4. Résultats  matrice 5×5 avec contraintes

Contraintes : catégories (max 1/cat), paires `(0,1)` et `(2,3)`, fournisseurs globaux, `slate_size=2`.

| Méthode | Temps | Objectif | Faisable | Violations |
| --- | --- | --- | --- | --- |
| **S1** | 0.00003 s | 0.510 | **Non** | 2 |
| **S2** | 0.00005 s | 0.488 | Oui | 0 |
| **S3** | 0.00019 s | 0.488 | Oui | 0 |
| **C4** | 0.00009 s | 0.488 | Oui | 0 |
| **C5** | 0.00009 s | 0.488 | Oui | 0 |
| **C6** | 0.00005 s | 0.488 | Oui | 0 |
| **C7** | 0.00006 s | 0.488 | Oui | 0 |
| **C8** | 0.00005 s | 0.488 | Oui | 0 |
| **S4 CSP** | **0.015 s** | 0.488 | Oui | 0 |

**Observation clé** : même avec contraintes, le CSP est **~500× plus lent** que S2/C6 sur 5×5. Toutes les méthodes classiques corrigées atteignent le **même objectif** que le CSP sur cette instance.

Seul **S1** viole les contraintes (comme prévu).

---

## 5. Résultats  MovieLens avec contraintes (FC)

Contraintes synthétiques : 5 catégories (max 1/user), 4 fournisseurs, `slate_size=3`.

### Taille 1600 (40 users × 120 items)

| Méthode | Temps | Objectif | Faisable | Violations |
| --- | --- | --- | --- | --- |
| **S1** | 0.0001 s | 87.73 | Non | 25 |
| **S2** | 0.0016 s | 87.61 | Non | 19 |
| **S3** | 0.014 s | 85.57 | Non | 1 |
| **C4** | 0.0065 s | 85.28 | **Oui** | 0 |
| **C5** | 0.0009 s | 85.28 | **Oui** | 0 |
| **C6** (pool=12) | 0.028 s | 85.28 | **Oui** | 0 |
| **C6** (pool=25) | **0.285 s** | 85.28 | **Oui** | 0 |
| **C7** | 0.0024 s | 1.52 | Non | 120 |
| **C8** | 0.0004 s | 85.28 | **Oui** | 0 |
| **S4 CSP** | 0.139 s | 85.28 | **Oui** | 0 |

### Quand le classique devient-il plus lourd que le CSP ?

**Oui, c'est possible**  mais seulement pour certaines méthodes et certaines tailles :

| Condition | Méthode concernée | Exemple mesuré |
| --- | --- | --- |
| Pool exhaustif large | **C6** | pool=25, size=1600 : **0.285 s > CSP 0.155 s** |
| Diversité soft coûteuse | **S3** | size=1600 : 0.014 s (reste < CSP) |
| Repair S2 sur gros pool | **S2** | size=1600 : 0.0016 s (reste < CSP) |

**C6 exhaustif** est la méthode classique la plus dangereuse : complexité `O(C(pool, slate))`. Avec pool=25 et slate=3, cela fait 2300 combinaisons **par utilisateur**.

Le CSP reste plus lent en général, mais **C6 avec grand pool le dépasse**.

---

## 6. Synthèse : le classique devient-il plus lourd que le CSP ?

| Question | Réponse |
| --- | --- |
| Top-N sans contraintes (S1) vs CSP ? | Classique **100–1000× plus rapide** |
| Top-N + repair (S2) vs CSP ? | Classique **~100× plus rapide** (5×5 : 500×) |
| Exhaustif (C6) grand pool vs CSP ? | **Classique peut dépasser le CSP** (0.285 s vs 0.155 s) |
| Même qualité de résultat ? | C4/C5/C6/C8 = même objectif que CSP quand faisable |
| Garantie faisabilité ? | Seul **S4 (CSP)** garantit toujours |
| Contraintes globales (fournisseurs) ? | Seuls **C7** (partiel) et **S4** (complet) les gèrent |

### Pourquoi le classique reste généralement plus rapide

1. **Pas de solveur** : simples boucles Python / numpy
2. **Pas de variables booléennes** : pas de propagation CP
3. **Travail local** : S2 ne regarde qu'un pool de ~6–12 items
4. **Early stop** : dès que faisable, S2 s'arrête

### Quand préférer le CSP malgré tout

1. Contraintes **globales** (fournisseurs)  C7 est ordre-dépendant et souvent infaisable
2. Contrainte **explicative** stricte  C8 est incomplet
3. Besoin de **signal d'infaisabilité** (`status=False`)
4. Besoin d'**optimalité garantie** sans exploser le pool (C6)
5. Ajout fréquent de **nouvelles contraintes**  une ligne CP vs réécrire une heuristique

---

## 9. Résultats  grandes matrices MovieLens (avec contraintes)

Protocole : FC, `slate_size=3`, `n_candidates=50`, contraintes catégories (max 1/user) + fournisseurs.
Script : `benchmark_large_scale.py`.

### 9.1 Temps CSP de référence selon la taille

| Taille cible | Dimensions | Notes | Temps CSP (S4) |
| --- | --- | --- | --- |
| 1 600 | 40 × 120 | petite | 0.116 s |
| 6 400 | 80 × 240 | moyenne | 0.502 s |
| 16 000 | 126 × 378 | grande | 0.813 s |
| 36 000 | 190 × 570 | très grande | 1.382 s |
| 64 000 | 253 × 759 | max testé (~34 % des users ML-100K) | **1.805 s** |

Le temps CSP **augmente avec la taille** (plus de variables booléennes), mais reste sous 2 s pour 253 utilisateurs.

### 9.2 C6 exhaustif vs CSP  effet de la taille ET du pool

| Taille | Pool | Temps C6 | Temps CSP | Ratio C6/CSP | C6 > CSP ? |
| --- | --- | --- | --- | --- | --- |
| 1 600 | 12 | 0.026 s | 0.116 s | 0.22× | Non |
| 1 600 | 25 | **0.219 s** | 0.116 s | **1.89×** | **Oui** |
| 1 600 | 35 | **0.349 s** | 0.116 s | **3.01×** | **Oui** |
| 6 400 | 12 | 0.108 s | 0.502 s | 0.22× | Non |
| 6 400 | 25 | **0.643 s** | 0.502 s | **1.28×** | **Oui** |
| 6 400 | 35 | **1.922 s** | 0.502 s | **3.83×** | **Oui** |
| 16 000 | 25 | **1.136 s** | 0.813 s | **1.40×** | **Oui** |
| 16 000 | 35 | **3.359 s** | 0.813 s | **4.13×** | **Oui** |
| 36 000 | 25 | **1.833 s** | 1.382 s | **1.33×** | **Oui** |
| 36 000 | 35 | **5.532 s** | 1.382 s | **4.00×** | **Oui** |
| 64 000 | 25 | **2.680 s** | 1.805 s | **1.48×** | **Oui** |
| 64 000 | 30 | **4.900 s** | 1.805 s | **2.71×** | **Oui** |
| 64 000 | 35 | **8.215 s** | 1.805 s | **4.55×** | **Oui** |

**Conclusion empirique** : dès `pool ≥ 25`, **C6 devient plus lent que le CSP** quelle que soit la taille testée (1 600 à 64 000). Plus la matrice grandit, **l'écart se creuse** (8.2 s vs 1.8 s à 64 000).

Formule : `temps_C6 ≈ n_users × C(pool, slate_size)` → croissance **linéaire en users** et **combinatoire en pool**.

### 9.3 Autres méthodes classiques à grande échelle (pool=25, size=64000)

| Méthode | Temps | vs CSP (1.805 s) | Faisable | Objectif |
| --- | --- | --- | --- | --- |
| **S2** repair | 0.024 s | 75× plus rapide | Non (41 viol.) | 302.79 |
| **S3** hybride | 0.377 s | 4.8× plus rapide | Oui | 301.86 |
| **C4** glouton contraint | 0.121 s | 15× plus rapide | Oui | 301.86 |
| **C5** filter-then-rank | 0.017 s | 106× plus rapide | Oui | 301.86 |
| **C6** exhaustif (pool=25) | **2.680 s** | **1.5× plus lent** | Oui | 301.86 |
| **S4 CSP** | 1.805 s | référence | Oui | 301.86 |

Même à **253 users × 759 items**, S2/S3/C4/C5 restent **plus rapides que le CSP**. Seul **C6 avec grand pool** le dépasse.

### 9.4 Réponse à la question : « en augmentant la matrice, le classique devient-il plus lourd que le CSP ? »

| Méthode classique | Avec la taille de matrice seule | Avec pool exhaustif large |
| --- | --- | --- |
| S1, S2, C5 | Non  reste très rapide | Non |
| S3, C4 | Non  croît lentement (~0.1–0.5 s) | Non |
| **C6 exhaustif** | Oui si pool ≥ 25 (dès 1 600 cellules) | **Oui, de plus en plus** (8 s à 64 000) |

La taille de matrice **amplifie** le coût de C6 (plus d'utilisateurs à traiter), mais le basculement C6 > CSP est surtout dû au **pool exhaustif**, pas à la matrice seule.

Fichiers : `results/benchmark_large_scale.csv`, `results/benchmark_large_scale.json`

---

## 10. Conclusion générale du rapport

### 10.1 Validation des résultats

1. **Sans contraintes** : CSP et classique (S1) produisent des recommandations **identiques** (FC et SVD, Jaccard = 1.0).
2. **Avec contraintes (5×5)** : S2/C4/C5/C6/C8 atteignent le **même objectif** que le CSP ; S1 viole les contraintes.
3. **Faisabilité** : seul le CSP garantit `status=False` quand aucune solution n'existe ; le classique peut retourner une slate invalide silencieusement (S2 à grande échelle : 41 violations sur 253 users).

### 10.2 Vitesse  qui gagne ?

| Contexte | Gagnant | Facteur typique |
| --- | --- | --- |
| Top-N sans contraintes | **Classique (S1)** | 100–1000× plus rapide |
| Contraintes + repair (S2, C4, C5) | **Classique** | 10–100× plus rapide |
| Contraintes + diversité soft (S3) | **Classique** | 3–5× plus rapide |
| Contraintes + exhaustif grand pool (C6) | **CSP** | C6 jusqu'à **4.5× plus lent** |
| Contraintes globales (fournisseurs) | **CSP** | C7 classique échoue largement |

### 10.3 Complications du classique (synthèse)

Ajouter des contraintes au classique n'est **pas une ligne de code** comme en CSP. Chaque type de contrainte ajoute :

- un **compteur** ou une **vérification** (`_violations`) ;
- une **boucle de correction** (S2) ou un **filtre amont** (C5) ;
- parfois un **traitement global** séparé (C7 pour les fournisseurs) ;
- **aucune garantie** de faisabilité ou d'optimalité, sauf C6 qui devient coûteux.

### 10.4 Recommandation architecture (S1–S4)

```text
Contraintes actives ?
  ├─ Non           → S1 (classique pur, le plus rapide)
  ├─ Simples/local → S2 ou C5 (rapide, ~OK sur petites instances)
  ├─ Diversité     → S3 (soft) ou C4 (hard local)
  ├─ Optimal pool  → C6 petit pool OU S4 CSP si pool > 25
  └─ Globales      → S4 CSP (C7 classique insuffisant)
```

### 10.5 Conclusion

> Le modèle CSP n'est pas justifié par la vitesse brute  le Top-N classique reste supérieur pour la recommandation simple. Sa valeur apparaît dès que les contraintes métier se multiplient : faisabilité garantie, optimalité, contraintes globales et extensibilité déclarative. En revanche, tenter de reproduire cette rigueur côté classique via l'énumération exhaustive (C6) devient **plus coûteux que le CSP** dès un pool de 25 candidats, et l'écart croît avec la taille de la matrice  confirmant que l'approche CP est plus scalable que le brute-force classique pour l'optimisation sous contraintes.

---

## 7. Efficacité sans contraintes (rappel)

Sur MovieLens sans contraintes métier, le classique (S1) reste **100–1000× plus rapide** que le CSP. Voir `benchmark_efficiency.py`.

---

## 11. Fichiers générés

```text
validation/results/
├── benchmark_large_scale.csv
├── benchmark_large_scale.json
├── benchmark_all_methods_5x5_fc.csv
├── benchmark_all_methods_movielens_fc.csv
├── benchmark_all_methods_all.json
├── compare_fc_sans_contraintes.json
├── compare_svd_sans_contraintes.json
├── compare_contraintes_5x5_all.json
└── benchmark_efficiency_all.json
```

Régénérer tout :

```bash
.venv/bin/python validation/benchmark_all_methods.py --scenario both --method fc
.venv/bin/python validation/benchmark_large_scale.py --sizes 1600 6400 16000 36000 64000 --pool-sizes 12 25 30 35
.venv/bin/python validation/benchmark_efficiency.py --method both
```
