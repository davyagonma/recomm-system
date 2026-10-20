# Global Constraints

| Contrainte | Définition | Utilisation en recommandation |
| --- | --- | --- |
| AllDifferent | Impose que toutes les variables prennent des valeurs différentes. | Utiliser pour éviter les doublons, par exemple attribuer un item unique à chaque position, ou construire un top-N sans répétition. |
| AllDifferentExcept0 | Comme AllDifferent, mais la valeur 0 est autorisée plusieurs fois. | Pratique pour modéliser des slots vides, des items non assignés ou des recommandations optionnelles. |
| AllDifferentExceptN | Comme AllDifferent, mais certaines valeurs de N peuvent se répéter. | Utile pour exclure seulement certains cas spéciaux tout en gardant l’unicité sur le reste. |
| AllEqual | Force toutes les variables à avoir la même valeur. | Sert à imposer une cohérence globale, par exemple le même profil de recommandation, la même catégorie ou le même score cible. |
| AllEqualExceptN | Force l’égalité sauf pour certaines valeurs autorisées à varier. | Permet de garder une cohérence de groupe tout en laissant une valeur de secours ou de neutralisation. |
| Circuit | Force les variables à former un cycle. | Utile pour des séquences de navigation, des parcours d’items ou des recommandations ordonnées qui doivent boucler. |
| Inverse | Relie deux tableaux comme fonctions inverses. | Sert à relier une permutation d’items et sa position inverse, par exemple item vers rang et rang vers item. |
| Table | Force une combinaison de variables à correspondre à une ligne autorisée. | Très utile pour encoder des règles métier, des profils utilisateurs valides ou des combinaisons de recommandations autorisées. |
| ShortTable | Variante de Table avec des jokers. | Permet de définir des règles partielles, par exemple des contraintes sur certains attributs d’un item sans fixer tous les autres. |
| NegativeTable | Interdit les combinaisons présentes dans la table. | Sert à bloquer des scénarios non désirés, comme des associations item-utilisateur incompatibles. |
| Regular | Contraint une séquence de variables via un automate. | Adapté pour filtrer des séquences recommandées selon des règles de navigation, de diversité ou de contenu. |
| MDD | Représente les combinaisons valides sous forme de graphe acyclique. | Intéressant pour compacter des règles complexes de recommandation et accélérer le filtrage de nombreux scénarios valides. |
| IfThenElse | Exprime une condition de type si-alors-sinon. | Utile pour appliquer des règles personnalisées, par exemple changer la logique de recommandation selon le profil ou le contexte. |
| InDomain | Force une variable à rester dans un domaine donné. | Sert à borner les notes, les catégories, les tranches d’âge ou tout attribut de recommandation autorisé. |
| Xor | Impose un ou exclusif entre les arguments. | Permet de gérer des choix exclusifs, par exemple recommander soit un item, soit une alternative, mais pas les deux. |
| Cumulative | Gère l’occupation cumulée d’une ressource. | Utile pour planifier des recommandations dans le temps, limiter une capacité d’affichage ou répartir une charge de diffusion. |
| CumulativeOptional | Version de Cumulative avec tâches optionnelles. | Pratique quand toutes les recommandations ne sont pas obligatoires, par exemple dans une liste candidate dont on garde seulement certaines entrées. |
| NoOverlap | Empêche les tâches de se chevaucher. | Sert à éviter que deux recommandations occupent la même fenêtre ou le même emplacement d’affichage. |
| NoOverlapOptional | Version de NoOverlap avec tâches optionnelles. | Utile quand certains éléments recommandés peuvent être ignorés si la contrainte d’affichage devient trop forte. |
| Precedence | Imposer un ordre entre des variables. | Sert à contrôler l’ordre d’apparition des items recommandés, par exemple faire passer un contenu prioritaire avant les autres. |
| GlobalCardinalityCount | Fixe le nombre d’occurrences de chaque valeur. | Très utile pour équilibrer une liste recommandée, par exemple imposer une diversité par genre, catégorie ou source. |
| Increasing | Force une séquence non décroissante. | Peut servir à ordonner des recommandations par score, popularité ou niveau de préférence. |
| Decreasing | Force une séquence non croissante. | Adapté quand on veut classer les items du meilleur au moins bon. |
| IncreasingStrict | Force une séquence strictement croissante. | Utile pour éviter les égalités, par exemple dans un ranking où chaque position doit être distincte. |
| DecreasingStrict | Force une séquence strictement décroissante. | Sert à produire un top-N strictement ordonné sans ex æquo. |
| LexLess | Compare deux listes lexicographiquement. | Peut servir à départager deux listes recommandées et à imposer un ordre de préférence entre solutions. |
| LexLessEq | Compare deux listes lexicographiquement avec égalité autorisée. | Pratique pour sélectionner la solution la plus petite selon une règle de tie-break. |
| LexChainLess | Imposer un ordre lexicographique entre toutes les lignes d’une matrice. | Utile pour éviter les permutations équivalentes et réduire les solutions redondantes d’un recommender. |
| LexChainLessEq | Variante non stricte de LexChainLess. | Permet de stabiliser les sorties de recommandation tout en gardant plusieurs solutions équivalentes si nécessaire. |
| DirectConstraint | Appelle directement une fonction du solveur. | Sert à injecter une contrainte métier spécifique au moteur de recommandation quand aucune contrainte standard ne suffit. |



