# Version initiale, septembre 2024

Ces deux notebooks sont le rendu d'origine du test technique, conservés tels
quels, sorties comprises. Ils ne sont pas exécutables en l'état : ils pointent
vers des chemins absolus de ma machine d'alors.

Ils sont ici pour une raison précise : la
[revue critique](../../rapports/revue_de_la_version_initiale.md) chiffre ce qui
n'allait pas dans ce rendu, et une critique sans la pièce d'origine ne se vérifie
pas.

| Fichier | Contenu | Temps passé à l'époque |
| --- | --- | --- |
| `projet1_nettoyage_v2024.ipynb` | Les neuf filtres de nettoyage | 1 h 49 |
| `projet2_modelisation_v2024.ipynb` | Les trois approches de prédiction | environ 6 h |

Les trois erreurs qui invalidaient les conclusions, repérables directement dans
ces fichiers :

- `projet2`, cellule de séparation des variables : `y = dtf1.T2M_D_med`, soit la
  température sèche du capteur au lieu de la température humide demandée ;
- `projet2`, section Stull : la cible de l'ajustement est la sortie de la formule
  elle-même, d'où le R² de 0,99999 ;
- `projet2`, cellule `train_test_split` : découpage aléatoire alors que l'énoncé
  impose un découpage chronologique, et les données postérieures au 15 mars 2023
  ne sont jamais utilisées.
