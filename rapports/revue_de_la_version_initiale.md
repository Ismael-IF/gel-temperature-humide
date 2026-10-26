# Revue critique de la première version, et ce qui a changé

Ce projet part d'un test technique que j'ai rendu en septembre 2024 (1 h 49 pour le
nettoyage, environ 6 h pour la modélisation). Je l'ai repris entièrement. Ce
document liste ce qui n'allait pas, pourquoi, et ce que la correction change
numériquement. Il est volontairement direct : savoir relire son propre travail
est une compétence de métier, et les erreurs listées ici sont des erreurs
courantes, qu'aucun message d'exécution ne signale.

Les trois premières sont les seules qui rendaient les conclusions fausses. Les
autres sont des défauts de rigueur ou d'artisanat.

## Les trois erreurs qui changeaient la conclusion

### 1. Le modèle apprenait la mauvaise variable

```python
# version initiale
x = dtf1.drop('T2M_D_med', axis=1)
y = dtf1.T2M_D_med.values        # température SÈCHE du capteur
```

L'énoncé demande de prédire `T2M_U_med`, la température **humide**. La première
version a entrainé et évalué les trois approches sur `T2M_D_med`, la température
**sèche** du capteur, après avoir retiré la colonne cible du jeu de données à
l'étape de sélection des variables. Tout le projet répondait donc à une autre
question que celle posée.

Le plus gênant est que rien ne le signale : la tache est même un peu plus facile
(la température sèche sans abri est proche de la température sous abri décalée de
1,5 °C), donc les scores étaient flatteurs. Reproduit avec la chaine actuelle,
apprendre `T2M_D_med` donne 1,17 °C de RMSE, du même ordre que les 1,21 °C
obtenus sur la vraie cible. Aucune métrique ne pouvait rattraper l'erreur : seule
la relecture de l'énoncé la révèle.

**Correction.** La cible est déclarée une fois dans `config/config.toml`, et un
test vérifie qu'aucun jeu de variables ne contient ni `T2M_U_med` ni `T2M_D_med`
(`tests/test_features.py::test_aucun_jeu_de_variables_ne_fuit_la_mesure_du_capteur`).
`T2M_D_med` est une mesure du même capteur sans abri que la cible : un modèle qui
la reçoit en entrée ne peut pas servir une parcelle non équipée, ce qui est
pourtant tout l'intérêt du produit.

### 2. La formule de Stull était évaluée contre elle-même

```python
# version initiale
Tw = 0.2831*hr**(0.2735)*Ts + 0.0003018*hr**2 + 0.01289*hr - 4.0962   # "cible"
def modele(hr1, a, b, c, d):
    return a*hr1**(0.2735)*Ts + b**hr1**2 + c*hr1 - d                 # noter b**hr1
popt, pcov = curve_fit(modele, hr, Tw)
MSE = np.sqrt(mean_squared_error(Tw, Tw_pred))                        # Tw, pas la mesure
```

La cible utilisée pour ajuster puis noter la formule était la sortie de la
formule elle-même, pas la mesure du capteur. Le R² de 0,99999 et la « RMSE » de
0,011 °C ne mesuraient que l'accord de la formule avec elle-même, ce qui est vrai
par construction. La conclusion du rapport, « l'approche la plus efficace est la
formule de Stull », découlait entièrement de cet artefact.

Reproduit proprement (`artefacts/experience_pieges.json`), l'ajustement circulaire
donne un R² de 1,0 exactement. Confrontée aux mesures de terrain, la même formule
obtient un R² de 0,726 et une RMSE de 1,40 °C.

Le code contenait par ailleurs `b**hr1**2` au lieu de `b*hr1**2` : une puissance
au lieu d'un produit. L'ajustement convergeait quand même, ce qui montre à quel
point un `curve_fit` qui « marche » ne garantit rien.

**Correction.** La formule est évaluée contre la mesure, comme les autres
approches, dans deux versions : coefficients publiés (aucun apprentissage) et
coefficients recalibrés sur nos capteurs. L'exposant de l'humidité reste celui
publié ; une fois cet exposant fixé, la formule est linéaire en ses quatre
coefficients, donc le recalibrage est une simple régression linéaire. Plus de
point de départ hasardeux, plus de risque de non-convergence.

### 3. Le protocole d'évaluation ignorait la consigne temporelle

L'énoncé est explicite : apprendre sur les données du 15/03/2020 au 14/03/2023,
comparer les approches « pour tous les enregistrements disponibles depuis le
15/03/2023 ». La première version a fait un `train_test_split` aléatoire à
l'intérieur de la seule fenêtre d'apprentissage. Les 291 431 lignes postérieures
au 15 mars 2023, qui constituent le jeu de test demandé, n'ont jamais été
utilisées.

Pour la régression polynomiale, c'est pire : le modèle est ajusté et noté sur les
mêmes lignes, sans aucune séparation. Le score annoncé est un score
d'apprentissage.

Un découpage aléatoire n'est pas neutre ici. Les variables atmosphériques sont
estimées sur une grille : deux stations voisines à la même heure portent des
valeurs d'entrée identiques. Un tirage aléatoire place donc des lignes quasi
jumelles des deux cotés de la séparation. Mesuré
(`artefacts/experience_protocole.json`) : 1,165 °C en découpage aléatoire contre
1,211 °C sur la saison suivante, soit 3,9 % d'optimisme.

**Correction.** Découpage chronologique strict, avec une saison de validation
(novembre 2022 à mars 2023) réservée au choix des modèles et des
hyperparamètres, et la période de test jamais touchée avant la mesure finale.

## Les erreurs de nettoyage

### 4. Le filtre de santé capteur portait sur la mauvaise maille

```python
# version initiale
masque6 = (dtf['nbre_records'] < 4)   # seuil horaire improvisé
```

L'énoncé demande de supprimer les **journées** où le capteur émet moins de 50
fois. La première version a converti ce seuil journalier en seuil horaire en
divisant mentalement par 24, puis l'a appliqué ligne à ligne. Deux conséquences :
une journée saine dont une heure est creuse perdait cette heure, et une journée
réellement défaillante restait dans le jeu dès que ses heures dépassaient le
seuil. 135 899 lignes ont été supprimées sur ce critère, qui ne correspond pas à
la règle demandée.

Il y a un second piège, plus subtil. Le cumul journalier doit être calculé sur les
données **brutes**. Si on le calcule après avoir retiré les heures diurnes, le
cumul médian par station-jour tombe à 48 mesures, juste sous le seuil de 50 : le
filtre viderait alors la moitié du jeu pour une raison purement artificielle. Ce
détail d'ordre d'application vaut un million de lignes.

**Correction.** Cumul par `(station_id, date)` calculé sur le brut, puis
suppression des journées entières concernées : 73 819 lignes retirées, et un test
qui encode les deux cas limites
(`tests/test_cleaning.py::test_filtre_journalier_porte_sur_le_cumul_de_la_journee`).

### 5. Le filtre du site usine ne filtrait pas ce qu'il devait

```python
# version initiale
masque7 = (dtf['WD'] < 0.05) | (dtf['lat'] == 47.266) | (dtf['lon'] == -1.561)
```

Trois problèmes dans une ligne. `WD` est la direction du vent en degrés, elle n'a
rien à voir avec une distance géographique : ce test supprimait les
enregistrements dont le vent venait du nord. Les deux autres conditions testent
une égalité exacte de coordonnée, alors qu'aucune station n'est exactement au
point de référence, et qu'une égalité de latitude seule inclurait toute une
parallèle terrestre.

Résultat : 66 586 lignes supprimées sans rapport avec la question, et les 130
stations réellement situées sur le site de fabrication conservées dans le jeu.
Ces stations sont en calibration : leurs mesures sont précisément celles dont il
faut se méfier.

**Correction.** Distance euclidienne en degrés au point de Nantes, tolérance
0,05 degré. 130 stations et 15 836 lignes écartées (123 753 si la règle est
appliquée seule au brut), et un test de non-régression avec une station de même
latitude mais à 7 degrés de longitude
(`tests/test_cleaning.py::test_site_usine_est_un_disque_et_non_une_egalite`).

### 6. Détails de nettoyage

- `dtf.dropna(inplace=True)` supprime les lignes contenant un manquant sur
  **n'importe quelle** colonne. Cela fonctionnait parce qu'une seule colonne en
  contenait encore, mais la règle écrite n'était pas la règle voulue. Le filtre
  cible maintenant explicitement la colonne de la cible.
- La colonne d'index anonyme `Unnamed: 0`, artefact d'export, était conservée
  jusque dans le livrable Parquet.
- Aucun journal : impossible de dire quelle règle porte la réduction de volume.
  Le nettoyage produit maintenant `artefacts/journal_nettoyage.csv`, avec pour
  chaque règle son effet marginal et son effet si elle était appliquée seule. Les
  deux chiffres diffèrent beaucoup, et c'est l'écart qui est informatif.
- Le CSV brut de 1,4 Go était relu intégralement en float64 à chaque exécution.
  Il est converti une fois en Parquet typé (141 Mo, lecture en 1,1 s au lieu
  d'une minute), ce qui change la façon de travailler : on peut relancer la chaine
  complète pour tester une hypothèse.

## Les erreurs de mesure et d'interprétation

### 7. MSE et RMSE confondues

> « MSE = 1.2968 [...] nous trouvons en moyenne une erreur de prédiction de
> 1.13 °C »

La MSE s'exprime en degrés carrés. C'est sa racine, 1,139, qui s'exprime en
degrés. La phrase annonce une MSE et commente une RMSE. Le rapport de synthèse
affichait d'ailleurs 1,3767 pour la régression polynomiale là où le notebook
affichait 1,1733 : la même quantité, une fois au carré, une fois non, sans que la
différence soit signalée.

**Correction.** Une seule fonction calcule les métriques, les noms sont explicites
(`rmse`, `mae`, `biais`), et un test interdit la confusion
(`tests/test_metrics.py::test_rmse_et_mse_ne_sont_pas_confondues`).

### 8. Le R² lu comme un taux d'erreur

> « R2 = 0.9084 [...] une bonne capacité à prédire la variable cible avec une
> erreur de près de 9.17 % »

Le R² est une part de variance expliquée, pas un pourcentage d'erreur. 1 moins le
R² ne donne pas un taux d'erreur, et sur un problème de température en degrés
Celsius, un pourcentage d'erreur n'a de toute façon pas de sens physique (l'échelle
possède un zéro arbitraire : une erreur de 0,5 °C autour de 0 °C ne vaut pas un
pourcentage).

Le R² dépend en plus de la variance du jeu sur lequel il est calculé. Sur la
période de test, dont la dispersion est plus faible, le même modèle affiche un R²
plus bas sans être moins bon. Il est conservé dans les tableaux pour mémoire, mais
la discussion se fait sur des degrés.

### 9. Aucune des métriques métier demandées

L'énoncé suggérait explicitement la distribution des erreurs, la fréquence de
dépassement des seuils de 1 °C et 2 °C, et une table de contingence du bon
classement par rapport au gel. Aucune des trois n'était présente : la comparaison
reposait sur une MSE et un R² globaux.

C'est dommage, parce que ce sont ces métriques qui font parler les résultats. Sur
la période de test, les erreurs dépassent 1 °C dans 39 % des heures et 2 °C dans
11 % : une RMSE de 1,21 °C ne le laissait pas deviner. Et à la maille de la nuit,
qui est celle de la décision agronomique, le classement des modèles n'est pas le
même que sur la RMSE : la régression polynomiale détecte 70,7 % des nuits de gel
contre 66,9 % pour la formule recalibrée, alors que sa RMSE est légèrement moins
bonne.

J'ai ajouté une maille que l'énoncé ne demandait pas : l'évènement
« station-nuit ». Compter les erreurs à l'heure multiplie les pertes par la durée
de l'épisode, alors qu'une nuit de gel de six heures ne détruit la récolte qu'une
fois et ne déclenche le dispositif antigel qu'une fois.

### 10. Une lecture de résidus qui ne pouvait pas être faite

> « nous avons un motif aléatoire [...] nous pouvons donc suggérer que le modèle
> est approprié »

La même phrase conclut les trois modèles. Elle s'appuie sur un nuage de 1,4
million de points tracé sans transparence ni agrégation : à cette densité, le
graphique est un bloc saturé où aucune structure n'est visible, ni présente ni
absente. Un graphique illisible ne permet pas de valider un modèle.

En découpant par régime de température, la structure apparait : l'erreur augmente
nettement dans le froid, c'est-à-dire exactement dans les situations qui
déclenchent une alerte. La conclusion est donc l'inverse de celle annoncée.

**Correction.** Nuages en densité (hexbin), erreurs déclinées par régime de
température, par mois, par heure et par station.

## Les défauts d'artisanat

### 11. Un modèle trop couteux pour ce qu'il apporte

`RandomForestRegressor(n_estimators=50)` sans profondeur maximale sur 1,4 million
de lignes construit cinquante arbres développés jusqu'aux feuilles pures : quelques
gigaoctets en mémoire, un modèle sérialisé très lourd et un temps d'apprentissage
de plusieurs minutes.

`HistGradientBoostingRegressor` atteint la même performance en une quarantaine de
secondes, pour un modèle de quelques mégaoctets. Sur ce jeu, c'est même la formule
recalibrée à quatre coefficients qui obtient la meilleure RMSE : le choix du modèle
le plus riche n'était pas justifié.

### 12. Degré 3 choisi sans justification

Le degré du polynôme était fixé à 3 sans comparaison. La sélection sur la saison
de validation montre que le degré 2 est meilleur de 0,08 °C, et que le degré 3
appliqué aux dix variables de l'énoncé (285 termes fortement corrélés) dégrade
nettement le résultat. Il manquait aussi toute régularisation et toute mise à
l'échelle avant élévation en puissance, alors que les variables vont de 0,05 kPa
à 400 W/m².

### 13. Structure et reproductibilité

Deux notebooks, des chemins absolus vers le bureau de ma machine, du code
dupliqué entre les deux fichiers, aucun test, aucune configuration, et des
résultats recopiés de mémoire dans le rapport de synthèse, avec de fait deux
valeurs différentes pour la même métrique.

La version actuelle est un paquet Python testé, piloté par un fichier de
configuration, avec une interface en ligne de commande par étape. Chaque tableau
du rapport indique le fichier de `artefacts/` dont il sort, réécrit à chaque
exécution de la chaine.

## Ce que je retiens

L'erreur la plus grave n'était pas une erreur de code : prédire `T2M_D_med` au
lieu de `T2M_U_med` est une erreur de lecture de l'énoncé, et aucun test unitaire
ne l'aurait attrapée. C'est pour cela que la première chose que fait maintenant
la configuration est de déclarer la cible et les variables autorisées, et que le
rapport commence par rappeler la question posée.

L'erreur la plus instructive est l'évaluation circulaire de la formule de Stull.
Un R² de 0,99999 sur des données réelles de capteurs n'existe pas. Un score trop
beau est un signal, pas un résultat : il faut remonter à ce qui est comparé à
quoi.
