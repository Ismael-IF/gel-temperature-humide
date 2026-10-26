# Résultats

Rapport détaillé du projet. Tous les tableaux viennent des fichiers du dossier
[`artefacts/`](../artefacts), écrits par la chaine de traitement : chaque
exécution de `python -m gel tout` les réécrit, et chaque chiffre cité ici se
retrouve dans le CSV nommé à la fin de la section.

Les choix de méthode derrière ces résultats sont justifiés un par un dans
[docs/choix_techniques.md](../docs/choix_techniques.md).

---

## 1. Nettoyage

Le jeu brut compte **11 773 387 lignes** et 1 739 identifiants de station. Après
les neuf filtres de l'énoncé, il reste **1 870 803 lignes** (15,9 % du volume
initial) sur 1 537 stations, du 15 mars 2020 au 31 mai 2023. La température
humide est négative sur 16,1 % des heures conservées : le nettoyage concentre le
jeu sur les situations qui intéressent le produit.

| filtre                      | lignes_supprimees | part_supprimee_pct | lignes_concernees_seule | lignes_restantes |
| --------------------------- | ----------------- | ------------------ | ----------------------- | ---------------- |
| valeurs_erreur_capteur      | 19178             | 0.163              | 19178                   | 11754209         |
| hors_saison_gel             | 4686719           | 39.808             | 4695901                 | 7067490          |
| diurne                      | 2861729           | 24.307             | 5108930                 | 4205761          |
| humide_superieure_seche     | 108307            | 0.920              | 672693                  | 4097454          |
| trop_chaud                  | 1631863           | 13.861             | 8386166                 | 2465591          |
| incoherence_abri            | 503917            | 4.280              | 2893441                 | 1961674          |
| cible_manquante             | 1216              | 0.010              | 47222                   | 1960458          |
| journee_sous_echantillonnee | 73819             | 0.627              | 1241003                 | 1886639          |
| site_usine                  | 15836             | 0.135              | 123753                  | 1870803          |

La colonne `lignes_concernees_seule` donne ce que chaque règle supprimerait si
elle était appliquée seule au jeu brut. L'écart avec `lignes_supprimees` mesure
le recouvrement entre règles, et il est important : le filtre sur la température
sous abri viserait à lui seul plus de huit millions de lignes, mais n'en retire
qu'un cinquième une fois la saison estivale et les heures diurnes déjà écartées.
Avec un seul chiffre par règle, on attribuerait toute la réduction de volume au
filtre appliqué en premier.

Deux filtres méritent une attention particulière, parce que leur maille change le
résultat :

- **Le filtre de santé capteur porte sur le cumul journalier par station**, et ce
  cumul doit être calculé sur les données brutes. Calculé après le retrait des
  heures diurnes, le cumul médian par station-jour tombe à 48 mesures, juste sous
  le seuil de 50 : le filtre viderait alors la moitié du jeu pour une raison
  purement artificielle.
- **Le filtre du site de calibration est une distance**, pas une égalité de
  coordonnée. Il concerne 130 stations et 123 753 lignes. Aucune station n'a
  exactement les coordonnées du point de référence, donc une comparaison
  d'égalité ne retirerait rien.

Deux constats de qualité de données, non demandés par l'énoncé :

- `station_id` n'identifie pas un capteur de façon stable : 365 identifiants sur
  1 537 couvrent des positions séparées de plus de 0,01 degré, jusqu'à 4 degrés
  de latitude, et 59 871 lignes brutes portent le même triplet (station, date,
  heure) avec des coordonnées différentes. Signalé, non corrigé : sans règle
  métier, choisir quelle ligne garder serait arbitraire.
- Un point de rosée à -46 °C subsiste après les filtres. Ce n'est pas une mesure,
  c'est un code d'erreur. Le cas est résiduel, une ligne sur 1,87 million.

Sources : `artefacts/journal_nettoyage.csv`, `artefacts/resume_nettoyage.json`.
Figure : [réduction du volume](figures/nettoyage_volumes.png).

---

## 2. Découpage temporel

| jeu           | lignes | debut      | fin        | stations | part_gel_pct | heures_de_gel |
| ------------- | ------ | ---------- | ---------- | -------- | ------------ | ------------- |
| apprentissage | 919081 | 2020-03-15 | 2022-05-31 | 1038     | 16.15        | 148395        |
| validation    | 660291 | 2022-11-01 | 2023-03-14 | 762      | 21.36        | 141052        |
| test          | 291431 | 2023-03-15 | 2023-05-31 | 1409     | 3.74         | 10902         |

La validation est la dernière saison de la fenêtre d'apprentissage. Elle sert à
choisir les modèles et les hyperparamètres, jamais à les noter. Une fois le choix
fait, le modèle retenu est réajusté sur apprentissage plus validation.

La proportion d'heures de gel passe de 21 % sur la validation à moins de 4 % sur
le test, qui ne couvre que la fin de la saison. Les deux périodes ne sont pas
interchangeables, et c'est volontairement la plus exigeante qui sert de juge :
les gelées tardives de printemps sont les plus rares et les plus couteuses,
puisqu'elles touchent une végétation déjà sortie de dormance.

Source : `artefacts/resume_decoupage.csv`.

---

## 3. Sélection des modèles sur la validation

**Régression polynomiale**, 56 configurations essayées (cinq jeux de variables,
trois degrés, quatre niveaux de régularisation). Les cinq meilleures :

| jeu_variables | degre | alpha | nb_termes | rmse_validation |
| ------------- | ----- | ----- | --------- | --------------- |
| complet       | 2     | 10.0  | 170       | 1.1716          |
| complet       | 2     | 1.0   | 170       | 1.1716          |
| complet       | 2     | 0.01  | 170       | 1.1716          |
| complet       | 2     | 0.1   | 170       | 1.1716          |
| enonce        | 2     | 10.0  | 65        | 1.1729          |

Le degré 3 appliqué aux dix variables de l'énoncé donne 1,256 °C de RMSE de
validation, nettement moins bon que le degré 2 : avec 285 termes fortement
corrélés, le polynôme ajuste du bruit. Choisir le degré d'avance, comme je
l'avais fait en 2024, coutait ici près d'un dixième de degré.

Le niveau de régularisation, lui, ne change presque rien : les quatre valeurs
d'alpha donnent le même résultat à la quatrième décimale. C'est un indice de plus
que le modèle n'est pas limité par sa capacité.

**Gradient boosting** :

| max_iter | learning_rate | max_leaf_nodes | min_samples_leaf | rmse_validation |
| -------- | ------------- | -------------- | ---------------- | --------------- |
| 400      | 0.1           | 31             | 50               | 1.1752          |
| 600      | 0.05          | 63             | 50               | 1.1758          |
| 800      | 0.05          | 127            | 100              | 1.1798          |

Les trois configurations se tiennent dans un centième de degré, et la plus grosse
est la moins bonne. Augmenter la capacité du modèle n'apporte rien.

Sources : `artefacts/grille_polynomiale.csv`, `artefacts/grille_gbm.csv`.

---

## 4. Erreur sur la période de test

| modele                       | RMSE (°C) | MAE (°C) | biais (°C) | erreur abs. p95 (°C) | R2    | erreurs > 1 °C (%) | erreurs > 2 °C (%) |
| ---------------------------- | --------- | -------- | ---------- | -------------------- | ----- | ------------------ | ------------------ |
| Stull recalibré              | 1.208     | 0.948    | -0.035     | 2.441                | 0.796 | 38.8               | 10.7               |
| Régression polynomiale       | 1.213     | 0.950    | -0.065     | 2.453                | 0.795 | 39.0               | 10.7               |
| Gradient boosting            | 1.218     | 0.953    | -0.064     | 2.462                | 0.793 | 39.2               | 10.9               |
| Stull (coefficients publiés) | 1.402     | 1.067    | 0.055      | 2.908                | 0.726 | 42.0               | 15.3               |
| Persistance 2 m              | 1.886     | 1.518    | 1.395      | 3.750                | 0.504 | 59.0               | 28.6               |

Trois lectures.

**Les trois approches demandées se tiennent dans un centième de degré.** Un
gradient boosting à dix-sept variables ne bat pas une formule physique à quatre
coefficients. La limite ne vient pas du modèle.

**Une RMSE de 1,21 °C ne veut pas dire « précis à 1,21 °C ».** L'erreur dépasse
1 °C dans 39 % des heures et 2 °C dans 11 %. C'est une lecture que l'énoncé
demandait explicitement, et elle est beaucoup plus parlante pour un utilisateur.

**Le biais est le chiffre à surveiller.** Il est quasi nul pour les trois modèles
ajustés, mais vaut +1,40 °C pour la persistance : prédire la température humide
par la température sous abri revient à surestimer systématiquement, donc à
sous-estimer le risque de gel. C'est l'erreur la plus couteuse, et une RMSE seule
la masque.

### Les modèles sont-ils vraiment départagés ?

Plutot que de comparer deux RMSE globales séparées par cinq millièmes, je calcule
la RMSE de chaque modèle sur chacune des 72 nuits de la période de test.

| modele                       | rmse_globale | rmse_mediane_par_nuit | nuits_gagnees | nuits_evaluees | ecart_moyen_vs_reference | ecart_type_de_l_ecart |
| ---------------------------- | ------------ | --------------------- | ------------- | -------------- | ------------------------ | --------------------- |
| Stull recalibré              | 1.2084       | 1.2060                | 38            | 72             | 0.0000                   | 0.0000                |
| Régression polynomiale       | 1.2129       | 1.2270                | 15            | 72             | 0.0233                   | 0.0503                |
| Gradient boosting            | 1.2181       | 1.2308                | 10            | 72             | 0.0276                   | 0.0516                |
| Stull (coefficients publiés) | 1.4024       | 1.3558                | 3             | 72             | 0.1425                   | 0.1280                |
| Persistance 2 m              | 1.8859       | 1.5853                | 6             | 72             | 0.4737                   | 0.3244                |

La formule recalibrée gagne 38 nuits sur 72, soit un peu plus d'une sur deux.
L'écart moyen qui la sépare de la régression polynomiale, 0,023 °C, est deux fois
plus petit que la variation de cet écart d'une nuit à l'autre, 0,050 °C : selon
la nuit, c'est l'un ou l'autre qui gagne. Annoncer un vainqueur entre ces deux-là
sur la RMSE serait abusif.

Les deux références, en revanche, sont nettement distancées : la formule non
recalibrée perd 0,14 °C en moyenne et la persistance 0,47 °C, avec des écarts
largement supérieurs à leur propre dispersion.

Sources : `artefacts/metriques_test.csv`, `artefacts/comparaison_par_nuit.csv`.
Figures : [distribution des erreurs](figures/distribution_erreurs.png),
[prédictions contre mesures](figures/predictions_vs_mesures.png).

---

## 5. Détecter les nuits de gel

La décision n'est pas prise à l'heure mais à la nuit. Une gelée est un
évènement : si une nuit compte six heures sous zéro, l'exploitant ne perd pas six
fois sa récolte et ne déclenche pas six fois son dispositif antigel.

Sur la période de test : **46 138 couples station-nuit**, dont **2 913 nuits de
gel** (6,3 %). Une nuit est gélive si le minimum nocturne de température humide
passe sous 0 °C ; le modèle alerte si le minimum qu'il prévoit passe sous le
seuil d'alerte, ici 0 °C.

| modele                       | détection POD (%) | fausses alertes FAR (%) | CSI   | nuits détectées | nuits ratées | alertes inutiles |
| ---------------------------- | ----------------- | ----------------------- | ----- | --------------- | ------------ | ---------------- |
| Régression polynomiale       | 70.7              | 20.3                    | 0.599 | 2059            | 854          | 523              |
| Gradient boosting            | 70.4              | 20.0                    | 0.598 | 2050            | 863          | 514              |
| Stull recalibré              | 66.9              | 18.7                    | 0.580 | 1949            | 964          | 449              |
| Stull (coefficients publiés) | 56.4              | 17.1                    | 0.505 | 1644            | 1269         | 340              |
| Persistance 2 m              | 31.2              | 6.4                     | 0.306 | 909             | 2004         | 62               |

**Le classement change par rapport à la RMSE.** La régression polynomiale détecte
110 nuits de gel de plus que la formule recalibrée, alors que sa RMSE est
légèrement moins bonne. L'explication est structurelle : la RMSE est dominée par
les nuits douces, de loin les plus nombreuses, tandis que la détection ne se joue
que dans la queue froide de la distribution. Un modèle un peu moins précis en
moyenne mais mieux ajusté sur le froid rend un meilleur service.

C'est la raison pour laquelle le choix final ne peut pas se faire sur une
métrique de régression seule.

Pour situer le gain : la persistance, qui est l'approximation implicite quand on
n'a pas de capteur, ne détecte que 31 % des nuits de gel. Les modèles ajustés
font plus que doubler ce taux sans aucun matériel supplémentaire.

Sources : `artefacts/contingence_gel_nuit.csv`,
`artefacts/contingence_gel_horaire.csv`.
Figure : [détection des nuits de gel](figures/contingence_gel.png).

---

## 6. Où placer le seuil d'alerte

Hypothèses économiques, paramétrables dans `config/config.toml` : une nuit de gel
non détectée coute 4 000 € par hectare (perte de récolte), une alerte inutile
120 € par hectare (mobilisation du dispositif antigel, main-d'oeuvre, énergie).
Le rapport entre les deux vaut 33.

| modele                       | seuil d'alerte (°C) | cout au seuil optimal (k€) | cout au seuil 0 °C (k€) | économie (%) | détection POD (%) | fausses alertes FAR (%) |
| ---------------------------- | ------------------- | -------------------------- | ----------------------- | ------------ | ----------------- | ----------------------- |
| Régression polynomiale       | 2.5                 | 722                        | 3479                    | 79.3         | 98.9              | 63.1                    |
| Stull recalibré              | 2.6                 | 726                        | 3910                    | 81.4         | 98.6              | 62.1                    |
| Gradient boosting            | 2.3                 | 727                        | 3514                    | 79.3         | 98.2              | 60.4                    |
| Stull (coefficients publiés) | 2.5                 | 843                        | 5117                    | 83.5         | 96.7              | 57.8                    |
| Persistance 2 m              | 3.0                 | 1080                       | 8023                    | 86.5         | 93.6              | 50.4                    |

Avec une asymétrie de couts aussi forte, le seuil qui minimise le cout total
n'est pas 0 °C mais environ **+2,5 °C sur le minimum nocturne prédit**. La
détection passe à 99 %, le cout des erreurs est divisé par cinq, et deux alertes
sur trois s'avèrent inutiles.

Ce chiffre de +2,5 °C n'est pas une vérité : il dépend du rapport entre les deux
couts, et de lui seul. D'où cette table, qui permet de se situer sans avoir à
chiffrer sa perte au centime (modèle : régression polynomiale) :

| rapport des couts | seuil d'alerte (°C) | détection POD (%) | fausses alertes FAR (%) | nuits ratées | alertes inutiles |
| ----------------- | ------------------- | ----------------- | ----------------------- | ------------ | ---------------- |
| 5                 | 1.2                 | 91.7              | 43.3                    | 243          | 2036             |
| 10                | 1.5                 | 94.4              | 47.9                    | 162          | 2529             |
| 20                | 2.0                 | 97.0              | 56.1                    | 86           | 3617             |
| 33                | 2.5                 | 98.9              | 63.1                    | 33           | 4915             |
| 50                | 2.5                 | 98.9              | 63.1                    | 33           | 4915             |
| 100               | 2.9                 | 99.7              | 68.2                    | 10           | 6223             |

Deux conséquences pratiques. Un seuil unique à 0 °C est le plus mauvais choix
possible, puisqu'il maximise le nombre de gelées ratées, qui sont la seule erreur
chère. Et le taux de fausses alertes doit être assumé explicitement, pas subi :
deux alertes inutiles sur trois est acceptable si l'alerte est bon marché à
traiter, inacceptable si elle réveille quelqu'un à trois heures du matin, ce qui
plaide pour une alerte graduée plutot que binaire.

Sources : `artefacts/seuil_alerte_recommande.csv`,
`artefacts/balayage_seuil_alerte.csv`.
Figure : [cout des erreurs selon le seuil](figures/cout_par_seuil.png).

---

## 7. Où le modèle se trompe

**Par régime de température humide mesurée** (RMSE en °C) :

| regime  | Stull (coefficients publiés) | Stull recalibré | Régression polynomiale | Gradient boosting | Persistance 2 m |
| ------- | ---------------------------- | --------------- | ---------------------- | ----------------- | --------------- |
| < -3    | 1.995                        | 1.604           | 1.620                  | 1.608             | 2.610           |
| -3 à -1 | 1.638                        | 1.462           | 1.451                  | 1.455             | 2.426           |
| -1 à 0  | 1.544                        | 1.424           | 1.376                  | 1.385             | 2.425           |
| 0 à 2   | 1.685                        | 1.511           | 1.468                  | 1.475             | 2.528           |
| 2 à 5   | 1.478                        | 1.337           | 1.340                  | 1.348             | 2.392           |
| > 5     | 1.323                        | 1.098           | 1.113                  | 1.117             | 1.505           |

L'erreur est la plus forte dans le régime le plus froid, c'est-à-dire exactement
là où une alerte se décide : 1,6 °C sous -3 °C contre 1,1 °C au-dessus de 5 °C.
Une RMSE globale de 1,2 °C n'est donc pas la précision du produit la nuit où il
sert, c'est une moyenne tirée vers le bas par les nuits sans enjeu.

Une précaution de lecture pour le découpage par heure
([figure](figures/rmse_par_heure.png)) : après le filtre sur le rayonnement, les
heures de milieu de journée ne subsistent que par temps très couvert. Elles
comptent un millier d'observations contre plus de 150 000 pour les heures de
nuit, et leur erreur plus faible décrit un régime particulier, pas une meilleure
performance.

**Dispersion entre stations** :

| modele                       | stations_evaluees | rmse_mediane | rmse_p90 | rmse_max | stations_rmse_sup_2C |
| ---------------------------- | ----------------- | ------------ | -------- | -------- | -------------------- |
| Stull recalibré              | 1277              | 1.164        | 1.565    | 2.337    | 3                    |
| Régression polynomiale       | 1277              | 1.170        | 1.535    | 2.368    | 5                    |
| Gradient boosting            | 1277              | 1.174        | 1.541    | 2.350    | 5                    |
| Stull (coefficients publiés) | 1277              | 1.318        | 1.886    | 2.921    | 82                   |
| Persistance 2 m              | 1277              | 1.847        | 2.616    | 3.498    | 529                  |

La médiane par station est plus basse que la RMSE globale : quelques capteurs
concentrent une part importante de l'erreur. Le recalibrage de la formule fait
passer le nombre de stations au-dessus de 2 °C de RMSE de 82 à 3, ce qui
représente autant de parcelles qui passent d'un service inutilisable à un service
exploitable.

**Capteurs déjà vus contre capteurs nouveaux** :

| modele                 | population         | lignes | stations | rmse  | mae   | biais  |
| ---------------------- | ------------------ | ------ | -------- | ----- | ----- | ------ |
| Stull recalibré        | stations connues   | 243471 | 1084     | 1.220 | 0.961 | -0.068 |
| Stull recalibré        | stations nouvelles | 47960  | 325      | 1.146 | 0.883 | 0.129  |
| Régression polynomiale | stations connues   | 243471 | 1084     | 1.224 | 0.962 | -0.097 |
| Régression polynomiale | stations nouvelles | 47960  | 325      | 1.155 | 0.889 | 0.099  |
| Gradient boosting      | stations connues   | 243471 | 1084     | 1.229 | 0.965 | -0.096 |
| Gradient boosting      | stations nouvelles | 47960  | 325      | 1.161 | 0.893 | 0.095  |

La période de test contient 325 stations absentes de l'apprentissage, soit une
expérience naturelle de généralisation. Leur erreur n'est pas dégradée, elle est
même un peu plus basse. Le modèle n'a donc pas appris le comportement de capteurs
particuliers mais une relation physique, ce qui est la condition pour servir une
parcelle nouvellement équipée, voire une parcelle sans capteur.

**Variables qui portent la prédiction** (importance par permutation, en °C de
RMSE perdus quand la variable est permutée) :

| variable | importance | ecart_type |
| -------- | ---------- | ---------- |
| T2M      | 1.6120     | 0.0035     |
| tw_stull | 0.3016     | 0.0015     |
| WS2M     | 0.0196     | 0.0006     |
| CLDC     | 0.0084     | 0.0007     |
| PRECIP   | 0.0070     | 0.0001     |
| wd_sin   | 0.0015     | 0.0004     |
| wd_cos   | 0.0014     | 0.0001     |
| DEWT2M   | 0.0013     | 0.0002     |

Deux variables font presque tout : la température sous abri, et l'estimation de
Stull qui résume elle-même température et humidité. Les autres n'apportent
presque rien, et pour une raison identifiable : l'humidité relative, le point de
rosée et le déficit de pression saturante sont trois écritures de la même
quantité physique. Recalculer l'humidité relative depuis la température et le
point de rosée redonne la colonne fournie à moins de 0,6 point près dans 95 % des
cas, et le déficit de pression saturante à moins de 0,01 kPa près.

Sur dix variables autorisées, trois disent la même chose. C'est l'explication du
plafond commun aux trois approches, et de l'instabilité du polynôme de degré 3.

Sources : `artefacts/metriques_par_regime.csv`,
`artefacts/metriques_par_heure.csv`, `artefacts/metriques_par_mois.csv`,
`artefacts/dispersion_par_station.csv`,
`artefacts/stations_connues_vs_nouvelles.csv`, `artefacts/importances_gbm.csv`.
Figures : [erreur par régime](figures/erreur_par_regime.png),
[par mois](figures/rmse_par_mois.png), [par heure](figures/rmse_par_heure.png),
[importance des variables](figures/importances_gbm.png).

---

## 8. Expériences de controle

**Découpage aléatoire contre découpage chronologique.** Le même modèle obtient
1,165 °C de RMSE sur un tirage aléatoire 70 / 30 de la période d'apprentissage,
1,175 °C sur la saison de validation et 1,211 °C sur la saison de test, soit
3,9 % de mieux pour le protocole permissif. L'écart reste modéré parce que le jeu
couvre trois saisons et un millier de capteurs, mais il va toujours dans le même
sens : des lignes de la même heure et de stations voisines partagent les mêmes
variables atmosphériques estimées, et se retrouvent des deux cotés de la
séparation.

**Cible erronée.** Apprendre la température sèche du capteur (`T2M_D_med`) au
lieu de la température humide donne 1,17 °C de RMSE, un chiffre du même ordre que
celui des modèles corrects. Rien dans le résultat ne signale l'erreur : seule la
relecture de l'énoncé la révèle. C'est le type de faute qu'aucune métrique ne
rattrape, et c'était celle de ma première version.

**Évaluation circulaire.** Ajuster la formule de Stull sur ses propres sorties,
puis mesurer l'accord entre l'ajustement et ces mêmes sorties, donne un R² de 1,0
et une RMSE de 0,0 °C. Le calcul est exact et vide de sens : aucune mesure de
terrain n'intervient. Confrontée aux mesures réelles, la même formule obtient un
R² de 0,726.

Sources : `artefacts/experience_protocole.json`,
`artefacts/experience_pieges.json`.

---

## 9. Formule retenue et cout d'exploitation

```
formule publiée    Tw = 0,2831 hr^0,2735 Ts + 3,018e-04 hr² + 0,01289 hr - 4,0962
formule recalibrée Tw = 0,2927 hr^0,2735 Ts + 3,818e-04 hr² - 0,04978 hr + 0,1973
```

Le coefficient principal bouge peu (0,2831 vers 0,2927). Ce qui change vraiment,
c'est le terme constant, qui passe de -4,10 à +0,20. La formule publiée décrit un
psychromètre sous abri à 2 m ; nos capteurs mesurent sans abri à 50 cm, dans le
couvert végétal. Le recalibrage absorbe ce décalage de dispositif, et c'est tout
ce qui lui manquait : il gagne 0,19 °C de RMSE et rejoint des modèles bien plus
riches.

Durées mesurées : formule recalibrée 0,2 s, régression polynomiale 70 s (dont la
recherche sur 56 configurations), gradient boosting 90 s, nettoyage complet
depuis le Parquet 6 s. Chaine complète, de la lecture des données brutes aux
figures : 255 s.

À performance équivalente, la formule recalibrée est le modèle le moins couteux à
exploiter : quelques opérations arithmétiques, aucune dépendance à une
bibliothèque de machine learning, un réentrainement qui tient dans une seconde,
et une écriture qu'un agronome peut vérifier à la main. C'est elle que je
déploierais, le gradient boosting restant utile comme référence pour savoir, à
chaque nouvelle saison, si la formule laisse du signal sur la table.

La lecture métier complète est dans [la note de décision](note_de_decision.md).

Source : `artefacts/resume_execution.json`.
