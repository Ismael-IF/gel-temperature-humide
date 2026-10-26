# Journal des choix techniques

Chaque décision de ce projet, pourquoi je l'ai prise, et ce que j'ai écarté. Ce
document existe parce qu'un résultat sans justification ne se discute pas : six
mois plus tard, ni moi ni personne ne saurait dire si un choix était réfléchi ou
subi.

Il se lit dans l'ordre du projet : cadrage, données, protocole, modèles,
évaluation, outillage.

---

## 1. Cadrage

### 1.1 La cible est `T2M_U_med`, et elle est déclarée dans la configuration

**Décision.** La variable à prédire est la température humide horaire médiane à
50 cm sans abri. Elle est écrite une seule fois, dans `config/config.toml`, et un
test vérifie qu'elle n'apparait dans aucun jeu de variables d'entrée.

**Pourquoi.** Dans ma première version, j'avais entrainé les modèles sur
`T2M_D_med`, la température sèche du même capteur. Le code tournait, les scores
étaient bons, et le projet répondait à une autre question que celle posée. Je
préfère maintenant que la cible soit un paramètre visible plutot qu'une ligne de
code au milieu d'un notebook.

### 1.2 Seules les 10 variables atmosphériques servent d'entrée

**Décision.** Ni l'identifiant de station, ni l'heure, ni la latitude. Les
variables dérivées sont toutes des fonctions des 10 autorisées.

**Pourquoi.** C'est la consigne, mais surtout c'est le cas d'usage. L'intérêt du
modèle est de servir une parcelle **sans capteur**. S'il s'appuie sur
l'identifiant d'une station, il ne sait rien faire d'une parcelle qu'il n'a
jamais vue. Ajouter l'heure et le mois améliorerait sans doute le score sur ce
jeu de données, mais au prix de ce qui fait la valeur du produit.

**Alternative écartée.** Un modèle enrichi avec l'heure, la position et la
station. Je l'ai gardé en tête comme piste d'amélioration en production, pas
comme réponse à l'énoncé.

### 1.3 La décision se prend à la nuit, pas à l'heure

**Décision.** Les métriques de détection et le calcul de cout sont faits à la
maille station-nuit, en prenant le minimum nocturne. Les métriques d'erreur
restent horaires.

**Pourquoi.** Une gelée est un évènement. Si une nuit compte six heures sous
zéro, l'exploitant ne perd pas six fois sa récolte et ne déclenche pas six fois
son dispositif antigel. Compter les erreurs à l'heure multiplierait le cout par
la durée de l'épisode, ce qui fausse complètement l'arbitrage économique.

Une nuit va de midi à midi, pour garder dans le même évènement la soirée et la
fin de nuit qui la suit.

---

## 2. Données et nettoyage

### 2.1 Conversion du CSV en Parquet typé, une fois pour toutes

**Décision.** Le CSV brut de 1,4 Go est converti en Parquet avec des types
compacts (float32, int32), par blocs de 2 millions de lignes. 141 Mo, lecture en
1,1 seconde.

**Pourquoi.** Relire le CSV prenait près d'une minute et saturait la mémoire. Ce
n'est pas qu'un gain de confort : quand une exécution coute une minute, on
n'essaie pas d'hypothèse ; quand elle coute une seconde, on en essaie dix. Les
mesures météo ont au plus une décimale, float32 est largement suffisant.

### 2.2 Chaque filtre est une fonction qui renvoie un masque

**Décision.** Les neuf filtres de l'énoncé sont neuf fonctions
`(DataFrame) -> masque booléen des lignes à supprimer`, assemblées par
`nettoyer()`.

**Pourquoi.** Chacune devient testable isolément, et le journal d'audit peut
mesurer l'effet de chaque règle. Dans ma première version, les filtres étaient
une suite de masques anonymes dans des cellules de notebook : impossible de
tester, impossible de dire quelle règle supprimait quoi.

### 2.3 Le journal de nettoyage donne deux chiffres par règle

**Décision.** Pour chaque filtre, le journal donne les lignes qu'il retire
effectivement dans l'enchainement, et celles qu'il retirerait s'il était
appliqué seul au jeu brut.

**Pourquoi.** Les règles se recouvrent beaucoup. Le filtre sur la température
sous abri viserait huit millions de lignes à lui seul, mais n'en retire qu'un
cinquième une fois la saison estivale et les heures diurnes déjà écartées. Avec
un seul chiffre, on attribue toute la réduction de volume à la règle appliquée
en premier, ce qui est arbitraire.

### 2.4 Le filtre de santé capteur se calcule sur les données brutes

**Décision.** Le cumul de mesures par station et par jour est calculé **avant**
tout autre filtre, et sert ensuite à supprimer les journées sous les 50 mesures.

**Pourquoi.** C'est le point le plus piégeux du nettoyage. Si on calcule ce cumul
après avoir retiré les heures diurnes, le cumul médian par station-jour tombe à
48 mesures, juste sous le seuil : le filtre viderait la moitié du jeu pour une
raison purement artificielle. L'ordre d'application vaut ici un million de
lignes.

Et le seuil porte bien sur la journée entière, pas sur l'heure : une journée
saine peut avoir une heure creuse, une journée défaillante peut avoir une heure
correcte.

### 2.5 Le filtre du site de calibration est une distance

**Décision.** Sont écartées les stations situées à moins de 0,05 degré du point
de Nantes, au sens de la distance euclidienne en degrés.

**Pourquoi.** Une comparaison d'égalité sur la latitude ou la longitude ne retire
rien, puisque aucune station n'est exactement au point de référence. Ce filtre
concerne 130 stations et 123 753 lignes : ce sont des capteurs en fabrication ou
en calibration, donc précisément ceux dont les mesures ne sont pas fiables.

### 2.6 Ce que j'ai constaté sans que l'énoncé le demande

- `station_id` n'identifie pas un capteur de façon stable : 365 identifiants sur
  1 537 couvrent des positions séparées de plus de 0,01 degré, et 59 871 lignes
  brutes portent le même triplet (station, date, heure) avec des coordonnées et
  des mesures différentes. Je le signale dans le rapport, je ne le corrige pas :
  sans règle métier, choisir quelle ligne garder serait une décision arbitraire.
- L'humidité relative, le point de rosée et le déficit de pression saturante
  sont trois écritures de la même quantité physique. Recalculer l'humidité
  relative à partir de la température et du point de rosée redonne la colonne
  fournie à moins de 0,6 point près dans 95 % des cas. C'est l'explication du
  plafond de performance commun aux trois approches.

---

## 3. Protocole d'évaluation

### 3.1 Découpage strictement chronologique

**Décision.** Apprentissage du 15/03/2020 au 31/05/2022, validation du
01/11/2022 au 14/03/2023, test à partir du 15/03/2023. Aucun recouvrement.

**Pourquoi.** C'est la consigne de l'énoncé, et c'est aussi la seule façon de
simuler la réalité : en production, on prédit une saison qu'on n'a pas vue.

J'ai chiffré ce que coute l'alternative : avec un découpage aléatoire, le même
modèle affiche 1,165 °C de RMSE contre 1,211 °C sur la saison suivante, soit
3,9 % d'optimisme. L'explication est que les variables atmosphériques sont
estimées sur une grille : deux stations voisines à la même heure portent des
entrées identiques, et un tirage aléatoire place ces quasi-jumelles des deux
cotés de la séparation.

### 3.2 Une saison de validation, et pas de validation croisée

**Décision.** La dernière saison de la fenêtre d'apprentissage sert à choisir les
modèles et les hyperparamètres. Pas de validation croisée par blocs temporels.

**Pourquoi.** Une validation croisée temporelle aurait multiplié par cinq le
temps d'apprentissage pour départager des configurations qui se tiennent dans un
centième de degré. Le gain d'information n'aurait pas changé une seule décision.
Sur un jeu plus petit ou des écarts plus serrés, le choix aurait été différent.

### 3.3 Le modèle retenu est réentrainé sur apprentissage plus validation

**Décision.** Une fois les hyperparamètres choisis sur la validation, le modèle
final est réajusté sur les deux périodes réunies.

**Pourquoi.** Se priver de la saison la plus récente pour prédire la saison
suivante n'aurait pas de sens. La validation a servi à choisir, pas à mesurer :
la mesure, c'est le test.

### 3.4 La période de test est la plus difficile, et c'est volontaire

**Observation.** Le test ne couvre que la fin de la saison (15 mars au 31 mai
2023), avec 3,7 % d'heures de gel contre 21 % sur la validation.

**Pourquoi je la garde.** Ce sont les gelées tardives de printemps, les plus
rares et les plus couteuses, puisqu'elles touchent une végétation déjà sortie de
dormance. Un modèle qui ne sert pas à ce moment-là ne sert à rien.

---

## 4. Modèles

### 4.1 Deux références de controle en plus des trois approches demandées

**Décision.** J'ajoute `Persistance 2 m` (prédire la température humide par la
température sous abri) et la formule de Stull non recalibrée.

**Pourquoi.** Un score ne veut rien dire tout seul. La persistance ne demande ni
données ni calcul, la formule publiée ne demande aucun apprentissage : tout
modèle qui ne les bat pas largement ne justifie pas d'être mis en production.
Ici, la persistance plafonne à 31 % de nuits de gel détectées contre 71 % pour
les modèles ajustés, ce qui donne la mesure du gain réel.

### 4.2 La formule de Stull recalibrée est une régression linéaire

**Décision.** L'exposant de l'humidité reste celui publié par Stull (0,2735), et
les quatre coefficients sont réestimés par moindres carrés.

**Pourquoi.** À exposant fixé, la formule `Tw = a·hr^p·Ts + b·hr² + c·hr + d` est
linéaire en (a, b, c, d). La recalibrer est donc une régression linéaire sur
trois colonnes : résultat exact, instantané, sans point de départ à choisir ni
risque de non-convergence. C'est aussi ce qui m'évite l'erreur de ma première
version, un `curve_fit` à quatre paramètres lancé sans réfléchir.

**Alternative testée puis écartée.** Optimiser aussi l'exposant. Gain mesuré :
0,002 °C de RMSE sur le test, pour un paramètre de plus et une formule qui
s'éloigne de la littérature sans raison physique. Je garde l'exposant publié.

**Pourquoi cette recalibration a du sens.** La formule publiée décrit un
psychromètre sous abri à 2 m. Nos capteurs mesurent sans abri à 50 cm, dans le
couvert végétal. Le recalibrage absorbe ce décalage de dispositif, et c'est tout
ce qui lui manquait : il gagne 0,2 °C sur la formule d'origine et rejoint des
modèles bien plus riches.

### 4.3 Le degré du polynôme est choisi, pas décidé

**Décision.** 56 configurations comparées sur la validation : cinq jeux de
variables, trois degrés, quatre niveaux de régularisation Ridge. Le degré 2 sur
le jeu complet l'emporte.

**Pourquoi.** « La régression polynomiale la plus pertinente possible » ne veut
pas dire « un degré choisi d'avance ». Dans ma première version j'avais pris le
degré 3 sans comparaison : la validation montre qu'il perd 0,08 °C, parce que ses
285 termes fortement corrélés ajustent du bruit.

Trois détails qui comptent : la standardisation précède l'élévation en puissance
(sans quoi des variables allant de 0,05 kPa à 400 W/m² donnent une matrice mal
conditionnée), la régularisation Ridge est indispensable dès le degré 2 à cause
de la colinéarité, et les configurations dont le développement dépasse 400 termes
sont écartées parce qu'elles ne tiennent pas en mémoire et ne seraient de toute
façon pas déployables.

### 4.4 La grille polynomiale est cherchée sur un échantillon

**Décision.** La recherche des hyperparamètres se fait sur 250 000 lignes tirées
au hasard (graine fixée) ; le modèle retenu est réajusté sur la totalité.

**Pourquoi.** Un polynôme de degré 3 sur 10 variables produit 285 termes : à
920 000 lignes, la matrice dépasse 2 Go, pour un classement des configurations
qui ne change pas. C'est un compromis assumé, écrit dans le code et dans le
rapport, pas un raccourci caché.

### 4.5 Gradient boosting plutot que foret aléatoire ou réseau de neurones

**Décision.** `HistGradientBoostingRegressor` de scikit-learn.

**Pourquoi.** Ma première version utilisait une foret aléatoire de 50 arbres non
élagués sur 1,4 million de lignes : plusieurs gigaoctets en mémoire, plusieurs
minutes d'apprentissage, un modèle énorme à sérialiser. Le gradient boosting sur
histogrammes atteint la même performance en 90 secondes, pour quelques
mégaoctets, sans mise à l'échelle ni imputation. Un réseau de neurones aurait
ajouté des hyperparamètres et une dépendance lourde pour un problème qui plafonne
de toute façon.

### 4.6 Les variables dérivées encodent de la physique, pas des astuces

**Décision.** Direction du vent en sinus et cosinus, écart au point de rosée,
pressions de vapeur saturante et réelle (Tetens), cisaillement de vent, et
l'estimation de Stull elle-même donnée en entrée.

**Pourquoi.** La direction du vent est circulaire : 359 et 1 degré sont voisins,
alors qu'un arbre les traite comme deux extrêmes. Les pressions de vapeur évitent
au modèle de réapprendre une exponentielle par morceaux. Et donner l'estimation
de Stull en entrée revient à ne lui laisser à apprendre que la correction entre
l'abri à 2 m et le capteur à 50 cm, ce qui est une tache bien plus simple. C'est
confirmé par l'importance par permutation : la température sous abri et
l'estimation de Stull font presque tout.

---

## 5. Évaluation

### 5.1 Trois niveaux de lecture, pas un seul chiffre

**Décision.** L'erreur en degrés (RMSE, MAE, biais, dépassements de 1 et 2 °C),
la qualité de détection du gel (POD, FAR, CSI), et le cout des erreurs de
décision.

**Pourquoi.** Chacun intéresse quelqu'un de différent, et ils ne donnent pas le
même classement. La régression polynomiale a une RMSE légèrement moins bonne que
la formule recalibrée, mais détecte une centaine de nuits de gel de plus. La
raison est structurelle : la RMSE est dominée par les nuits douces, qui sont les
plus nombreuses, alors que la détection ne se joue que dans la queue froide.
Choisir sur la seule RMSE aurait conduit à retenir le modèle qui rend le moins
service.

Le biais est signé, et c'est volontaire : un modèle qui surestime la température
humide sous-estime le risque de gel, ce qui est la seule erreur réellement
couteuse. Une RMSE le masque complètement.

### 5.2 Les modèles sont comparés nuit par nuit

**Décision.** Plutot que de comparer deux RMSE globales, je calcule la RMSE de
chaque modèle sur chacune des nuits de la période de test, puis je compte les
nuits gagnées et je regarde l'écart moyen face à sa propre dispersion.

**Pourquoi.** Deux modèles séparés par 0,01 °C ne sont pas départagés, et il faut
pouvoir le dire. Le regroupement par nuit est le bon niveau : les heures d'une
même nuit se ressemblent beaucoup, les traiter comme indépendantes donnerait une
fausse impression de précision.

### 5.3 Le cout des erreurs est asymétrique, et le seuil d'alerte en découle

**Décision.** Une nuit de gel non détectée coute 4 000 € par hectare, une alerte
inutile 120 € : un rapport de 33. Le seuil d'alerte est celui qui minimise le
cout total, soit environ +2,5 °C sur le minimum nocturne prédit.

**Pourquoi.** C'est le résultat le plus important du projet, et il ne vient pas
du modèle. Avec une asymétrie pareille, alerter à 0 °C est le plus mauvais
réglage possible : il maximise le nombre de gelées ratées, qui sont la seule
erreur chère. À +2,5 °C, la détection passe à 99 % et le cout des erreurs est
divisé par cinq, au prix de deux alertes sur trois inutiles.

**Comment je défends les chiffres.** Je ne les défends pas, et c'est le point :
le seuil optimal ne dépend que du **rapport** entre les deux couts, pas des
montants. Le rapport donne la table pour des rapports de 5 à 100, de sorte qu'un
exploitant se situe sans avoir à chiffrer sa perte au centime. Un maraicher sous
abri et un viticulteur en floraison n'ont pas le même arbitrage.

### 5.4 L'erreur est déclinée par régime, par mois, par heure et par station

**Décision.** Aucune métrique n'est publiée sans son découpage.

**Pourquoi.** Une RMSE globale de 1,2 °C n'est pas la précision du produit la
nuit où il sert : l'erreur est nettement plus forte dans le froid, c'est-à-dire
dans les situations qui déclenchent une alerte. Et la dispersion entre stations
est importante : quelques dizaines de capteurs concentrent les mauvais résultats.

Une précaution de lecture figure dans le rapport : après le filtre sur le
rayonnement, les heures de milieu de journée ne subsistent que par temps très
couvert, avec un millier d'observations contre 150 000 pour les heures de nuit.
Leur erreur plus faible décrit un régime particulier, pas une meilleure
performance.

### 5.5 Deux expériences de controle, pour chiffrer des pièges

**Décision.** Le projet mesure explicitement ce que coutent un découpage
aléatoire, une cible erronée et une évaluation circulaire.

**Pourquoi.** Ce sont les trois erreurs de ma première version. Aucune ne
provoque de plantage : le code tourne, les scores sont flatteurs, la conclusion
est fausse. Les chiffrer est la seule façon de montrer qu'on les a comprises, et
pas seulement corrigées.

---

## 6. Outillage

### 6.1 Un paquet Python et une interface en ligne de commande, pas un notebook

**Décision.** Le code vit dans `src/gel/`, piloté par `python -m gel <etape>`. Le
notebook ne sert qu'à présenter le raisonnement.

**Pourquoi.** Dans ma première version, le code était dupliqué entre deux
notebooks, avec des chemins absolus vers mon bureau. Rien n'était testable ni
rejouable. Ici chaque étape écrit ses sorties et peut être relancée seule : on
refait la modélisation en trois minutes sans refaire le nettoyage.

### 6.2 Toute valeur discutable est dans `config/config.toml`

**Décision.** Chemins, seuils de nettoyage, bornes du découpage, grilles
d'hyperparamètres, hypothèses de cout. Format TOML, lu par `tomllib`, qui est
dans la bibliothèque standard depuis Python 3.11.

**Pourquoi.** Tout ce qui pourrait être rediscuté avec le métier doit être
visible sans ouvrir le code. Et les hypothèses de cout, en particulier, sont
faites pour être changées par quelqu'un qui n'est pas développeur.

### 6.3 Les tests portent sur les erreurs que j'ai réellement commises

**Décision.** 42 tests, dont sept qui encodent une erreur précise de la première
version : le filtre journalier appliqué à l'heure, le site usine testé par
égalité, la MSE confondue avec la RMSE, la cible erronée, le cout compté à
l'heure, la fuite de la mesure capteur en entrée, l'ordre des régimes de
température.

**Pourquoi.** Un test qui ne protège de rien est du décor. Ceux-là protègent de
fautes qui sont déjà arrivées, et qui ne produisaient aucun message d'erreur.

### 6.4 Ce que je n'ai volontairement pas fait

- **Pas de MLflow, pas de DVC, pas de Docker, pas d'API.** Ce projet n'est pas un
  projet d'industrialisation : il tient en un fichier de configuration, une
  commande et quatre minutes d'exécution. Ajouter une couche de suivi
  d'expériences pour trois modèles qui tiennent dans un tableau serait de
  l'affichage. J'ai traité ces sujets ailleurs, sur des projets où ils se
  justifient.
- **Pas de recherche d'hyperparamètres étendue.** Trois configurations de
  gradient boosting, qui se tiennent dans un centième de degré. Pousser plus loin
  aurait consommé du temps sur un problème dont le plafond vient des données.
- **Pas de modèle par station.** C'est la piste d'amélioration que je classe en
  premier, mais elle change la nature du produit : il faut un historique par
  capteur, donc on ne sert plus une parcelle sans matériel.

---

## 7. Ce que je ferais différemment avec plus de temps

1. **Calibrer la formule par station.** Les écarts d'erreur entre capteurs sont
   importants, et une calibration locale les absorberait probablement en partie.
2. **Prédire directement le minimum de la nuit**, puisque c'est la grandeur de
   décision, au lieu de le reconstituer comme minimum de prédictions horaires.
3. **Ajouter la dynamique temporelle** (tendance des trois dernières heures,
   refroidissement depuis le coucher du soleil). L'énoncé l'interdisait ; en
   production ces variables existent, et la physique du refroidissement radiatif
   suggère qu'elles portent du signal.
4. **Suivre la dérive en exploitation**, avec l'erreur par station recalculée en
   continu. Un capteur qui dérive ressemble beaucoup à un modèle qui se dégrade,
   et les deux se détectent de la même façon.
