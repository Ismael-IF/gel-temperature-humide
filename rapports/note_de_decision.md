# Note de décision : étendre l'alerte gel aux parcelles sans capteur

Note courte à destination d'une équipe produit ou d'un responsable
d'exploitation. Les chiffres détaillés et la méthode sont dans
[resultats.md](resultats.md).

## La question posée

Le risque de gel se lit sur la température humide mesurée à hauteur de végétal,
sans abri, vers 50 cm. C'est cette grandeur qui dit si un bourgeon va geler, et
c'est elle qui décide si une protection par aspersion protège la parcelle ou
l'endommage. Elle demande un capteur dédié, physiquement installé dans la
parcelle.

Les variables atmosphériques à 2 m sous abri (température, point de rosée,
humidité, vent, nébulosité, rayonnement, précipitations), elles, sont
disponibles partout, par estimation. D'où la question : peut-on reconstituer la
température humide à 50 cm à partir de ces seules variables, et donc servir une
alerte gel là où il n'y a pas de capteur ?

La réponse est oui, avec une précision connue et des limites précises.

## Ce que le modèle sait faire

Sur la saison de test (printemps 2023, jamais vue à l'apprentissage, 1 409
stations, 46 081 nuits de station) :

- erreur typique de **1,2 °C** sur la température humide horaire ;
- l'erreur dépasse 1 °C une heure sur trois et 2 °C une heure sur dix ;
- à la maille qui compte, la nuit, le modèle **détecte 71 % des nuits de gel**
  avec 20 % d'alertes injustifiées, si l'on déclenche l'alerte à 0 °C ;
- la performance ne se dégrade pas sur les capteurs installés après la période
  d'apprentissage, ce qui est l'indication la plus directe que le modèle est
  transposable à une parcelle nouvelle.

Pour situer : prédire la température humide par la température sous abri, ce qui
est l'approximation implicite quand on n'a pas de capteur, ne détecte que 31 %
des nuits de gel. Le modèle multiplie donc par plus de deux le taux de détection
sans aucun matériel supplémentaire.

## Ce que le modèle ne sait pas faire

- **Il n'est pas un substitut au capteur sur les parcelles à enjeu.** Une erreur
  de 1,2 °C est du même ordre que la marge de décision agronomique. Sur une
  parcelle à forte valeur en période sensible, le capteur reste nécessaire.
- **Il est moins précis dans le froid**, c'est-à-dire là où il sert. L'erreur
  croit nettement sous -1 °C. La précision annoncée en moyenne n'est pas la
  précision la nuit de l'alerte.
- **Il ne prévoit pas l'avenir.** Il reconstitue la température humide à partir
  de variables atmosphériques de la même heure. Branché sur une prévision
  météorologique, il hérite de l'erreur de cette prévision, qui s'ajoute à la
  sienne.
- **Il n'a pas vu d'évènement extrême.** Trois saisons de données, et un
  printemps de test. La gelée noire exceptionnelle n'est pas dans
  l'échantillon.

## La décision à prendre : où placer le seuil d'alerte

C'est le point le plus important de cette note, et ce n'est pas une question de
modèle.

Rater une gelée coute une récolte. Déclencher pour rien coute une nuit de
mobilisation. Les deux ne sont pas du même ordre : avec les hypothèses retenues
ici (4 000 € par hectare pour une gelée non détectée, 120 € pour une alerte
inutile, soit un rapport de 33), le seuil qui minimise le cout total n'est pas
0 °C mais **environ +2,5 °C sur le minimum nocturne prédit**.

À ce réglage, le modèle détecte **99 % des nuits de gel**, au prix de deux
alertes sur trois qui s'avèrent inutiles, et le cout total des erreurs baisse
d'environ 80 % par rapport à un seuil à 0 °C.

Ce chiffre de +2,5 °C n'est pas une vérité : il dépend du rapport entre les deux
couts, et de lui seul. Le tableau de sensibilité dans
[resultats.md](resultats.md) donne le seuil pour des rapports de 5 à 100, ce qui
permet à chaque exploitant de se situer sans avoir à chiffrer sa perte au
centime. Un maraicher sous abri et un viticulteur en pleine floraison n'ont pas
le même arbitrage, et le produit devrait leur laisser ce réglage.

Deux conséquences pratiques :

1. **Un seuil unique à 0 °C est le plus mauvais choix possible.** Il maximise le
   nombre de gelées ratées, qui sont la seule erreur réellement couteuse.
2. **Le taux de fausses alertes doit être assumé explicitement**, pas subi. Deux
   alertes inutiles sur trois est acceptable si l'alerte est bon marché à
   traiter, inacceptable si elle réveille quelqu'un à trois heures du matin. Cela
   plaide pour une alerte graduée (vigilance, puis déclenchement) plutot que
   binaire.

## Recommandation de mise en oeuvre

**Déployer la formule recalibrée, pas le modèle d'apprentissage automatique.**

Les trois approches comparées se tiennent dans deux centièmes de degré, et
l'écart n'est pas statistiquement départageable entre les deux meilleures. Dans
cette situation, le critère de choix n'est plus la performance mais le cout de
possession. Une formule à quatre coefficients :

- se calcule en quelques opérations arithmétiques, y compris sur un
  microcontrolleur de station, donc sans aller-retour serveur ;
- se vérifie à la main par un agronome, ce qui compte quand une alerte est
  contestée ;
- se recalibre en une seconde sur de nouvelles données, et par station si besoin ;
- ne dépend d'aucune bibliothèque de machine learning à maintenir en production.

Le gradient boosting garde un intérêt : il sert de référence pour savoir si la
formule laisse du signal sur la table. Sur ce jeu, il n'en laisse pas, et c'est
une information qui vaut d'être réutilisée à chaque nouvelle saison de données.

## Ce que je mesurerais ensuite

Par ordre de valeur attendue :

1. **Calibrer par station**, avec les quatre coefficients de la formule
   réestimés sur l'historique de chaque capteur. Les écarts d'erreur entre
   stations sont importants : quelques dizaines de stations concentrent les
   mauvais résultats, et c'est probablement un effet d'exposition ou de
   positionnement du capteur, qu'une calibration locale absorberait.
2. **Prédire le minimum de la nuit directement**, puisque c'est la grandeur de
   décision, au lieu de l'obtenir comme minimum de prédictions horaires.
3. **Ajouter la dynamique temporelle** (tendance des trois dernières heures,
   refroidissement depuis le coucher du soleil). L'énoncé interdisait ces
   variables ; en production, elles sont disponibles et la physique du
   refroidissement radiatif suggère qu'elles portent du signal.
4. **Suivre la dérive en exploitation**, avec l'erreur par station et par mois
   recalculée en continu, et une alerte quand une station décroche. Un capteur
   qui dérive ressemble beaucoup à un modèle qui se dégrade.
