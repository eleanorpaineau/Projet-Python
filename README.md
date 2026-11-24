# Projet-Python

# Sujet : Les récessions sont-elles bonnes pour la santé?
## Chômage et mortalité dans les pays de l'OCDE entre 1991 et 2023, une réplication du modèle de Gerdtham et Ruhm.

*Auteurs :  Marie Dominge, Guillaume Gourtay et Eleanor Paineau*

**Notre problématique pour ce projet est : Comment la mortalité réagit-elle aux ralentissements économiques ?**

Pour répondre à cette question, nous nous sommes appuyés sur l'article de 2006 *Deaths rise in good economic times: Evidence from the OECD* de Gerdtham et Ruhm qui s'intéresse au lien entre taux de chômage et mortalité avec des données entre 1960 et 1997 pour 23 pays de l'OCDE.

Nous avons étendu l'études aux actuels 38 pays de l'OCDE, et avons poursuivi l'étude sur la période 1991 à 2023, particulièrement intéressante parce que marquée par plusieurs grands chocs économiques (éclatement bulle internet en 2000, crise des subprimes en 2007, crise de la dette souveraine de la zone euro en 2012, pandémie de Covid-19 en 2020).

## Données

Pour notre étude, nos données proviennent de deux sources :
- Les données économiques et démographique (taux de chômage, part des plus de 65 ans dans la population...) provient de la World Bank (Banque Mondiale), collectée par API via wbgapi.
- Les données relatives à la mortalité viennent de la base de données Global Burden of Disease (GBD) de l'Institute for Health Metrics and Evaluation.

*Précision:* Les données de la Banque mondiale sont téléchargées automatiquement au premier lancement, puis mises en cache dans data_WB/wb_brut.parquet

## Principaux résultats
| Cause de décès | Effet d'un point de chômage | Significativité |
|:---|---:|:---:|
| Accidents de la route | **−0,96 %** | \*\*\* |
| Autres accidents | **−0,85 %** | \*\*\* |
| Maladies cardiovasculaires | **−0,45 %** | \*\*\* |
| Suicide | **+1,02 %** | \*\*\* |
| Toutes causes | −0,16 % | / |

<sub>\*\*\* significatif à 1 %, \*\* à 5 %, \* à 10 %. Écarts-types clusterisés par pays.</sub>

- La mortalité est globalement procyclique : elle baisse quand le chômage augmente, comme chez Gerdtham et Ruhm.
- Le suicide fait exception : il augmente avec le chômage. Mais l'effet varie fortement d'un pays à l'autre : il est quasi nul en France et en Finlande, et nettement positif dans les pays baltes, en Corée et au Japon.

## Structure du dépôt

Projet-Python/
- `README.md` : présentation du projet
- `main.ipynb` : notebook principal, **à lire en premier**
- `requirements.txt` : liste des dépendances
- `data_WB/` : cache des données Banque mondiale (généré automatiquement)
- `modules/`
  - `Fonctions.py` : fonctions de régression et de graphiques



## Références

- Global Burden of Disease Collaborative Network.
Global Burden of Disease Study 2023 (GBD 2023) Results. 
Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. 
Available from https://vizhub.healthdata.org/gbd-results/.
- Gerdtham, U.G. and Ruhm, C.J., 2006. Deaths rise in good economic times: evidence from the OECD. Economics & Human Biology, 4(3), pp.298-316.
- Ruhm, C.J., 2015. Recessions, healthy no more?. Journal of health economics, 42, pp.17-28.
- International Labour Organization (2026) Unemployment, total (% of total labor force) (modeled ILO estimate) [SL.UEM.TOTL.ZS]. ILOSTAT database. Repris dans : World Bank, World Development Indicators. Disponible à : https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS (consulté le 27 octobre 2026)
- World Bank (2026) GDP per capita, PPP (constant 2021 international $) [NY.GDP.PCAP.PP.KD]. World Development Indicators. Disponible à : https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD (consulté le 27 octobre 2026)