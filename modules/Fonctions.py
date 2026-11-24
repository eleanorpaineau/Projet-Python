import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import pandas as pd
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
import seaborn as sns

### Constantes

np.random.seed(123)
metrics = ["Number", "Percent", "Rate"]
colonness_GBD = ["location_id", "location_name", "cause_id", "cause_name", "metric_id", "metric_name", "year", "value","upper_bound", "lower_bound"]
causes_generales = [["All causes", 1, [294]],["Malignant neoplasm", 2, [1012, 459, 489]], ["Major cardiovascular diseases", 3,[491]], ["Liver diseases, cirrhosis", 4, [521]], ["Pneumonia and influenza", 5, [322]], ["Motor vehicle accidents", 6, [689, 688]],["Suicide", 7, [718]], ["Other accidents", 8, [696]], ["Homicide", 9, [724, 945,854]]]
numero_agg = [1012, 459, 489, 689, 688, 724, 945, 854] # numeros des causes aggrégées
numero_non_agg = [294,491,521,322,718,696] # numeros des causes non aggrégées 

countries = ["AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA",
             "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA"]

indicateurs_vers_croissance = {
    "Taux_chômage": "Croissance_taux_chômage",
    "Taux_chômage_F": "Croissance_chômage_F",
    "Taux_chômage_H": "Croissance_chômage_H",
    "PIB/hab_ppa": "Croissance_PIB_hab_ppa",
    "Population": "Croissance_population",
}

code_vers_label = {
    "SL.UEM.TOTL.ZS": "Taux_chômage",
    "SL.UEM.TOTL.FE.ZS": "Taux_chômage_F",
    "SL.UEM.TOTL.MA.ZS": "Taux_chômage_H",
    "NY.GDP.PCAP.PP.KD": "PIB/hab_ppa",
    "SP.POP.TOTL": "Population",
    "SP.POP.65UP.TO.ZS": "Pourcentage_pop_65plus",
    "SH.XPD.GHED.GD.ZS": "Pourcentage_PIB_dépenses_santé"
}
#Création d'un tableau qui associe les codes pays à leurs noms complets en anglais
noms_pays = pd.DataFrame({
    "economy": countries,

    "location_name": ["Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Costa Rica",
    "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan", "South Korea",
    "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands", "New Zealand",
    "Norway", "Poland", "Portugal", "Slovakia", "Slovenia", "Spain", "Sweden",
    "Switzerland", "Turkey", "United Kingdom", "United States"]
})

couleurs = [ "blue","orange","green","red","purple","brown","pink", "gray","olive"]


################# Fonctions de formatage des tableaux 

##### Fonctions pour les deux datafrmes 

# Fonction qui prend en argument un dataframe et le nom de la colonne des pays et renvoie le même dataframe avec le numéro d'identité du pays correspondant à chaque pays par ordre alphabétique.
def ajout_numero_pays(df, nom_colonne_pays, nom_colonne_id_pays) :
    df_court = df.drop(columns=[nom_colonne_id_pays])
    df_pays = df_court[[nom_colonne_pays]].drop_duplicates().sort_values(by=nom_colonne_pays).reset_index(drop=True) # dataframe avec une occurence de chaucun des pays triés par ordre alphabétique
    df_pays[nom_colonne_id_pays] = df_pays.index + 1 # ajout du numéro d'identité du pays
    df = df_court.merge(df_pays, on=nom_colonne_pays, how="left") # fusion des deux dataframes
    return df

##### Fonctions IHME

# fonction de création d'un dataframe qui ne comporte que les colonnes utiles pour l'analyse et qui convertit les colonnes value, upper_bound et lower_bound en numériques

def passage_en_numerique(df_GBD):
    df_GBD_modifie = df_GBD[["location_id", "location_name", "cause_id", "cause_name", "metric_id", "metric_name", "year", "value","upper_bound", "lower_bound"]]

    # conversion du texte en valeurs pour value, upper_bound, lower_bound
    df_GBD_modifie["value"] = pd.to_numeric(df_GBD_modifie["value"])
    df_GBD_modifie["upper_bound"] = pd.to_numeric(df_GBD_modifie["upper_bound"])
    df_GBD_modifie["lower_bound"] = pd.to_numeric(df_GBD_modifie["lower_bound"])
    
    if df_GBD_modifie["cause_id"].isin(numero_agg).any() : # si la cause est une cause aggrégée
        # ajout des écarts-types et variances pour après calculer les nouvelles upper_bound et lower_bound
        df_GBD_modifie["ecart_type"] = (df_GBD_modifie["upper_bound"] - df_GBD_modifie["lower_bound"]) / (2 * 1.96)
        df_GBD_modifie["var"] = df_GBD_modifie["ecart_type"]**2
    return df_GBD_modifie

# fonction qui prend en argument un dataframe, la liste de listes des causes associées aux causes générales et renvoie un dataframe qui contient le pays, l'année,la métrique, les valeurs aggrégées, et les lower_bound et upper_bound.
# pour le calcul de l'intervalle de confiance : méthode des variances
# - on calcule l'écart type (SE) aggrégé pour chaque cause générale : il s'agit de la racine de la somme des variances de chaque cause 
# - l'intervale de confiance correspond à la somme des estimations ± 1,96 × SE
# - on obtient alors : upperbound = value + 1,96 × SE    &    lowerbound = value - 1,96 × SE 

def transformations_dataframe_GBD(df, causes_generales, metric) :
    L = [] # initialisation de la liste
    for j in range(len(causes_generales)) : # pour chaque cause générale
        cause = causes_generales[j] # liste de causes associées 
        for i in range(len(metric)) : # pour chaque métrique
            if len(cause[2])>1 : # si la cause est une cause aggrégée
                df_filtre_i = df[(df["cause_id"].isin(cause[2])) & (df["metric_id"] == i+1)] # filtration
                df_aggreg_i = df_filtre_i.groupby(["location_id","location_name", "year"], as_index=False)["value"].sum() # aggrégation
                df_aggreg_i = df_aggreg_i.assign(metric_id = i+1, metric_name = metric[i] ) # ajout du nom et le numéro de la métrique
                df_aggreg_i = df_aggreg_i.assign(cause_id = cause[1], cause_name = cause[0])
                L.append(df_aggreg_i)
                df_aggreg_var_i = df_filtre_i.groupby(["location_id","location_name", "year"], as_index=False)["var"].sum() # somme des variances
                df_aggreg_var_i["ecart_type"] = np.sqrt(df_aggreg_var_i["var"]) # écart-type
                M = 1.96*df_aggreg_var_i["ecart_type"]

                L[3*j+i]["lower_bound"] = L[3*j+i]["value"] - M # calcul du lower_bound
                L[3*j+i]["upper_bound"] = L[3*j+i]["value"] + M # calcul du upper_bound
            elif len(cause[2]) == 1 : # si la cause est une cause individuelle
                df_filtre_i = df[(df["cause_id"].isin(cause[2])) & (df["metric_id"] == i+1)] # filtration
                df_filtre_i = df_filtre_i.assign(cause_id = cause[1], cause_name = cause[0])
                L.append(df_filtre_i)
    return(pd.concat(L, ignore_index=True)) # renvoie le dataframe concaténé

#fonction qui prend un dataframe en argument, une année et une métrique et renvoie ce dataframe pour les années au dessus de l'année et pour la métrique choisie.
def filtrage_dataframe(df, annee, metric_id) :
    df_filtre = df[(df["year"] >= annee) & (df["metric_id"] == metric_id)].copy() # filtration du dataframe
    return df_filtre

##### Fonctions API World Bank

def en_long_WB(df):
    df_reset = df.reset_index() # On transforme les index en colonnes
    df_long = df_reset.melt(id_vars=["economy", "series"], var_name="year", value_name="valeur")     # On met les colonnes d'années en une seule colonne 'year'
    df_long["year"] = df_long["year"].astype(str).str.replace("YR", "").astype(int) # On enlève YR
    df_final = df_long.pivot(index=["economy", "year"], columns="series", values="valeur").reset_index() # On répartit les indicateurs en colonnes
    return df_final

def labelisation_dataframe_WB(df, noms_pays, nom_colonne_pays, nom_colonne_id_pays, indicateurs):
    df_deux_noms = df.merge(noms_pays, on="economy", how="left") # fusion du dataframe de nomes de pays avec le df WB
    df_un_nom = df_deux_noms.drop(columns=["economy"]) # renommage de la colonne économie en location_id
    df_pays = df_un_nom[[nom_colonne_pays]].drop_duplicates().sort_values(by=nom_colonne_pays).reset_index(drop=True) # dataframe avec une occurence de chaucun des pays triés par ordre alphabétique
    df_pays[nom_colonne_id_pays] = df_pays.index + 1 # ajout du numéro d'identité du pays
    df = df_un_nom.merge(df_pays, on=nom_colonne_pays, how="right") # fusion des deux dataframes
    df_renomme=df.rename(columns=indicateurs) # renommage des colonnes
    return df_renomme

########################## Fonctions pour les figures

##### Fonctions pour les figures du dataframe de GBD

#Fonction qui permet de créer un sous dataframe à partir d'un dataframe original, d'une cause et d'un pays
#Cela permet de ne garder que l'évolution d'une cause précise pour un pays donné
def sous_dataframe_GBD(df, cause_id, location_id) :
    df_sous = df[(df["cause_id"] == cause_id) & (df["location_id"] == location_id)]
    return(df_sous)

#fonction qui dessine le tabeleau ci-dessus
def plot_dataframe(df, cause_id, location_id) :
    df_sous = sous_dataframe_GBD(df, cause_id, location_id)
    plt.figure()
    plt.plot(df_sous["year"], df_sous["taux_mortalite"], marker='o')
    plt.fill_between(df_sous["year"], df_sous["lower_bound"], df_sous["upper_bound"], color='gray', alpha=0.2, label='Standard error')
    plt.title(f"Evolution of mortality rate of {df_sous['cause_name'].iloc[0]} in {df_sous['location_name'].iloc[0]}")
    plt.xlabel("Time (Year)")
    plt.ylabel("Mortality Rate (number of deaths per 100,000 population)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1),
          ncol=1, fancybox=True, shadow=True)    
    plt.grid()
    plt.show()
    
#Fonction qui plot sur un même graphique l'évolution d'une cause pour N pays dont l'id_location est dans une liste
def plot_dataframe_multiple_countries_GBD(df, cause_id, location_ids):
    N=len(location_ids)
    plt.figure()
    for location_id in location_ids:
         df_sous = sous_dataframe_GBD(df, cause_id, location_id)
         plt.plot(df_sous["year"], df_sous["taux_mortalite"], marker='o', label=f"{df_sous['location_name'].iloc[0]}")
         plt.fill_between(df_sous["year"], df_sous["lower_bound"], df_sous["upper_bound"], color='gray', alpha=0.2)
    plt.title(f"Evolution of mortality rate of {df_sous['cause_name'].iloc[0]}")
    plt.xlabel("Time (Year)")
    plt.ylabel("Mortality Rate (number of deaths per 100,000 population)")
    plt.legend(loc='upper right', bbox_to_anchor=(1+0.4*ceil(N/14), 1),
          ncol=ceil(N/14), fancybox=True, shadow=True)
    plt.grid()
    plt.show()
  
# Fonction qui plot sur un même graphique l'évolution de N causes pour un pays dont l'id_location est donnée   
def plot_dataframe_multiple_causes_GBD(df, cause_ids, location_id):
    plt.figure()
    for cause_id in cause_ids:
        df_sous = sous_dataframe_GBD(df, cause_id, location_id)
        plt.plot(df_sous["year"], df_sous["taux_mortalite"], marker='o', label=f"{df_sous['cause_name'].iloc[0]}")
        plt.fill_between(df_sous["year"], df_sous["lower_bound"], df_sous["upper_bound"], color='gray', alpha=0.2)
    plt.title(f"Evolution of mortality rates of death causes in {df_sous['location_name'].iloc[0]}")
    plt.xlabel("Time (Year)")
    plt.ylabel("Mortality Rate (number of deaths per 100,000 population)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1),
          ncol=1, fancybox=True, shadow=True)
    plt.grid()
    plt.show()


##### Fonctions pour les figures du dataframe de World Bank

#Fonction qui permet de plot un sous dataframe à partir d'un dataframe original d'un pays et d'une mesure démographique
#Cela permet de ne garder que l'évolution d'une mesure démographique précise pour un pays donné
def sous_dataframe_WB(df, mesure, location_id) :
    if not isinstance(mesure, str):
        mesure=str(mesure)
    df_sous = df[(df["location_id"] == location_id)]
    return( df_sous[["location_id", "location_name", "year", mesure]])

def plot_dataframe_WB(df, mesure, location_id):
    df_mesure = sous_dataframe_WB(df, mesure, location_id)
    plt.figure()
    plt.plot(df_mesure["year"], df_mesure[mesure], marker='o', label='Value')
    plt.title(f"Evolution du {mesure} en {df_mesure['location_name'].iloc[0]}")
    plt.xlabel("Time (Year)")
    plt.ylabel(f"{mesure}")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1),
          ncol=1, fancybox=True, shadow=True)    
    plt.grid()
    plt.show()

def plot_dataframe_multiple_countries_WB(df, mesure, location_ids):
    N=len(location_ids)
    plt.figure()
    for location_id in location_ids:
         df_mesure = sous_dataframe_WB(df, mesure, location_id)
         plt.plot(df_mesure["year"], df_mesure[mesure], marker='o', label=f"{df_mesure['location_name'].iloc[0]}")
    plt.title(f"Evolution du {mesure}")
    plt.xlabel("Time (Year)")
    plt.ylabel(f"{mesure}")
    plt.legend(loc='upper right', bbox_to_anchor=(1+0.4*ceil(N/14), 1), ncol=ceil(N/14), fancybox=True, shadow=True)
    plt.grid()
    plt.show()


##### Fonctions pour les figures du dataframe final

def sous_dataframe_final(df, mesure, cause_id, location_id) :
    if not isinstance(mesure, str):
        mesure=str(mesure)
    df_sous = df[(df["cause_id"] == cause_id) & (df["location_id"] == location_id)]
    return(df_sous[["location_id", "location_name", "year", "cause_id", "cause_name", "taux_mortalite", mesure]])

def plot_mortalite_demographie_pays(df, mesure, cause_id, location_ids) : 
    plt.figure()
    N=len(location_ids)
    for location_id in location_ids:     
        # selection des informations utiles
        df_sous = sous_dataframe_final(df, mesure, cause_id, location_id)
        df_reg = pd.DataFrame({
            "y": pd.to_numeric(df_sous["taux_mortalite"], errors="coerce"),
            "x": pd.to_numeric(df_sous[mesure], errors="coerce"),
        })

        # Calcul de la régression
        modele = smf.ols('y ~ x', data=df_reg).fit()
        xs = np.linspace(df_reg["x"].min(), df_reg["x"].max(), 100)
        a = modele.params["x"]
        b = modele.params["Intercept"]
        signe = "+" if b >= 0 else "-"
        equation = f"y = {a:.3g}x {signe} {abs(b):.3g}"
        pred = a * xs + b

        # affichage
        plt.scatter(df_sous[mesure], df_sous["taux_mortalite"], alpha=0.7, label=f"{df_sous['location_name'].iloc[0]}")
        plt.plot(xs, pred,label=f"{df_sous['location_name'].iloc[0]} : {equation} (R² = {modele.rsquared:.2f})")

        plt.xlabel(f"{mesure}")
        plt.ylabel("Taux de mortalité (pour 100 000)")
        plt.title(f"{df_sous['cause_name'].iloc[0]} en fonction du {mesure}")
        plt.legend(loc='upper right', bbox_to_anchor=(1.5+0.4*ceil(N/14), 1), ncol=ceil(N/14), fancybox=True, shadow=True)
    plt.show()


################### Etude corrélations par cause et chômage
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def correlation_par_cause(df, x, y):
    df = df.dropna(subset=[x, y]).sort_values(["cause_name", "location_id", "year"]).copy()
    df_cause = df.groupby(["cause_name", "location_id"])

    df["variation_mortalite"] = df_cause[y].pct_change() * 100
    df["variation_chomage"] = df_cause[x].diff()

    # matrices 2x2 par cause, puis on garde la case (ligne y, colonne x)
    corr_niveaux = df.groupby("cause_name")[[x, y]].corr().xs(y, level=1)[x]
    corr_variation = (df.groupby("cause_name")[["variation_chomage", "variation_mortalite"]]
                        .corr().xs("variation_mortalite", level=1)["variation_chomage"])

    df_corr = pd.DataFrame({
        "corr_niveaux": corr_niveaux,
        "corr_variation": corr_variation,
    })
    return df_corr.round(2)


def regression(df, cause, x="Taux_chômage"):
    df_cause = df[df["cause_name"]==cause].dropna(subset=["taux_mortalite", x]).copy()
    df_cause["log_value"]=np.log(df_cause["taux_mortalite"])
    df_cause = df_cause.rename(columns={x: "chomage"})

    modele=smf.ols("log_value~chomage+C(location_id)+C(year)", data=df_cause).fit(cov_type="cluster", cov_kwds={"groups": df_cause["location_id"]})
    return modele.params["chomage"], modele.pvalues["chomage"]


def regression_ruhm(df, cause, x="Taux_chômage",avec_dep_sante=False, avec_pop65=False):
    if avec_dep_sante:
        df_cause = df[df["cause_name"] == cause].dropna(subset=["taux_mortalite", x, "Pourcentage_pop_65plus", "Population", "Pourcentage_PIB_dépenses_santé"]).copy()
    else:
        df_cause = df[df["cause_name"] == cause].dropna(subset=["taux_mortalite", x, "Pourcentage_pop_65plus", "Population"]).copy()
    df_cause["log_value"] = np.log(df_cause["taux_mortalite"])
    df_cause = df_cause.rename(columns={x: "chomage" })
    
    terme_sante = " + Pourcentage_PIB_dépenses_santé" if avec_dep_sante else ""
    terme_pop65= "+ Pourcentage_pop_65plus" if avec_pop65 else ""
    modele = smf.wls(f"log_value ~ chomage{terme_sante}{terme_pop65} + C(location_id) + C(year) + C(location_id):year",
        data=df_cause, weights=df_cause["Population"]).fit(cov_type="cluster", cov_kwds={"groups": df_cause["location_id"]})
    
    ligne = {"cause": cause}

    # On place les variables principales en premier pour une lecture directe
    preums = ["Intercept", "chomage"]
    if avec_dep_sante:
        preums.append("Pourcentage_PIB_dépenses_santé")
    if avec_pop65:
        preums.append("Pourcentage_pop_65plus")
    tous_params = preums + [p for p in modele.params.index if p not in preums]

    for param in tous_params:
        if param in modele.params:
            ligne[f"{param}_coef"] = modele.params[param]
            ligne[f"{param}_pval"] = modele.pvalues[param]

    return pd.DataFrame([ligne])

def graphe_ruhm(df, cause, location_ids, x="Taux_chômage"):
    df_cause = df[df["cause_id"] == cause].dropna(subset=["taux_mortalite", x, "Population"]).copy()
    df_cause["log_value"] = np.log(df_cause["taux_mortalite"])
    df_cause = df_cause.rename(columns={x: "chomage"})
    
    controles = "C(location_id) + C(year) + C(location_id):year"
    poids = df_cause["Population"]
    df_cause["res_y"] = smf.wls(f"log_value ~ {controles}", data=df_cause, weights=poids).fit().resid * 100
    df_cause["res_x"] = smf.wls(f"chomage ~ {controles}", data=df_cause, weights=poids).fit().resid
    cause_name =df_cause['cause_name'].iloc[0]
    # calcul des coefficients de régression
    res_reg = regression_ruhm(df, cause_name, x)
    coef, p = res_reg["chomage_coef"].iloc[0], res_reg["chomage_pval"].iloc[0]

    sel = df_cause[df_cause["location_id"].isin(location_ids)]
    fig, ax = plt.subplots(figsize=(10, 6))

    # représentation graphique
    for i, (loc_id, g) in enumerate(sel.groupby("location_id")):
        couleur = plt.cm.tab10(i % 10)
        nom = g["location_name"].iloc[0]
        a, b = np.polyfit(g["res_x"], g["res_y"], 1)
        xs = np.linspace(g["res_x"].min(), g["res_x"].max(), 50)   # droite sur l'étendue du pays

        ax.scatter(g["res_x"], g["res_y"], s=10, alpha=0.35, color=couleur)
        ax.plot(xs, a * xs + b, color=couleur, linewidth=2, label=f"{nom} ({a:+.2f} %)")

    # droite de Ruhm (tous pays)
    xs = np.linspace(sel["res_x"].min(), sel["res_x"].max(), 100)
    ax.plot(xs, coef * 100 * xs, color="black", linewidth=3, linestyle="--",
            label=f"Ruhm, tous pays ({coef * 100:+.2f} %, p = {p:.3f})")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Chômage : écart à la tendance (points)")
    ax.set_ylabel("Mortalité : écart à la tendance (%)")
    ax.set_title(f"{cause_name} : chômage et mortalité par pays")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    plt.show()


############ Histogrammes répartition moyenne

def _grille(n, ncols=3, hauteur_ligne=3.5, largeur=14):
    """Crée une grille de n graphiques et masque les cases vides."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(largeur, hauteur_ligne * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def histogrammes_moyennes(df, variables, groupe="location_name", ncols=3, bins=8, afficher=True):
    fig, axes = _grille(len(variables), ncols)         

    for ax, (col, titre) in zip(axes, variables.items()):
        moyennes = df.groupby(groupe)[col].mean().dropna()
        ax.hist(moyennes, bins=bins, edgecolor="white")
        ax.set_title(f"{titre}\n(moyenne par pays, n = {len(moyennes)})", fontsize=10)
        ax.set_ylabel("Nombre de pays")

    fig.tight_layout()
    if afficher:
        plt.show()


def evolutions_moyennes(df, variables, temps="year", ncols=3, afficher=True):
    fig, axes = _grille(len(variables), ncols)    

    for ax, (col, titre) in zip(axes, variables.items()):
        df.groupby(temps)[col].mean().plot(ax=ax)
        ax.set_title(f"{titre}\n(moyenne annuelle sur les pays)", fontsize=10)
        ax.set_xlabel("Année")
        ax.set_ylabel(titre)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if afficher:
        plt.show()


def evolutions_par_pays(df, variables, mis_en_avant, pays="location_name", temps="year", ncols=3, palette=plt.cm.tab10.colors, afficher=True):
    """
    Trace l'évolution de chaque variable pour tous les pays :
    les pays de `mis_en_avant` en couleur, les autres en gris.
    """
    presents = set(df[pays].unique())
    mis_en_avant = [p for p in mis_en_avant if p in presents]
    couleurs = dict(zip(mis_en_avant, palette))

    fig, axes = _grille(len(variables), ncols)

    for ax, (col, titre) in zip(axes, variables.items()):
        pivot = df.pivot(index=temps, columns=pays, values=col)

        pivot.drop(columns=mis_en_avant).plot(ax=ax, color="lightgrey",
                                              legend=False, linewidth=1)
        pivot[mis_en_avant].plot(ax=ax, color=[couleurs[p] for p in mis_en_avant],
                                 legend=False, linewidth=2)

        ax.set_title(titre, fontsize=10)
        ax.set_xlabel("Année")
        ax.set_ylabel(titre)
        ax.grid(alpha=0.3)

    handles = [Line2D([0], [0], color=couleurs[p], linewidth=2) for p in mis_en_avant]
    handles.append(Line2D([0], [0], color="lightgrey", linewidth=1))
    fig.legend(handles, mis_en_avant + ["Autres pays"], title="Pays",
               loc="lower center", ncol=len(handles))

    fig.tight_layout(rect=[0, 0.06, 1, 1])   # place en bas pour la légende
    if afficher:
        plt.show()
        
        
########## Matrice corrélation (spearman)
def matrice_correlation(df, variables, groupe="location_name"):
   
    moyennes = df.groupby(groupe)[variables].mean().dropna()
    # Calculate the correlation matrix
    correlation_matrix_std = moyennes.corr(method='spearman')

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_std, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Matrice de corrélation des variables démographiques moyennes (Spearman)')
    plt.show()
