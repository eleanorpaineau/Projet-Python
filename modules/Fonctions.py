import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import wbgapi as wb
import pandas as pd
from pathlib import Path
from functools import reduce

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


def taux_croissance_dataframe_WB(df, indicateurs_vers_croissance, group_by): #ajoute les colonnes de taux de croissance pour les indicateurs cités dans indicateurs_vers_croissance dans le dataframe
    df.sort_values(by=[group_by, "year"], inplace=True) # tri du dataframe par pays et par année
    for col, croissance_col in indicateurs_vers_croissance.items():
            df[croissance_col] = df.groupby(group_by)[col].pct_change() * 100
    return(df)

########################## Fonctions pour les figures


#Fonction qui permet de créer un sous dataframe à partir d'un datafram original, d'une métrique, d'une cause et d'un pays
#Cela permet de ne garder que l'évolution d'une cause précise pour un pays et une métrique donnés


def sous_dataframe(df, metric_id, cause_id, location_id) :
    df_sous = df[(df["metric_id"] == metric_id) & (df["cause_id"] == cause_id) & (df["location_id"] == location_id)]
    return(df_sous)

#fonction qui dessine le tabeleau ci-dessus
def plot_dataframe(df, metric_id, cause_id, location_id) :
    df_sous = sous_dataframe(df, metric_id, cause_id, location_id)
    plt.figure()
    plt.plot(df_sous["year"], df_sous["value"], marker='o', label='Value')
    plt.fill_between(df_sous["year"], df_sous["lower_bound"], df_sous["upper_bound"], color='gray', alpha=0.2, label='Standard error')
    plt.title(f"Evolution of {df_sous['cause_name'].iloc[0]} in {df_sous['location_name'].iloc[0]} ({df_sous['metric_name'].iloc[0]})")
    plt.xlabel("Time (Year)")
    plt.ylabel("Value (number of deaths per 100,000 population)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1),
          ncol=1, fancybox=True, shadow=True)    
    plt.grid()
    plt.show()
    
#Fonction qui plot sur un même graphique l'évolution d'une cause pour N pays dont l'id_location est dans une liste
def plot_dataframe_multiple_countries(df, metric_id, cause_id, location_ids):
    N=len(location_ids)
    plt.figure()
    for location_id in location_ids:
         df_sous = sous_dataframe(df, metric_id, cause_id, location_id)
         plt.plot(df_sous["year"], df_sous["value"], marker='o', label=f"{df_sous['location_name'].iloc[0]}")
         plt.fill_between(df_sous["year"], df_sous["lower_bound"], df_sous["upper_bound"], color='gray', alpha=0.2)
    plt.title(f"Evolution of {df_sous['cause_name'].iloc[0]} ({df_sous['metric_name'].iloc[0]})")
    plt.xlabel("Time (Year)")
    plt.ylabel("Value (number of deaths per 100,000 population)")
    plt.legend(loc='upper right', bbox_to_anchor=(1+0.4*ceil(N/14), 1),
          ncol=ceil(N/14), fancybox=True, shadow=True)
    plt.grid()
    plt.show()
  
# Fonction qui plot sur un même graphique l'évolution de N causes pour un pays dont l'id_location est donnée   
def plot_dataframe_multiple_causes(df, metric_id, cause_ids, location_id):
    plt.figure()
    for cause_id in cause_ids:
        df_sous = sous_dataframe(df, metric_id, cause_id, location_id)
        plt.plot(df_sous["year"], df_sous["value"], marker='o', label=f"{df_sous['cause_name'].iloc[0]}")
        plt.fill_between(df_sous["year"], df_sous["lower_bound"], df_sous["upper_bound"], color='gray', alpha=0.2)
    plt.title(f"Evolution of causes in {df_sous['location_name'].iloc[0]} ({df_sous['metric_name'].iloc[0]})")
    plt.xlabel("Time (Year)")
    plt.ylabel("Value (number of deaths per 100,000 population)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1),
          ncol=1, fancybox=True, shadow=True)
    plt.grid()
    plt.show()