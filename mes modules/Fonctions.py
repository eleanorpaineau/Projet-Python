import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

np.random.seed(123)

### Constantes

metrics = ["Number", "Percent", "Rate"]
colonness_GBD = ["location_id", "location_name", "cause_id", "cause_name", "metric_id", "metric_name", "year", "value","upper_bound", "lower_bound"]


##### Fonctions IHME

# fonction de création d'un dataframe qui ne comporte que les colonnes utiles pour l'analyse et qui convertit les colonnes value, upper_bound et lower_bound en numériques

def passage_en_numerique(df_GBD):
    df_GBD_modifie = df_GBD[["location_id", "location_name", "cause_id", "cause_name", "metric_id", "metric_name", "year", "value","upper_bound", "lower_bound"]]

    # conversion du texte en valeurs pour value, upper_bound, lower_bound
    df_GBD_modifie["value"] = pd.to_numeric(df_GBD_modifie["value"])
    df_GBD_modifie["upper_bound"] = pd.to_numeric(df_GBD_modifie["upper_bound"])
    df_GBD_modifie["lower_bound"] = pd.to_numeric(df_GBD_modifie["lower_bound"])
    
    # ajout des écarts-types et variances pour après calculer les nouvelles upper_bound et lower_bound
    df_GBD_modifie["ecart_type"] = (df_GBD_modifie["upper_bound"] - df_GBD_modifie["lower_bound"]) / (2 * 1.96)
    df_GBD_modifie["var"] = df_GBD_modifie["ecart_type"]**2
    return df_GBD_modifie

# fonction qui prend en argument un dataframe, la liste des causes associées à une cause générale et renvoie une liste de trois dataframe (un pour chaque métrique) qui contient le pays, l'année, les valeurs aggrégées pour cette cause générale et la métrique

def dataframe_par_metrique(df, cause, metric) :
    L = [] # initialisation de la liste
    for i in range(1,4) : # pour chaque métrique
        df_filtre_i = df[(df["cause_id"].isin(cause)) & (df["metric_id"] == i)] # filtration
        df_aggreg_i = df_filtre_i.groupby(["location_id","location_name", "year"], as_index=False)["value"].sum() # aggrégation
        df_aggreg_i = df_aggreg_i.assign(metric_id = i, metric_name = metric[i-1] ) # ajout du nom et le numéro de la métrique  
        L.append(df_aggreg_i)
    return(L)