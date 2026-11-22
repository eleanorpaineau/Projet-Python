import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt



## Fonction qui prend en argument un dataframe et une liste de pays et trace la mortalité pour 100 000 en fonction du taux de chômage pour chaque pays

def plot_mortalite_chomage_pays(df,liste_pays) : 
    plt.figure(figsize=(9, 6))
    couleurs = np.random.rand(len(liste_pays), 3)   # couleurs aléatoires pour chaque pays

    for i in range(len(liste_pays)):   # pour chaque pays
        
        # selection des informations utiles
        df_utile = df[(df["cause_name"] == "All causes") & (df["location_name"] == liste_pays[i])]
        df_utile = df_utile.dropna(subset=["value", "Taux_chômage"])

        # Calcul de la régression
        modele = smf.ols("value ~ Taux_chômage", data=df_utile).fit()
        xs = np.linspace(df_utile["Taux_chômage"].min(), df_utile["Taux_chômage"].max(), 100)
        pred = modele.predict({"Taux_chômage": xs})
        a = modele.params["Taux_chômage"]
        b = modele.params["Intercept"]
        signe = "+" if b >= 0 else "-"
        equation = f"y = {a:.3g}x {signe} {abs(b):.3g}"


        # affichage

        plt.scatter(df_utile["Taux_chômage"], df_utile["value"], alpha=0.7, color = couleurs[i])
        plt.plot(xs, pred, color=couleurs[i],label=f"{liste_pays[i]} : {equation} (R² = {modele.rsquared:.2f})")

    plt.xlabel("Taux de chômage (%)")
    plt.ylabel("Taux de mortalité (pour 100 000)")
    plt.title("Mortalité en fonction du chômage")
    plt.legend()
    plt.show()

##Fonction qui prend en argument un dataframe et une liste d'années et trace la mortalité pour 100 000 en fonction du taux de chômage pour chaque année

def plot_mortalite_chomage_annee(df,liste_annee) : 
    plt.figure(figsize=(9, 6))
    couleurs = np.random.rand(len(liste_annee), 3)   # couleurs aléatoires pour chaque année

    for i in range(len(liste_annee)):   # pour chaque année
        
        # selection des informations utiles
        df_utile = df[(df["cause_name"] == "All causes") & (df["year"] == liste_annee[i])]
        df_utile = df_utile.dropna(subset=["value", "Taux_chômage"])

        # Calcul de la régression
        modele = smf.ols("value ~ Taux_chômage", data=df_utile).fit()
        xs = np.linspace(df_utile["Taux_chômage"].min(), df_utile["Taux_chômage"].max(), 100)
        pred = modele.predict({"Taux_chômage": xs})
        a = modele.params["Taux_chômage"]
        b = modele.params["Intercept"]
        signe = "+" if b >= 0 else "-"
        equation = f"y = {a:.3g}x {signe} {abs(b):.3g}"


        # affichage

        plt.scatter(df_utile["Taux_chômage"], df_utile["value"], alpha=0.7, color = couleurs[i])
        plt.plot(xs, pred, color=couleurs[i],label=f"{liste_annee[i]} : {equation} (R² = {modele.rsquared:.2f})")

    plt.xlabel("Taux de chômage (%)")
    plt.ylabel("Taux de mortalité (pour 100 000)")
    plt.title("Mortalité en fonction du chômage")
    plt.legend()
    plt.show()
    
##Fonction qui prend en argument un dataframe et renvoie les coefficients de la mortalité pour 100 000 sur le taux de chômage en fonction des années

def plot_coeff_reg_mortalite_chomage(df) : 
    plt.figure(figsize=(9, 6))
    annee, coef = [], []
    for i in range (1991, 2024) : # calcul du coefficient de la regression pour chaque annéee
        
        # selection des informations utiles
        df_utile = df[(df["cause_name"] == "All causes") & (df["year"] == i)]
        df_utile = df_utile.dropna(subset=["value", "Taux_chômage"])

        # Calcul de la régression
        modele = smf.ols("value ~ Taux_chômage", data=df_utile).fit()
        coef.append(modele.params["Taux_chômage"])
        annee.append(i)

    # affichage
    plt.scatter(annee, coef, alpha=0.7, color = "purple")

    plt.xlabel("Année")
    plt.ylabel("Coefficient de la régression")
    plt.title("Evolution du coefficient de la régression de la mortalité pour 100 000 sur le taux de chômage")
    plt.legend()
    plt.show()