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
    df_cause = df[df["cause_name"]==cause].dropna(subset=["value", x]).copy()
    df_cause["log_value"]=np.log(df_cause["value"])
    df_cause = df_cause.rename(columns={x: "chomage"})

    modele=smf.ols("log_value~chomage+C(location_id)+C(year)", data=df_cause).fit(cov_type="cluster", cov_kwds={"groups": df_cause["location_id"]})
    return modele.params["chomage"], modele.pvalues["chomage"]
