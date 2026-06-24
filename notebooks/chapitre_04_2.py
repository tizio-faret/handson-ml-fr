import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Chapitre 4 - entraînement de modèles**
    ## Deuxième partie : régularisation et régression logistique
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ce notebook est inspiré du chapitre 4 de _Hands-On Machine Learning with Scikit-Learn and PyTorch_ d'Aurélien Géron. Le code est adapté de [ageron/handson-mlp](https://github.com/ageron/handson-mlp), sous licence Apache 2.0.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    rouge = "#c53030"
    orange = "#dd6b20"
    gris = "#718096"
    bleu = "#2b6cb0"
    vert = "#2f855a"
    violet = "#6b46c1"
    jaune = "#fcbf49"
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # I. Régression linéaire

    ## A. Principe
    """)
    return


if __name__ == "__main__":
    app.run()
