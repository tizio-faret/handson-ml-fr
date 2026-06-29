import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Chapitre 4 - entraînement de modèles**
    ## Première partie : régressions, descente de gradient et courbes d'apprentissage
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
    return bleu, gris, jaune, mo, orange, rouge, vert, violet


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # I. Régression linéaire

    ## A. Principe

    Au chapitre 2, c'est le premier modèle que l'on a entrainé après avoir traité nos données. Pour un district donné, il faisait l'hypothèse que le prix d'un logement était une combinaison linéaire de ses différentes caractéristiques.

    ### Notation in extenso

    Cet exemple capture bien l'essence du problème de la régression linéaire : on fait l'hypothèse que le label $y$ s'explique linéairement par rapport aux prédicteurs $x_i$.

    $$ y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n + \epsilon$$

    Supposant que cette modélisation représente correctement la structure sous-jacente du problème, on s'en sert pour réaliser des prédictions $\hat{y}$ :

    $$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

    $$ y - \hat{y} = \epsilon $$

    - $y$ est la valeur du label.
    - $\hat{y}$ est la valeur prédite du label.
    - $x_i$ est la valeur du $i^{\text{ème}}$ prédicteur.
    - $\theta_j$ est le $j^{\text{ème}}$ paramètre du modèle. Il inclut le biais $\theta_0$ et les poids des prédicteurs $\theta_1, \theta_2, \cdots, \theta_n$.
    - $\epsilon$ représente l'erreur résiduelle.
    - $n$ est le nombre de prédicteurs.

    > Le biais $\theta_0$ est aussi appelé intercept.

    ### Notation vectorielle

    On peut aussi opter pour une notation vectorielle, plus concise :

    $$\hat{y} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta} \cdot \mathbf{x}$$

    - $h_{\boldsymbol{\theta}}$ est la fonction d'hypothèse, qui utilise les paramètres du modèle $\boldsymbol{\theta}$.
    - $\boldsymbol{\theta}$ est le *vecteur de paramètres* du modèle, contenant le terme de biais $\theta_0$ et les poids des features $\theta_1$ à $\theta_n$.
    - $\mathbf{x}$ est le *vecteur de features* de l'instance, contenant $x_0$ à $x_n$, avec $x_0$ toujours égal à 1.
    - $\boldsymbol{\theta} \cdot \mathbf{x}$ est le produit scalaire des vecteurs $\boldsymbol{\theta}$ et $\mathbf{x}$, qui est égal à $\theta_0 \mathbf{x}_0 + \theta_1 \mathbf{x}_1 + \theta_2 \mathbf{x}_2 + \dots + \theta_n \mathbf{x}_n$.

    > On notera désormais les vecteurs en gras - c'est une convention assez courante.

    ### Notation matricielle

    Les notations $y$, $\hat{y}$ et $\mathbf{x}$ présentées ci-dessus ne concernaient jusque-là qu'une seule instance. Il nous sera utile pour la suite de formaliser la modélisation linéaire à l'échelle de $m$ instances :

    $$\mathbf{y} = \mathbf{X} \boldsymbol{\theta} + \boldsymbol{\epsilon}$$

    - $\mathbf{y}$ est le vecteur colonne dont la i-ème ligne correspond au label ${y}^{(i)}$ de la i-ème instance.
    - $\mathbf{X}$ est la matrice de **conception** ou de **design**, dont la i-ème ligne correspond au vecteur $\mathbf{x}^{(i)}$ de features de la i-ème instance.
    - $\boldsymbol{\theta}$ est le *vecteur de paramètres* du modèle. C'est **l'inconnue à déterminer**.
    - $\boldsymbol{\epsilon}$ est le vecteur colonne dont la i-ème ligne correspond au résidu ${\epsilon}^{(i)}={y}^{(i)} - \hat{y}^{(i)}$ associé à la i-ème instance.

    > On a donc $\hat{y}^{(i)}=h_{\boldsymbol{\theta}}\!\left(\mathbf{x}^{(i)}\right)=\boldsymbol{\theta}^{\mathsf{T}} \mathbf{x}^{(i)}$


    ### Fonctions d'erreur

    Pour adapter un modèle à un contexte donné, on doit définir une **mesure de performance**. On ajuste ensuite les paramètres du modèle de sorte à optimiser cette performance sur un dataset : c'est l'étape d'entraînement du modèle.

    Comme on l'a vu au chapitre 2, une métrique typique pour les problèmes de régression est la **RMSE** (Root Mean Squared Error). Nous verrons qu'il existe deux approches pour trouver la valeur $\hat{\boldsymbol{\theta}}$ qui minimise cette fonction.

    $$\text{RMSE}(\boldsymbol{\theta})  = \sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)}-y^{(i)}\right)^2} $$

    La fonction racine carrée étant croissante sur $\mathbb{R}_+$, $\hat{\boldsymbol{\theta}}$  minimise aussi la **MSE**.

    $$\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)}- y^{(i)} \right)^2$$

    En pratique, on utilise donc plutôt la MSE pour trouver $\hat{\boldsymbol{\theta}}$.

    > On écrit RMSE et MSE comme des fonctions de $\boldsymbol{\theta}$ pour alléger la notation. Si on voulait être plus rigoureux on devrait dire que ce sont des fonctions de $\mathbf{X}$, de $\mathbf{y}$ et de $\boldsymbol{\theta}$.

    ### Visualisation dans le plan

    Avec un seul prédicteur, le modèle est une **droite** dans le plan. Deux paramètres suffisent à la décrire entièrement :

    - $\theta_0$ est le **biais** : c'est **l'ordonnée à l'origine** de la droite de régression.
    - $\theta_1$ est la **pente** : c'est la **pente** de la droite de régression.

    Essayez d'amener la droite au plus près du nuage. L'erreur quadratique moyenne (MSE) affichée mesure la qualité de l'ajustement : c'est la moyenne des carrés des écarts verticaux (en rouge) entre la droite et chaque point.
    """)
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Données artificielles partagées par les deux visualisations
    _rng = np.random.default_rng(42)

    # Jeu avec un seul prédicteur 
    N_2D = 60
    x_2d = _rng.uniform(0, 2, N_2D)
    THETA0_VRAI, THETA1_VRAI = 4.0, 3.0
    y_2d = THETA0_VRAI + THETA1_VRAI * x_2d + _rng.normal(0, 1.0, N_2D)

    # Paramètres optimaux pour générer la droite de référence
    _A2 = np.c_[np.ones(N_2D), x_2d]
    theta_opt_2d = np.linalg.lstsq(_A2, y_2d, rcond=None)[0]

    # Jeu avec deux prédicteurs
    N_3D = 80
    x1_3d = _rng.uniform(0, 2, N_3D)
    x2_3d = _rng.uniform(0, 2, N_3D)
    y_3d = 2.0 + 1.5 * x1_3d - 2.0 * x2_3d + _rng.normal(0, 0.6, N_3D)

    _A3 = np.c_[np.ones(N_3D), x1_3d, x2_3d]
    theta_opt_3d = np.linalg.lstsq(_A3, y_3d, rcond=None)[0]
    return (
        N_2D,
        N_3D,
        np,
        plt,
        theta_opt_2d,
        theta_opt_3d,
        x1_3d,
        x2_3d,
        x_2d,
        y_2d,
        y_3d,
    )


@app.cell(hide_code=True)
def _(
    bleu,
    montrer_optimal_2d,
    montrer_residus_2d,
    np,
    orange,
    plt,
    rouge,
    theta0_2d,
    theta1_2d,
    theta_opt_2d,
    vert,
    x_2d,
    y_2d,
):
    _t0 = theta0_2d.value
    _t1 = theta1_2d.value

    # Prédictions du modèle courant et erreur quadratique moyenne
    _y_pred = _t0 + _t1 * x_2d
    _mse = float(np.mean((y_2d - _y_pred) ** 2))
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.scatter(x_2d, y_2d, color=bleu, s=35, alpha=0.8, label="Données", zorder=3)

    # Segments verticaux entre chaque point et la droite
    if montrer_residus_2d.value:
        for _xi, _yi, _ypi in zip(x_2d, y_2d, _y_pred):
            _ax.plot([_xi, _xi], [_yi, _ypi], color=rouge, lw=0.8, alpha=0.5, zorder=2)

    _xs = np.array([x_2d.min() - 0.1, x_2d.max() + 0.1])
    _ax.plot(_xs, _t0 + _t1 * _xs, color=orange, lw=2.5, label=fr"$\hat{{y}} = {_t0:.1f} + {_t1:.1f}\,x$", zorder=4,)

    if montrer_optimal_2d.value:
        _b0, _b1 = theta_opt_2d
        _ax.plot( _xs, _b0 + _b1 * _xs, color=vert, lw=2.0, ls="--", label=fr"Optimale : ${_b0:.2f} + {_b1:.2f}\,x$", zorder=4,)

    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_title(f"MSE = {_mse:.3f}")
    _ax.legend(loc="upper left")
    _ax.grid(True, alpha=0.3)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    theta0_2d = mo.ui.slider(start=-2.0, stop=10.0, step=0.1, value=2.0, label=r"$\theta_0$ (biais)")
    theta1_2d = mo.ui.slider(start=-4.0, stop=8.0, step=0.1, value=1.0, label=r"$\theta_1$ (pente)")
    montrer_residus_2d = mo.ui.checkbox(value=True, label="Afficher les résidus")
    montrer_optimal_2d = mo.ui.checkbox(value=False, label="Afficher la droite optimale")

    mo.vstack([mo.hstack([theta0_2d, theta1_2d], justify="start", gap=2), mo.hstack([montrer_residus_2d, montrer_optimal_2d], justify="start", gap=2),])
    return montrer_optimal_2d, montrer_residus_2d, theta0_2d, theta1_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisation dans l'espace

    En ajoutant un second prédicteur, on doit définir une valeur $\hat{y}$ pour chaque paire $(x_1, x_2)$ de valeurs possibles pour les prédicteurs. On a désormais une fonction de deux variables : sa représentation n'est donc plus une courbe (2D) mais une surface (3D). Puisque notre régression est linéaire, cette surface est un **plan**.

    La logique est identique, mais il y a maintenant trois paramètres à régler. Le plan orange est le modèle, chaque point bleu est une observation, et les segments rouges sont toujours les résidus.
    """)
    return


@app.cell(hide_code=True)
def _(
    N_3D,
    bleu,
    montrer_optimal_3d,
    montrer_residus_3d,
    np,
    orange,
    plt,
    rouge,
    theta0_3d,
    theta1_3d,
    theta2_3d,
    theta_opt_3d,
    vert,
    x1_3d,
    x2_3d,
    y_3d,
):
    _t0 = theta0_3d.value
    _t1 = theta1_3d.value
    _t2 = theta2_3d.value

    _y_pred = _t0 + _t1 * x1_3d + _t2 * x2_3d
    _mse = float(np.mean((y_3d - _y_pred) ** 2))

    _fig = plt.figure(figsize=(6, 4))
    _ax = _fig.add_subplot(111, projection="3d")

    # Le plan du modèle, tracé comme une surface sur la grille des prédicteurs.
    _g1, _g2 = np.meshgrid(np.linspace(0, 2, 12), np.linspace(0, 2, 12))
    _plan = _t0 + _t1 * _g1 + _t2 * _g2
    _ax.plot_surface(_g1, _g2, _plan, alpha=0.35, color=orange, edgecolor="none")

    # Résidus : segments verticaux des points vers le plan.
    if montrer_residus_3d.value:
        for _i in range(N_3D):
            _ax.plot([x1_3d[_i], x1_3d[_i]], [x2_3d[_i], x2_3d[_i]], [y_3d[_i], _y_pred[_i]], color=rouge, lw=0.7, alpha=0.5,)

    if montrer_optimal_3d.value:
        _b0, _b1, _b2 = theta_opt_3d
        _plan_opt = _b0 + _b1 * _g1 + _b2 * _g2
        _ax.plot_wireframe(_g1, _g2, _plan_opt, color=vert, lw=0.8, alpha=0.6)

    _ax.scatter(x1_3d, x2_3d, y_3d, color=bleu, s=25, alpha=0.9, depthshade=True)

    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_zlabel("$y$")
    _ax.set_title(fr"$\hat{{y}} = {_t0:.1f} + {_t1:.1f}\,x_1 + {_t2:.1f}\,x_2$ ; MSE = {_mse:.3f}")
    _ax.view_init(elev=22, azim=-60)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    theta0_3d = mo.ui.slider(start=-3.0, stop=7.0, step=0.1, value=0.0, label=r"$\theta_0$ (biais)")
    theta1_3d = mo.ui.slider(start=-4.0, stop=5.0, step=0.1, value=0.0, label=r"$\theta_1$ (poids de $x_1$)")
    theta2_3d = mo.ui.slider(start=-5.0, stop=4.0, step=0.1, value=0.0, label=r"$\theta_2$ (poids de $x_2$)")
    montrer_residus_3d = mo.ui.checkbox(value=True, label="Afficher les résidus")
    montrer_optimal_3d = mo.ui.checkbox(value=False, label="Afficher le plan optimal")

    mo.vstack([mo.hstack([theta0_3d, theta1_3d, theta2_3d], justify="start", gap=2), mo.hstack([montrer_residus_3d, montrer_optimal_3d], justify="start", gap=2),])
    return (
        montrer_optimal_3d,
        montrer_residus_3d,
        theta0_3d,
        theta1_3d,
        theta2_3d,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## B. L'équation normale

    Il existe deux approches pour trouver la valeur $\hat{\boldsymbol{\theta}}$ qui minimise la MSE. La première donne une valeur exacte (équation normale), la seconde est une méthode heuristique (descente de gradient).

    Soit l'expression matricielle de la MSE :

    $$MSE(\boldsymbol{\theta}) = \frac{1}{m} {\|\mathbf{y} - \mathbf{X} \boldsymbol{\theta}\|_2}^2$$

    Comme cette fonction est convexe, elle admet en $\hat{\boldsymbol{\theta}}$ un minimum global. Son expression est appelée **équation normale **:

    $$\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^T \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{y}$$

    ### Implémentation de l'équation normale

    On génère d'abord un jeu de données pseudo-linéaire pour pouvoir implémenter notre modèle de régression linéaire. On considère le cas à un seul prédicteur : $y = \theta_0 + \theta_1 x + \epsilon$ .

    > La génération de ce jeu requiert de fixer la relation linéaire entre $y$ et $x$. Une fois les données créées, on considère ces coefficients comme des paramètres cachés. Notre objectif va être de les estimer uniquement à partir des couples $(x_i, y_i)$ observés.
    """)
    return


@app.cell
def _(m_slider, np, sigma_slider):
    _rng = np.random.default_rng(seed=42)
    m = m_slider.value  # nombre d'instances

    # Génération aléatoire de m valeurs associées au prédicteur x
    X_reg = 2 * _rng.random((m, 1))  

    # Modélisation d'une relation linéaire bruitée 
    bruit = sigma_slider.value * _rng.standard_normal((m, 1))
    y_reg = 4 + 3 * X_reg + bruit
    return X_reg, m, y_reg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On visualise le nuage de points.
    """)
    return


@app.cell(hide_code=True)
def _(X_reg, bleu, plt, y_reg):
    plt.figure(figsize=(6, 4))
    plt.plot(X_reg, y_reg, ".", color=bleu)
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([0, 2, 0, 15])
    plt.grid()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Le vecteur $\boldsymbol{\theta}$ combine les inconnues à déterminer : le biais $\theta_0$ et le poids $\theta_1$ du prédicteur. Soit l'expression de $\hat{\boldsymbol{\theta}}$ :

    $$\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^T \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{y}$$

    Puisqu'on doit calculer un inverse et faire des produits de matrices, on va utiliser le package NumPy qui simplifie la syntaxe et optimise le temps de calcul. La syntaxe dédiée est la suivante :
    - `np.linalg.inv(A)` calcule l'inverse de $A$
    - `A @ B` calcule le produit de $A$ par $B$
    - `A.T` calcule la transposée $A^T$ de $A$
    """)
    return


@app.cell
def _(X_reg, np, y_reg):
    from sklearn.preprocessing import add_dummy_feature

    X_b_reg = add_dummy_feature(X_reg)  # On ajoute x0 = 1 à chaque instance
    theta_best_reg = np.linalg.inv(X_b_reg.T @ X_b_reg) @ X_b_reg.T @ y_reg
    return add_dummy_feature, theta_best_reg


@app.cell(hide_code=True)
def _(theta_best_reg):
    print(f"Valeur estimée de theta_0 : {theta_best_reg[0,0]:.5f}")
    print("Valeur réelle de theta_0 : 4\n")
    print(f"Valeur estimée de theta_1 : {theta_best_reg[1,0]:.5f}")
    print("Valeur réelle de theta_1 : 3")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Le bruit introduit dans la génération du jeu est **trop important au regard du nombre d'instances** pour permettre au modèle de retrouver les valeurs des paramètres.

    Essayez de modifier le nombre d'instance et la quantité de bruit pour faire évoluer la précision des estimations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    m_slider = mo.ui.slider(start=10, stop=1000, step=10, value=80, label=r"$m$ (nombre d'instances)")
    sigma_slider = mo.ui.slider(start=0.1, stop=1, step=0.1, value=1, label=r"$\sigma$ (quantité de bruit)")
    mo.vstack([mo.hstack([m_slider, sigma_slider], justify="start", gap=2)])
    return m_slider, sigma_slider


@app.cell(hide_code=True)
def _(
    X_reg,
    bleu,
    gris,
    m_slider,
    np,
    plt,
    rouge,
    sigma_slider,
    theta_best_reg,
    vert,
    y_reg,
):
    fig, (ax_theta, ax_data) = plt.subplots(1, 2, figsize=(10, 5))

    # Figure de gauche

    ax_theta.scatter(4, 3, color=vert, s=180, marker="*", zorder=3,
                     label="Vraies valeurs (4, 3)")

    ax_theta.plot([4, theta_best_reg[0, 0]], [3, theta_best_reg[1, 0]],
                  color=gris, ls="--", lw=0.8, alpha=0.6, zorder=2)

    ax_theta.scatter(theta_best_reg[0, 0], theta_best_reg[1, 0], color=rouge, s=90, zorder=4,
                     label=f"Estimation ({theta_best_reg[0,0]:.2f}, {theta_best_reg[1,0]:.2f})")

    ax_theta.axhline(3, color=gris, lw=0.5, alpha=0.4)
    ax_theta.axvline(4, color=gris, lw=0.5, alpha=0.4)

    ax_theta.set_xlabel(r"$\theta_0$")
    ax_theta.set_ylabel(r"$\theta_1$", rotation=0)
    ax_theta.set_xlim(3.6, 4.4)
    ax_theta.set_ylim(2.6, 3.4)
    ax_theta.set_aspect("equal")
    ax_theta.grid(True, alpha=0.3)
    ax_theta.legend(loc="upper left")
    ax_theta.set_title(f"Espace des paramètres  ($m$ = {m_slider.value} ; $\sigma$ = {sigma_slider.value})")

    # Figure de droite

    ax_data.plot(X_reg, y_reg, ".", color=bleu, markersize=5)

    x_line = np.array([0, 2])
    y_line = theta_best_reg[0, 0] + theta_best_reg[1, 0] * x_line
    ax_data.plot(x_line, y_line, "-", color=rouge, label="Régression linéaire")

    ax_data.set_xlabel("$x_1$")
    ax_data.set_ylabel("$y$", rotation=0)
    ax_data.axis([0, 2, 0, 15])
    ax_data.grid(True, alpha=0.3)
    ax_data.legend(loc="upper left")
    ax_data.set_title("Nuage de points")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > <a id="def-dataset"></a> Pour éviter que la modification des paramètres n'affecte toute la suite du Notebook (ce qui le ralentirait), on décide de fixer une bonne fois pour toute $m=200$ et $\sigma = 1$.
    """)
    return


@app.cell
def _(add_dummy_feature, np):
    _m = 200
    _data_rng = np.random.default_rng(seed=42)
    X = 2 * _data_rng.random((_m, 1))
    y = 4 + 3 * X + _data_rng.standard_normal((_m, 1))

    X_b = add_dummy_feature(X)
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return X, X_b, theta_best, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prédictions

    On peut désormais prédire le label $\hat{y}$ d'une instance $\mathbf{x}$ inédite  : $\hat{y}=\boldsymbol{\theta}^{\mathsf{T}} \mathbf{x}$ .

    On crée à cet effet deux nouvelles instances $x^{(1)}$ et $x^{(2)}$ que l'on combine dans un vecteur $X_{new}$ :

    $$ X_{\text{new}}= \begin{bmatrix} x^{(1)} \\ x^{(2)} \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \end{bmatrix} $$
    """)
    return


@app.cell
def _(add_dummy_feature, np, theta_best):
    X_new = np.array([[0], [2]])
    X_new_b = add_dummy_feature(X_new)  # ajoute x0 = 1 à chaque instance
    y_predict = X_new_b @ theta_best
    y_predict

    print("--- X_new_b @ theta_best ---")
    print(f"Estimation du label de l'instance x_1 : {y_predict[0,0]:.3f} \nEstimation du label de l'instance x_2 : {y_predict[1,0]:.3f}")
    return X_new, X_new_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## C. Décomposition en valeurs singulières

    En pratique, on ne calcule pas $\hat{\boldsymbol{\theta}}$  avec l'équation normale : elle est numériquement peu stable et $\mathbf{X}^T \mathbf{X}$ n'est pas toujours inversible. À la place, l'implémentation de la régression linéaire de Scikit-Learn mobilise une **décomposition en valeurs singulières (SVD)** de $\mathbf{X}$.

    ### L'idée de la SVD
    La SVD généralise en quelque sorte la diagonalisation aux **matrices rectangulaires**. Pour $\mathbf{A} \in M_{m,n}$, elle affirme qu'on peut écrire :

    $$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma} \mathbf{V}^T$$

    - $\mathbf{U}$ est orthogonale ($m \times m$)
    - $\mathbf{V}$ est orthogonale ($n \times n$)
    - $\boldsymbol{\Sigma}$ est diagonale rectangulaire

    $$\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1 & 0 & \cdots \\ 0 & \sigma_2 & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

    Les $\sigma_i$ sont **les racines carrées des valeurs propres** de $\mathbf{A}^T \mathbf{A}$. Ce sont les **valeurs singulières** de $\mathbf{A}$. Elles sont indicées par ordre décroissant : $\sigma_1 \geq \cdots \geq \sigma_n$.

    ### Équation normale vs. inversion matricielle

    Pour calculer $\hat{\boldsymbol{\theta}}$, on a choisi une mesure d'erreur (RMSE/MSE) et on a défini $\hat{\boldsymbol{\theta}}$ comme la valeur qui minimise cette erreur. Son expression exacte (équation normale) s'obtient en annulant le gradient de la MSE.

    Une approche plus simple profiterait que $\hat{\boldsymbol{y}} = \mathbf{X} \hat{\boldsymbol{\theta}}$ pour poser $\hat{\boldsymbol{\theta}} = \mathbf{X}^{-1} \hat{\boldsymbol{y}}$. Malheureusement $\mathbf{X}$ est rarement inversible :

    - $\mathbf{X}$ est généralement rectangulaire : on a rarement autant d'instances que de prédicteurs.
    - $\ker(\mathbf{X}) \neq \{0\}$ : certains prédicteurs peuvent être colinéaires.

    En pratique, on remplace l'inverse par un pseudo-inverse $\mathbf{X}^+$, que la SVD permet de calculer de manière stable.

    ### Construction d'un pseudo inverse par la SVD

    Soit $\mathbf{A} \in M_{m,n}$ , sa décomposition en valeurs singulières (SVD) donne : $\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$. On définit son pseudo-inverse $\mathbf{A}^+ \in M_{n,m}$ comme suit :

    $$\mathbf{A}^+ = \mathbf{V} \boldsymbol{\Sigma}^+ \mathbf{U}^T,$$

    où $\boldsymbol{\Sigma}^+$ s'obtient à partir de $\boldsymbol{\Sigma}$ en transposant la matrice et en remplaçant chaque valeur singulière non nulle par son inverse.

    On peut montrer (on ne le fera pas) que $\hat{\boldsymbol{\theta}} = \mathbf{X}^{+} \boldsymbol{y}$
    minimise aussi la MSE. C'est tout l'intérêt de la SVD.

    > Si $\mathbf{X}^T \mathbf{X}$ est inversible, l'expression du pseudo-inverse généré par la SVD et l'équation normale coïncident : $\mathbf{X}^+ = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$ .

    ### Remarque <a id="seuil-rcond"></a>

    En pratique on rencontre souvent des valeurs singulières $\sigma_i$ très petites (erreurs d'arrondi..). Puisque les inverser donnerait des $1/\sigma_i$ gigantesques, Scikit-learn fixe un seuil `rcond` de sorte qu'une valeur singulière vérifiant

    $$\sigma_i < \texttt{rcond} \times \sigma_{\max}$$

    est traitée comme nulle.

    ---

    ## D. Implémentation avec Scikit-Learn <a id="prediction-svd"></a>

    La régression linéaire s'implémente typiquement avec le prédicteur `LinearRegression()`.
    """)
    return


@app.cell
def _(X, y):
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    print("--- `lin_reg.intercept_` et `lin_reg.coef_` ---")
    print(f"Ordonnée à l'origine estimée : {lin_reg.intercept_[0]:.3f} \nPente estimée : {lin_reg.coef_[0,0]:.3f}")
    return LinearRegression, lin_reg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut ensuite prédire les labels de plusieurs instances en les combinant dans une matrice.

    > On a vu que si $\mathbf{X}^T \mathbf{X}$ est inversible (et c'est le cas ici), l'expression du pseudo-inverse généré par la SVD (utilisé par `lin_reg.predict`) et l'équation normale coïncident.

    On s'attend donc à retomber sur les mêmes prédictions que celles générées avec `theta_best`.
    """)
    return


@app.cell
def _(X_new, lin_reg):
    print("--- lin_reg.predict() ---")
    print(f"Estimation du label de l'instance x_1 : {lin_reg.predict(X_new)[0,0]:.3f} \nEstimation du label de l'instance x_2 : {lin_reg.predict(X_new)[1,0]:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comme convenu les prédictions sont identiques.

    ### La méthode `scipy.linalg.lstsq()`

    `LinearRegression()` mobilise la fonction `scipy.linalg.lstsq()`. Son appel renvoie :

    - `x` : la solution des moindres carrés, c'est-à-dire $\hat{\boldsymbol{\theta}}$.
    - `residues` : somme des carrés des résidus ${\|\mathbf{y} - \mathbf{X} \boldsymbol{\theta}\|_2}^2$.
    - `rank` : le rang de $\mathbf{X}$.
    - `s` : liste des valeurs singulières de $\mathbf{X}$ (les $\sigma_i$ de la SVD).

    > `residues` vaut un scalaire si $\hat{\boldsymbol{y}}$ est unidimensionnel et si $m > n$ avec $\mathbf{X}$ de rang plein. Sinon c'est un tableau vide.

    NB : `scipy.linalg.lstsq()` dispose du paramètre `rcond` [dont on explique l'utilité  ici](#seuil-rcond).
    """)
    return


@app.cell
def _(X_b, np, y):
    theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
    theta_best_svd

    print("--- scipy.linalg.lstsq() ---")
    print(f"Ordonnée à l'origine estimée : {theta_best_svd[0,0]:.3f} \nPente estimée : {theta_best_svd[1,0]:.3f}")
    return rank, s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On retrouve les mêmes valeurs qu'avec `lin_reg.intercept_` et `lin_reg.coef_`.

    ### Conditionnement - matrices inversibles

    Étant donné un système linéaire $\mathbf{A}\mathbf{x} = \mathbf{y}$, il arrive parfois qu'une petite variation sur $\mathbf{y}$ entraîne une grande variation sur $\mathbf{x}$. On dit dans ce cas que
    la matrice $\mathbf{A}$, ou le problème, est **mal conditionnée**.

    Plus formellement, pour une matrice inversible $\mathbf{A}$, on définit son conditionnement $K(\mathbf{A})$ comme suit :

    $$ K(\mathbf{A}) = \lVert \mathbf{A} \rVert \times \lVert \mathbf{A}^{-1} \rVert \in [1,+\infty[$$

    Si on choisit la norme 2 (norme spectrale), on a :

    $$K_2(\mathbf{A}) = \lVert \mathbf{A} \rVert_2 \times \lVert \mathbf{A}^{-1} \rVert_2 = \frac{\sigma_{\max}}{\sigma_{\min}}$$

    Où les $\sigma_i$ sont les valeurs singulières de $A$.

    ### Conditionnement - généralisation

    Si $\mathbf{A}$ est non inversible, on a ${\sigma_{\min}}=0$ et donc $K_2(\mathbf{A})=\infty$. Le problème est infiniment mal conditionné. C'est cohérent mais peu informatif : on lui préfère donc la définition ci-dessous.

    Si $\mathbf{A}$ est non inversible, on remplace $\mathbf{A}^{-1}$ par $\mathbf{A}^{+}$.

    $$K(\mathbf{A}) = \lVert \mathbf{A} \rVert_2 \times \lVert \mathbf{A}^+ \rVert_2 = \frac{\sigma_{\max}}{\sigma_r}$$

    Où $\sigma_r$ est la plus petite valeur singulière non nulle de $\mathbf{A}$.

    > On note $\sigma_r$ car le rang $r$ de $\mathbf{A}$ correspond à l'indice de sa plus petite valeur singulière non nulle.

    Puisque `scipy.linalg.lstsq()` renvoie justement le rang et la liste des valeurs singulières de $\mathbf{X}$, on peut facilement calculer son conditionnement $K(\mathbf{\mathbf{X}})$.<a id="complexite-svd-en"></a>
    """)
    return


@app.cell
def _(rank, s):
    K_X = s[0] / s[rank - 1]
    print(f"Le conditionnement de X vaut : {K_X:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SVD : complexité computationnelle <a id="complexite-svd"></a>

    Petit bilan. On rappelle que $m$ correspond au nombre d'instances et $n$ au nombre de features. En pratique, on a $m \gg n$ la plupart du temps - important pour ce qui suit.

    - Le calcul de $\hat{\boldsymbol{\theta}}$ avec l'équation normale requiert l'inversion matricielle de $\mathbf{X}^T\mathbf{X}$. C'est une opération dont la complexité est comprise entre $O(n^{2.4})$ et $O(n^{3})$, mais former $\mathbf{X}^T\mathbf{X}$ coûte en plus $O(mn^2)$. **Complexité totale retenue **: $O(mn^2)$.

    - Le calcul de $\hat{\boldsymbol{\theta}}$ avec le pseudo-inverse construit par SVD a aussi une complexité de $O(mn^2)$.

    L'avantage de la SVD n'est donc pas sa rapidité, mais bien sa **stabilité** et le fait qu'elle soit **toujours calculable**.

    # II. Descente de gradient

    > « Nous verrons qu'il existe deux approches pour trouver la valeur $\hat{\boldsymbol{\theta}}$ qui minimise la MSE. »

    La descente de gradient constitue la deuxième approche.

    ## A. Généralités

    L'idée générale consiste à parcourir itérativement **l'espace des paramètres du modèle** en choisissant arbitrairement une origine $\hat{\boldsymbol{\theta}_0}$, puis à se déplacer pas à pas dans la **direction de plus forte pente**, c'est à dire celle qui fait le plus diminuer la MSE. La taille d'un pas s'appelle le **learning rate**.

    Les schémas ci-dessous permettent de visualiser une descente de gradient sur [ce jeu de données](#def-dataset).

    > La figure de gauche exhibe la valeur de la MSE en différents points de l'espace des paramètres. La figure de droite propose une vue de dessus de cette courbe, en lignes de niveau. <a id="grad-vis"></a>
    """)
    return


@app.cell(hide_code=True)
def _(X, X_b, mo, np, y):
    # minimum théorique (équation normale)
    gd_opt = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y   

    _o0, _o1 = float(gd_opt[0, 0]), float(gd_opt[1, 0])
    gd_t0_lo, gd_t0_hi = _o0 - 3.5, _o0 + 3.5
    gd_t1_lo, gd_t1_hi = _o1 - 3.5, _o1 + 3.5

    # Grille de la surface de coût MSE(theta0, theta1)
    gd_TH0, gd_TH1 = np.meshgrid(np.linspace(gd_t0_lo, gd_t0_hi, 80),
                                 np.linspace(gd_t1_lo, gd_t1_hi, 80))
    _xr, _yr = X.ravel(), y.ravel()
    _pred = gd_TH0[..., None] + gd_TH1[..., None] * _xr[None, None, :]
    gd_Z = ((_pred - _yr[None, None, :]) ** 2).mean(axis=2)
    gd_zcap = float(gd_Z.max())

    def gd_mse_point(a, b):
        _e = (a + b * X.ravel()) - y.ravel()
        return float(np.mean(_e ** 2))

    # origine aléatoire, change à chaque réinitialisation
    gd_rng = np.random.default_rng(seed=42)    

    def gd_fresh():
        _a = gd_rng.uniform(gd_t0_lo + 0.6, gd_t0_hi - 0.6)
        _b = gd_rng.uniform(gd_t1_lo + 0.6, gd_t1_hi - 0.6)
        _th = np.array([[_a], [_b]])
        return {"k": 0, "theta": _th, "path": [_th.ravel()]}

    def gd_step_fn(state, eta):
        _th = state["theta"]
        _mm = X_b.shape[0]
        # gradient de la MSE
        _grad = (2.0 / _mm) * X_b.T @ (X_b @ _th - y)        
        _new = _th - eta * _grad
        return {"k": state["k"] + 1, "theta": _new, "path": state["path"] + [_new.ravel()]}

    gd_get, gd_set = mo.state(gd_fresh())
    return (
        gd_TH0,
        gd_TH1,
        gd_Z,
        gd_fresh,
        gd_get,
        gd_mse_point,
        gd_opt,
        gd_set,
        gd_step_fn,
        gd_t0_hi,
        gd_t0_lo,
        gd_t1_hi,
        gd_t1_lo,
        gd_zcap,
    )


@app.cell(hide_code=True)
def _(gd_fresh, gd_get, gd_set, gd_step_fn, mo):
    gd_eta = mo.ui.slider(0.01, 0.6, 0.01, value=0.39, label=r"$\eta$ (pas d'apprentissage)")
    gd_btn_step = mo.ui.button(label="Itération suivante",
                               on_change=lambda _: gd_set(gd_step_fn(gd_get(), gd_eta.value)))
    gd_btn_reset = mo.ui.button(label="Réinitialiser (nouvelle origine)",
                                on_change=lambda _: gd_set(gd_fresh()))

    mo.vstack([gd_eta, mo.hstack([gd_btn_step, gd_btn_reset], justify="start", gap=1)])
    return


@app.cell(hide_code=True)
def _(
    gd_TH0,
    gd_TH1,
    gd_Z,
    gd_get,
    gd_mse_point,
    gd_opt,
    gd_t0_hi,
    gd_t0_lo,
    gd_t1_hi,
    gd_t1_lo,
    gd_zcap,
    np,
    plt,
):
    _state = gd_get()
    _k, _theta = _state["k"], _state["theta"]
    _path = np.array(_state["path"])
    _dist = float(np.linalg.norm(_theta - gd_opt))
    _mse_now = gd_mse_point(_theta[0, 0], _theta[1, 0])
    _pz = np.array([gd_mse_point(a, b) for a, b in _path])
    # gd_zcap : léger décalage pour rester visible sur la surface
    _pz_disp = np.minimum(_pz, gd_zcap) + 0.02 * gd_zcap     

    _fig = plt.figure(figsize=(11, 4.5))

    # Surface de coût en 3D, à gauche
    _ax1 = _fig.add_subplot(1, 2, 1, projection="3d")
    _ax1.plot_surface(gd_TH0, gd_TH1, gd_Z, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True)
    _ax1.plot(_path[:, 0], _path[:, 1], _pz_disp, color="crimson", lw=1.8, marker="o", ms=3)
    _ax1.scatter([gd_opt[0, 0]], [gd_opt[1, 0]], [gd_mse_point(gd_opt[0, 0], gd_opt[1, 0])],
                 color="gold", marker="*", s=160)
    _ax1.set_xlabel(r"$\theta_0$"); _ax1.set_ylabel(r"$\theta_1$"); _ax1.set_zlabel("MSE")
    _ax1.set_zlim(0, gd_zcap); _ax1.view_init(elev=30, azim=-60)
    _ax1.set_title("Surface de coût")

    # Vue de dessus (lignes de niveau), à droite
    _ax2 = _fig.add_subplot(1, 2, 2)
    _ax2.contourf(gd_TH0, gd_TH1, gd_Z, levels=30, cmap="viridis")
    _ax2.contour(gd_TH0, gd_TH1, gd_Z, levels=12, colors="white", linewidths=0.4, alpha=0.5)
    _ax2.plot(_path[:, 0], _path[:, 1], color="crimson", lw=1.5, marker="o", ms=3, zorder=5)
    _ax2.scatter([_path[0, 0]], [_path[0, 1]], color="black", marker="s", s=50, zorder=6, label="origine")
    _ax2.scatter([_theta[0, 0]], [_theta[1, 0]], color="crimson", s=70, zorder=7, label="position courante")
    _ax2.scatter([gd_opt[0, 0]], [gd_opt[1, 0]], color="gold", marker="*", s=200,
                 edgecolor="black", lw=0.5, zorder=8, label="minimum")
    _ax2.set_xlim(gd_t0_lo, gd_t0_hi); _ax2.set_ylim(gd_t1_lo, gd_t1_hi)
    _ax2.set_xlabel(r"$\theta_0$"); _ax2.set_ylabel(r"$\theta_1$", rotation=0)
    _ax2.set_title(f"$k$ = {_k}    ;    distance = {_dist:.3f}    ;    MSE = {_mse_now:.3f}")
    _ax2.legend(loc="upper right", fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### L'algorithme formalisé <a id="etapes_BGD"></a>

    Formellement, on note $(\boldsymbol{\theta}_k)$ la suite des positions parcourues au cours de la descente : on calcule itérativement ses termes (les points rouges du schéma) jusqu'à ce qu'une **condition d'arrêt** soit satisfaite. Cette condition devra quantifier à quel point on est proche du minimum théorique, sans le calculer explicitement.

    1. Initialisation : on choisit un $\boldsymbol{\theta}_0$ quelconque.
    2. Itération : le gradient $\nabla\mathrm{MSE}(\boldsymbol{\theta})$ pointant dans la direction de plus forte pente, on se déplace dans la direction opposée :
    $$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\,\nabla\mathrm{MSE}(\boldsymbol{\theta}_k)$$
    où $\eta > 0$ est le **pas**, qu'on appelle aussi en machine learning le **learning rate**.

    La MSE étant convexe, la suite $(\boldsymbol{\theta}_k)$ **converge vers le minimum global** (à condition que le taux d'apprentissage ne soit pas trop élevé, nous y reviendrons). Cet avantage disparaîtra avec les modèles non linéaires des chapitres suivants.

    > Non seulement $\nabla\mathrm{MSE}(\boldsymbol{\theta})$ donne la direction de la plus forte pente, mais sa norme correspond à l'intensité de cette pente. C'est pourquoi la distance entre les positions parcourues semble diminuer (alors que le learning rate est fixe) : l'intensité de la plus forte pente diminue à mesure qu'on s'approche du minimum - c'est le « fond du bol ». C'est d'ailleurs un critère d'arrêt typique : on cesse d'itérer lorsque la pente est suffisamment douce.

    ### Calcul du gradient

    La descente de gradient relève de l'étape d'**entraînement** : on ajuste les paramètres du modèle pour lui permettre de réaliser, un jour, de bonnes prédictions.

    Par conséquent, au moment de la descente, on connaît entièrement les instances sur lesquelles on travaille : les valeurs numériques des vecteurs $\mathbf{X}$ et $\mathbf{y}$ sont fixées et figurent comme des constantes dans l'expression de la MSE. La seule variable est le vecteur $\boldsymbol{\theta}$.

    $$\mathrm{MSE}(\boldsymbol{\theta}) = \frac{1}{m}\,\lVert \mathbf{X}\boldsymbol{\theta} - \mathbf{y}\rVert_2^2 = \frac{1}{m}\sum_{i=1}^{m}\left(\boldsymbol{\theta}^{\mathsf T}\mathbf{x}^{(i)} - y^{(i)}\right)^2$$

    Dans le contexte de la régression, la MSE est donc une fonction à $\dim(\boldsymbol{\theta}) = n+1$ variables ($n$ est le nombre de features).

    > Plutôt que de calculer systématiquement le gradient comme on le détaille ci-dessous, Scikit pré-enregistre l'expression des gradients associés aux fonctions de coût les plus courantes (on a vu la MSE/RMSE, mais il en existe d'autres : log-loss, hinge loss, modified Huber, etc. ).

    /// details | Expression de $\nabla\mathrm{MSE}$ - démonstration (pour aller plus loin)
    On obtient l'expression de $\nabla\mathrm{MSE}(\boldsymbol{\theta})$ en passant par le développement de Taylor à l'ordre 1. On identifie ensuite le gradient via un produit scalaire :

    $$\begin{aligned} MSE(\boldsymbol{\theta} + \mathbf{h}) &= \frac{1}{m} ||\mathbf{X}(\boldsymbol{\theta} + \mathbf{h}) - \mathbf{y}||_2^2 \\ &= \frac{1}{m} \langle \mathbf{X}\boldsymbol{\theta} - \mathbf{y} + \mathbf{X}\mathbf{h}, \mathbf{X}\boldsymbol{\theta} - \mathbf{y} + \mathbf{X}\mathbf{h} \rangle \\  &= \frac{1}{m} \left( \langle \mathbf{X}\boldsymbol{\theta} - \mathbf{y}, \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \rangle + \langle \mathbf{X}\boldsymbol{\theta} - \mathbf{y}, \mathbf{X}\mathbf{h} \rangle + \langle \mathbf{X}\mathbf{h}, \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \rangle + \langle \mathbf{X}\mathbf{h}, \mathbf{X}\mathbf{h} \rangle \right) \\ &=  MSE(\boldsymbol{\theta}) + \underbrace{\frac{2}{m} \langle \mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}), \mathbf{h} \rangle}_{dMSE_{\boldsymbol{\theta}}(\mathbf{h})} + o(||\mathbf{h}||) \end{aligned}$$

    Puisque $dMSE_{\boldsymbol{\theta}}(\mathbf{h}) = \langle \nabla MSE(\boldsymbol{\theta}), \mathbf{h} \rangle$, on conclut :
    ///

    $$\nabla\mathrm{MSE}(\boldsymbol{\theta}) = \frac{2}{m}\,\mathbf{X}^{\mathsf T}(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}).$$

    Le code se contente d'**évaluer numériquement** cette formule au point courant $\boldsymbol{\theta}_k$.

    > Ici, le gradient sert à nous guider à chaque itération dans l'espace des paramètres. Mais on a mentionné un peu plus tôt que l'expression exacte de $\boldsymbol{\theta}$ minimisant la MSE - l'équation normale - s'obtenait en annulant... son gradient. Vérifions-le. $\nabla\mathrm{MSE}(\hat{\boldsymbol{\theta}})=\mathbf 0 \;\Longleftrightarrow\; \mathbf X^\mathsf T(\mathbf X\hat{\boldsymbol{\theta}}-\mathbf y)=\mathbf 0 \;\Longleftrightarrow\; \mathbf X^\mathsf T\mathbf X\hat{\boldsymbol{\theta}}=\mathbf X^\mathsf T\mathbf y \;\Longleftrightarrow\; \hat{\boldsymbol{\theta}} = (\mathbf X^\mathsf T\mathbf X)^{-1}\mathbf X^\mathsf T\mathbf y.$ Bingo !

    ### Difficultés <a id="difficultes"></a>

    La descente de gradient est confrontée à des difficultés diverses. On en cite quelques-unes.

    - Plus un modèle a de paramètres, plus l'espace de recherche a de dimensions. Le calcul du gradient devient alors plus coûteux et la convergence de la descente de gradient peut être ralentie.

    - Si le taux d'apprentissage est trop **faible**, l'algorithme doit enchaîner un grand nombre d'itérations pour converger : c'est long.

    - S'il est trop **élevé**, la descente sursaute le minimum et peut diverger (essayez [ici](#grad-vis) en fixant $\eta=0.6$ par exemple).

    ### Standardiser avant la descente

    Avant d'appliquer la descente de gradient, **mettre les features à la même échelle** (par exemple via `StandardScaler`) permet d'éviter que l'algorithme s'éternise.

    C'est de nouveau une histoire de conditionnement. Des features d'échelles très différentes déforment la cuvette de coût en une vallée allongée - le $K(\mathbf{X}^{\mathsf T}\mathbf{X})$ de $\nabla\mathrm{MSE}(\boldsymbol{\theta})$ explose - et la descente y zigzague péniblement le long de l'axe le plus plat.

    Standardiser, c'est arrondir la cuvette et donc réduire le nombre d'itérations. La figure ci-dessous illustre ce principe.

    > Pour recréer des conditions où des features ont des échelles différentes, il faut un dataset avec (au moins) deux prédicteurs. Ce faisant, le vecteur $\boldsymbol{\theta}$ se retrouve avec 3 composantes : il évolue dans l'espace et non plus dans le plan comme [ici](#grad-vis). Puisqu'on ne peut pas représenter la fonction de coût dans un espace à 4 dimensions $(\theta_0,\theta_1,\theta_2,RMSE(\boldsymbol{\theta}))$, on a retiré le biais $\theta_0$ des figures ci-dessous. Les graphiques restent instructifs : la standardisation arrondit la cuvette au regard de deux dimensions. Elle l'arrondit aussi au regard des trois dimensions, mais c'est un peu plus difficile à visualiser !
    """)
    return


@app.cell(hide_code=True)
def _(np):
    from sklearn.preprocessing import StandardScaler

    # Jeu de données à deux features d'échelles très différentes 
    _rng_std = np.random.default_rng(0)
    _m_std = 200

    # x1 <-> petite échelle [0, 1] ; x2 <-> grande échelle [0, 1000]
    _x1 = _rng_std.random(_m_std)
    _x2 = 1000.0 * _rng_std.random(_m_std)
    y_std = 5.0 + 3.0 * _x1 + 0.02 * _x2 + _rng_std.standard_normal(_m_std)

    # On retire le biais (theta_0 fixé) pour visualiser la cuvette dans le plan des poids (theta_1,
    X_raw = np.c_[_x1, _x2] # features brutes                      
    X_scaled = StandardScaler().fit_transform(X_raw) # features standardisées

    def _cost_grid(features, target, span=3.0, n=80):
        """Surface MSE(theta_1, theta_2) autour de l'optimum (sans biais)."""
        _opt = np.linalg.lstsq(features, target, rcond=None)[0]
        _a = np.linspace(_opt[0] - span, _opt[0] + span, n)
        _b = np.linspace(_opt[1] - span, _opt[1] + span, n)
        _T1, _T2 = np.meshgrid(_a, _b)
        _pred = _T1[..., None] * features[:, 0] + _T2[..., None] * features[:, 1]
        _Z = ((_pred - target) ** 2).mean(axis=2)
        return _T1, _T2, _Z, _opt

    T1_raw, T2_raw, Z_raw, opt_raw = _cost_grid(X_raw, y_std, span=3.0)
    T1_sc, T2_sc, Z_sc, opt_sc = _cost_grid(X_scaled, y_std, span=3.0)

    # Avant / après pour quantifier l'effet
    K_raw = np.linalg.cond(X_raw.T @ X_raw)
    K_sc = np.linalg.cond(X_scaled.T @ X_scaled)
    print(f"--- Conditionnement de XᵀX --- \nbrut        : {K_raw:.1f} \nstandardisé : {K_sc:.2f}")
    return (
        K_raw,
        K_sc,
        StandardScaler,
        T1_raw,
        T1_sc,
        T2_raw,
        T2_sc,
        Z_raw,
        Z_sc,
        opt_raw,
        opt_sc,
    )


@app.cell(hide_code=True)
def _(
    K_raw,
    K_sc,
    T1_raw,
    T1_sc,
    T2_raw,
    T2_sc,
    Z_raw,
    Z_sc,
    opt_raw,
    opt_sc,
    plt,
):
    _fig = plt.figure(figsize=(8, 6))

    # Ligne du haut : features brutes (cuvette en vallée, mauvais conditionnement)
    _ax1 = _fig.add_subplot(2, 2, 1, projection="3d")
    _ax1.plot_surface(T1_raw, T2_raw, Z_raw, cmap="viridis", alpha=0.7, linewidth=0, antialiased=True)
    _ax1.scatter([opt_raw[0]], [opt_raw[1]], [Z_raw.min()], color="gold", marker="*", s=180)
    _ax1.set_xlabel(r"$\theta_1$"); _ax1.set_ylabel(r"$\theta_2$"); _ax1.set_zlabel("MSE")
    _ax1.view_init(elev=30, azim=-60)
    _ax1.set_title("Surface de coût - données brutes")

    _ax2 = _fig.add_subplot(2, 2, 2)
    _ax2.contourf(T1_raw, T2_raw, Z_raw, levels=30, cmap="viridis")
    _ax2.contour(T1_raw, T2_raw, Z_raw, levels=12, colors="white", linewidths=0.4, alpha=0.5)
    _ax2.scatter([opt_raw[0]], [opt_raw[1]], color="gold", marker="*", s=200, edgecolor="black", lw=0.5)
    _ax2.set_xlabel(r"$\theta_1$"); _ax2.set_ylabel(r"$\theta_2$", rotation=0)
    _ax2.set_title(f"Lignes de niveau (brut) - $K(X^TX)$ = {K_raw:.0f}")

    # Ligne du bas : features standardisées (cuvette arrondie, bon conditionnement)
    _ax3 = _fig.add_subplot(2, 2, 3, projection="3d")
    _ax3.plot_surface(T1_sc, T2_sc, Z_sc, cmap="viridis", alpha=0.7, linewidth=0, antialiased=True)
    _ax3.scatter([opt_sc[0]], [opt_sc[1]], [Z_sc.min()], color="gold", marker="*", s=180)
    _ax3.set_xlabel(r"$\theta_1$"); _ax3.set_ylabel(r"$\theta_2$"); _ax3.set_zlabel("MSE")
    _ax3.view_init(elev=30, azim=-60)
    _ax3.set_title("Surface de coût - après StandardScaler")

    _ax4 = _fig.add_subplot(2, 2, 4)
    _ax4.contourf(T1_sc, T2_sc, Z_sc, levels=30, cmap="viridis")
    _ax4.contour(T1_sc, T2_sc, Z_sc, levels=12, colors="white", linewidths=0.4, alpha=0.5)
    _ax4.scatter([opt_sc[0]], [opt_sc[1]], color="gold", marker="*", s=200, edgecolor="black", lw=0.5)
    _ax4.set_xlabel(r"$\theta_1$"); _ax4.set_ylabel(r"$\theta_2$", rotation=0)
    _ax4.set_title(f"Lignes de niveau (standardisé) - $K(X^TX)$ = {K_sc:.2f}")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Mais ducoup, cette transformation détruit l'interpretabilité ?

    - Concernant les coefficients estimés : non, au contraire. L'ordre de grandeur des coefficients était initialement corrélé à l'importance des features associées mais aussi à l'ordre de grandeur de ces mêmes features. **La standardisation permet d'interpréter l'importance des paramètres**.
    - Concernant les variables (features) : non plus. La standardisation est réversible (le transformer `StandardScaler` dispose de la méthode `inverse_transform`) et ne s'applique pas à la feature cible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## B. Batch Gradient Descent (BGD)

    ### Pourquoi « batch » ? <a id="pourquoi-batch"></a>

    On parle de _Batch Gradient Descent_ lorsque le gradient $\nabla\mathrm{MSE}(\boldsymbol{\theta}) \in \mathcal{M}_{n,1}(\mathbb{R})$ est calculé en utilisant **l'intégralité des instances du training set**. On rappelle son expression :

    $$\nabla\mathrm{MSE}(\boldsymbol{\theta}) = \frac{2}{m}\,\mathbf{X}^{\mathsf T}(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}).$$

    Avec cette version _batch_, on obtient le gradient moyen exact, donc la direction de descente la plus fidèle, au prix de toucher aux $m$ lignes à chaque itération.

    > Le chapitre 1 introduit le concept de « batch learning » (en opposition au « online learning »). Levons immédiatement toute ambiguité : les deux notions n'ont rien à voir !

    <a id="mini-batch"></a>Cette implémentation est la plus _naturelle_ - c'est celle qu'on a présenté jusque-là - mais on peut procéder autrement : on peut par exemple **se contenter de 10 instances, différentes à chaque itération de la descente**. En notant $p=n+1$ le nombre de paramètres du modèle, on obtient :

    $$y \in \mathcal{M}_{m, 1}(\mathbb{R}) \rightarrow y \in \mathcal{M}_{10, 1}(\mathbb{R})$$

    $$X \in \mathcal{M}_{m, p}(\mathbb{R}) \rightarrow X \in \mathcal{M}_{10, p}(\mathbb{R})$$

    Par conséquent

    $$\nabla MSE(\theta) = \frac{2}{10} \underbrace{X^T \vphantom{(X\theta - y)}}_{\in \mathcal{M}_{p, 10}(\mathbb{R})} \underbrace{(X\theta - y)}_{\in \mathcal{M}_{10, 1}(\mathbb{R})}$$

    Finalement,

    $$\nabla MSE(\theta) \in \mathcal{M}_{p, 1}(\mathbb{R})$$

    Le gradient représente alors toujours une **direction dans l'espace à** $n+1$ **dimensions des paramètres**. Avec cette approche, la direction de descente est **moins représentative** de l'erreur moyenne du jeu de données : elle subit davantage le **bruit intrinsèque** de la dizaine d'instances échantillonnées, mais **le gradient est plus rapide à calculer**.

    On abordera un peu plus tard les principales variantes de la descente (stochastique et mini-batch) qui implémentent cette optimisation.

    > « batch gradient descent » est souvent traduit « descente de gradient par lots ».

    ### Implémentation explicite

    On implémente très simplement la BGD en suivant [les étapes présentées ici](#etapes_BGD).

    On choisit d'itérer $1000$ fois et on fixe le learning rate à $0.1$.
    """)
    return


@app.cell
def _(X_b, np, y):
    eta_grad = 0.1  # learning rate
    _n_epochs = 1000 
    m_grad = len(X_b)  # nombre d'instances

    _rng = np.random.default_rng(seed=42)    
    theta_grad = _rng.standard_normal((2, 1))  # initialisation aléatoire de la descente (theta_0)

    for _epoch in range(_n_epochs):
        _gradients = 2 / m_grad * X_b.T @ (X_b @ theta_grad - y)
        theta_grad = theta_grad - eta_grad * _gradients

    print("--- Batch Gradient Descent (BGD) ---")
    print(f"Ordonnée à l'origine estimée : {theta_grad[0,0]:.3f} \nPente estimée : {theta_grad[1,0]:.3f}")
    return (theta_grad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    L'estimation des paramètres coïncide au millième près avec [celle de la SVD](#prediction-svd).

    ---
    ## C. Learning rate

    Dans la version _batch_ de la descente de gradient, le learning rate $\eta$ est indépendant des itérations. Concrètement, dans l'équation $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\,\nabla\mathrm{MSE}(\boldsymbol{\theta}_k)$ , le pas $\eta$ ne dépend pas de $k$.

    Si on en a déjà dit un mot [ici](#difficultes), cette section propose d'étudier plus concrètement comment ce paramètre affecte la descente.

    ### Visualisation

    La figure ci-dessous illustre l'effet du learning rate sur la vitesse de convergence : les 20 droites de régression correspondent aux 20 premières itérations de la descente.

    > La droite la plus claire correspond à $\boldsymbol{\theta}_0$ , la plus foncée correspond à $\boldsymbol{\theta}_{20}$. <a id="visualisation-learning-rate"></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    eta_slider = mo.ui.slider(
        start=0.01, stop=0.55, step=0.01, value=0.43,
        label=r"$\eta$ (learning rate)", show_value=True,
    )
    eta_slider
    return (eta_slider,)


@app.cell(hide_code=True)
def _(X, X_b, X_new, X_new_b, bleu, eta_slider, np, plt, y):
    import matplotlib as mpl

    def plot_gradient_descent(theta, eta, n_epochs=1000, n_shown=20):
        _m = len(X_b)
        _theta = theta.copy()                       # ne pas écraser l'origine partagée
        _fig, _ax = plt.subplots(figsize=(6, 4))
        _ax.plot(X, y, ".", color=bleu)
        for _epoch in range(n_epochs):
            if _epoch < n_shown:
                _y_predict = X_new_b @ _theta
                _color = mpl.colors.rgb2hex(plt.cm.OrRd(_epoch / n_shown + 0.15))
                _ax.plot(X_new, _y_predict, linestyle="solid", color=_color)
            _gradients = 2 / _m * X_b.T @ (X_b @ _theta - y)
            _theta = _theta - eta * _gradients
        _ax.set_xlabel("$x_1$")
        _ax.set_ylabel("$y$", rotation=0)
        _ax.axis([0, 2, 0, 15])
        _ax.grid()
        _ax.set_title(fr"$\eta = {eta}$")
        return _fig

    # Origine fixe : le même theta_0 quel que soit eta, pour que la comparaison soit honnête
    theta_reg_plot = np.random.default_rng(42).standard_normal((2, 1))

    plot_gradient_descent(theta_reg_plot, eta_slider.value)
    return (mpl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En jouant avec le curseur, l'algorithme de descente semble converger plus rapidement quand $\eta \approx 0.22$.

    ### Optimisation empirique

    On va essayer de fine-tuner le learning rate $\eta$ avec une approche grid-search. On décide de retenir comme critère de sélection le **nombre d'itérations** requises pour que la descente s'approche à $10^{-4}$ près du minimum théorique.

    > On va tester 300 valeurs de $\eta$ dans $\left[ 0.01, 0.55 \right]$. Ça requiert de réaliser autant de descentes de gradient, que l'on plafonne à 10 000 itérations. On s'arrête lorsqu'on dépasse ce plafond ou qu'on est assez proche du minimum théorique : $\|\theta_k-\hat\theta\|<10^{-4}$.
    """)
    return


@app.cell(hide_code=True)
def _(X_b, gd_opt, np, y):
    def epochs_to_converge(theta0, eta, n_epochs=10000, seuil=1e-4):
        _theta = theta0.copy()
        for _epoch in range(1, n_epochs + 1):
            _grad = 2 / len(X_b) * X_b.T @ (X_b @ _theta - y)
            _theta = _theta - eta * _grad
            if not np.isfinite(_theta).all():
                return np.nan, np.inf                     
            if np.linalg.norm(_theta - gd_opt) < seuil:
                return _epoch, _theta
        return np.nan, _theta

    _theta0 = np.random.default_rng(42).standard_normal((2, 1))
    etas = np.linspace(0.01, 0.55, 300) # Grille de GridSearch

    with np.errstate(over="ignore", invalid="ignore"): # évite les warnings quand la descente diverge
        epochs = np.array([epochs_to_converge(_theta0, _e)[0] for _e in etas])

    i = np.nanargmin(epochs)
    # print(f"L{etas[i]:.3f}")
    return epochs, etas, i


@app.cell(hide_code=True)
def _(epochs, etas, i, plt, rouge, vert):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(etas, epochs, color=rouge, lw=2)
    _ax.scatter(etas[i], epochs[i], color=vert, s=70, zorder=5,
                label=f"optimum ($\\eta$={etas[i]:.3f})")
    _ax.set_xlabel(r"$\eta$ (learning rate)")
    _ax.set_ylabel(r"itérations avant $\|\theta_k-\hat\theta\|<10^{-4}$")
    _ax.grid(alpha=0.3); _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Une optimisation utile ?

    Enfin, il faut souligner que ce fine-tuning sert essentiellement un but pédagogique ; tel quel il n'est pas très utile.

    1. D'abord parce que la condition $\|\theta_k-\hat\theta\|<10^{-4}$ requiert que $\hat\theta$ soit connu via l'équation normale, or la descente n'a d'intérêt que si $\hat\theta$ n'existe pas déjà.

    2. Ensuite parce que dans le cadre de la régression, la fonction d'erreur (MSE) a la particularité d'être convexe : tout $\eta$ qui ne fasse pas diverger la descente la conduit à converger vers le même $\hat\theta$. On cherche donc uniquement à accélerer l'étape d'entraînement du modèle - mais pour trouver $\eta = 0.409$, on a dû réaliser 300 descentes de gradient. On n'a pas accéléré grand chose.

    > Pour un modèle **non convexe**, $\eta$ peut **changer** le minimum atteint. Il devient alors un vrai hyperparamètre de généralisation, à optimiser avec `GridSearchCV`.

    ---

    ## D. Implémentation avec Scikit-Learn

    Il n'y en a pas ! La section suivante compare la complexité computationnelle entre la solution en forme close (équation normale / décomposition SVD). Dans le contexte de la régression linéaire - et d'autant plus si $m \gg n$, ce qui est fréquent - la version batch de la descente de gradient n'a pas beaucoup d'intérêt.

    ---
    ## E. Complexité computationnelle

    ### BGD

    À chaque itération de la descente de gradient, on doit calculer :

    $$\nabla\mathrm{MSE}(\boldsymbol{\theta}) = \frac{2}{m}\,\mathbf{X}^{\mathsf T}(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}).$$

    Cela nécessite $O(mn)$ pour $X\theta$ et $O(mn)$ pour $X^T r$, donc $O(mn)$ par itération et $O(kmn)$ après $k$ itérations.

    **Complexité totale** : $O(kmn)$.

    ### SVD vs. BGD

    On a vu [un peu plus tôt](#complexite-svd-en) que la SVD a une complexité moyenne de $O(mn^2)$ quand $m \gg n$. Si on fait le bilan :
    $$\text{SVD} \approx O(mn^2)$$
    $$\text{BGD} \approx O(kmn)$$
    En simplifiant par $m n$ :
    $$\text{BGD meilleure si } k < n$$

    > Ce résultat est un peu grossier, il met notamment de côté les optimisations des bibliothèques de calcul et le conditionnement des données.

    En _théorie_ donc, la BGD fait mieux à condition qu'elle réalise **moins d'itérations qu'il n'y a de prédicteurs**. C'est un résultat un peu décevant, surtout que la SVD nous donne le minimum exact théorique, pas une approximation.

    Les autres implémentations de la descente de gradient (stochastique, mini-batch) permettent heureusement de réduire drastiquement la complexité. Surtout, la comparaison avec la SVD n'a d'intérêt que pour la régression linéaire : de nombreux modèles (régression logistique, réseaux de neurones) **n'admettent pas de solution en forme close**.

    La descente de gradient que l'on introduit ici va nous servir dans une bonne partie des chapitres à venir.

    ---

    ## F. Stochastic Gradient Descent (SGD)

    La descente de gradient **stochastique** propose une heuristique pour réduire le coût de calcul de la descente de gradient. On l'a vu [quand on a présenté la version batch](#pourquoi-batch), il est possible d'évaluer le gradient de la fonction de coût sur un nombre restreint d'instances.

    Cette approche consiste simplement à estimer le gradient sur **une seule instance tirée au hasard** :

    $$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta\,\nabla MSE\big(\boldsymbol{\theta}_k\,;\,\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\big), \qquad i \text{ tiré aléatoirement.}$$

    > Puisqu'on tire une instance différente à chaque itération $k$ de la descente, on devrait plutôt noter leur indice $i_k$

    On remplace donc le gradient exact par un estimateur, construit à partir d'une seule observation : on perd en précision mais **le coût d'une itération passe de $O(mn)$ à $O(n)$**.

    La section qui suit justifie mathématiquement l'importance de réaliser ces tirages **sans remise**.

    /// details | Tirages sans remise - démonstration (pour aller plus loin)

    Soit $X_k$ la variable aléatoire réelle modélisant l'indice de l'instance sélectionnée à la $k$-ème itération de la descente.

    $$X_k \sim \mathcal{U}(\llbracket 1, m \rrbracket) \quad , \quad m \text{ nombre total d'instances}$$

    Si l'on suppose que **les tirages s'effectuent avec remise**, les variables aléatoires $X_1, X_2, \dots, X_N$ sont I.I.D.

    Soit $i \in \llbracket 1, m \rrbracket$ l'indice d'une instance donnée. En supposant que l'on itère $N=m$ fois (une époque), la probabilité que cette instance ne soit jamais sélectionnée s'écrit :

    $$\mathbb{P} \left( \bigcap_{k=1}^m (X_k \neq i) \right) \overset{\text{I.I.D}}{=} \prod_{k=1}^m \mathbb{P}(X_k \neq i) = \left( \frac{m-1}{m} \right)^m .$$

    Soit $Z$ la variable aléatoire qui compte le nombre d'instances jamais sélectionnées au terme d'une époque.

    $$Z = \sum_{j=1}^m Y_j \quad , \quad Y_j = \begin{cases} 1 \text{ si } j \text{ n'est jamais sélectionnée} \\ 0 \text{ sinon.} \end{cases}$$

    Par conséquent, $\displaystyle  \forall j \in \llbracket 1, m \rrbracket$, on a $\displaystyle Y_j \sim \mathcal{B} \left( \left( \frac{m-1}{m} \right)^m \right)$. Calculons l'espérance de $Z$ :

    $$\mathbb{E}(Z) = \sum_{j=1}^m \mathbb{E}(Y_j) = m\left( \frac{m-1}{m} \right)^m$$

    En proportion, $\frac{\mathbb{E}(Z)}{m} = \left( \frac{m-1}{m} \right)^m \underset{m \to +\infty}{\sim} e^{-1} \approx$ **$37 \%$ des instances sont ignorées**.

    C'est tout l'intérêt de faire des tirages sans remise : chaque instance est vue exactement une fois par époque, aucune observation ne peut être surreprésentée ou ignorée à cause du hasard. La **variance** introduite par l'échantillonnage sur **l'estimation du gradient** s'en trouve réduite, ce qui conduit généralement à une convergence plus **stable** et plus **rapide**.

    ///

    ### Notion d'époque

    Cette problématique de la remise permet aussi de bien comprendre la notion d'**époque** - qui n'est pas propre à la version _stochastique_ de la descente.

    Pour compter les itérations d'une descente de gradient, on utilise l'équivalent training set de sorte que :

    $$ \text{une époque  = } m \text{ itérations.}$$

    Dans le cas de la descente stochastique, si on fait des tirages sans remise :

    $$ \text{une époque = chaque instance a été vue exactement une fois.} $$


    ### Visualisations

    On propose plusieurs visualisations pour bien comprendre ce qu'il se passe avec cette version stochastique.

    À chaque itération, on tire une seule instance $(\mathbf{x}^{(i)}, y^{(i)})$ au hasard. Le segment rouge,  qui correspond à son résidu $\epsilon^{(i)} = y^{(i)} - \hat y^{(i)}$, suffit à estimer le gradient :

    $$\nabla MSE\big(\boldsymbol{\theta}_k\,;\,\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\big) = \frac{2}{m}\,\mathbf{X}^{\mathsf T}(\mathbf{X}\boldsymbol{\theta}_k - \mathbf{y})= \frac{2}{1}\,\mathbf{x}^{(i)}\big(\mathbf{x}^{(i)\mathsf T}\boldsymbol{\theta}_k - y^{(i)}\big) = 2\big(\hat y^{(i)} - y^{(i)}\big)\,\mathbf{x}^{(i)} = -\,2\,\epsilon^{(i)}\,\mathbf{x}^{(i)}$$

    La droite courante (en orange) se déplace de $-\eta\nabla MSE$, elle est comme « tirée » par ce résidu.

    > Le bouton « Itération suivante » sépare les deux temps : d'abord la sélection du point, puis la mise à jour des paramètres sur ce seul résidu.
    """)
    return


@app.cell(hide_code=True)
def _(N_2D, mo, np, x_2d, y_2d):
    sgd_lr = 0.1                  # learning rate choisi pour la lisibilité pédagogique
    sgd_theta_init = (0.0, 0.0)   # départ volontairement éloigné de l'optimum
    SGD_MAX_STEPS = 200

    _seq_rng = np.random.default_rng(7)
    _sequence = []
    while len(_sequence) < SGD_MAX_STEPS:
        _sequence.extend(_seq_rng.permutation(N_2D).tolist())
    SGD_POINT_SEQUENCE = _sequence[:SGD_MAX_STEPS]

    def sgd_reset_state():
        return {"theta": sgd_theta_init, "theta_prev": sgd_theta_init,
                "point": None, "step": 0, "phase": "start"}

    def sgd_advance(s):
        if s["phase"] in ("start", "apply"):
            if s["step"] >= SGD_MAX_STEPS:
                return s
            _i = SGD_POINT_SEQUENCE[s["step"]]
            return {"theta": s["theta"], "theta_prev": s["theta"],
                    "point": _i, "step": s["step"], "phase": "select"}
        _i = s["point"]
        _t0, _t1 = s["theta"]
        _xi, _yi = x_2d[_i], y_2d[_i]
        _err = (_t0 + _t1 * _xi) - _yi  
        _g0, _g1 = _err, _err * _xi     
        return {"theta": (_t0 - sgd_lr * _g0, _t1 - sgd_lr * _g1),
                "theta_prev": (_t0, _t1),
                "point": _i, "step": s["step"] + 1, "phase": "apply"}

    get_sgd, set_sgd = mo.state(sgd_reset_state())

    sgd_next = mo.ui.button(label="Itération suivante  ▶",
                            on_change=lambda _: set_sgd(sgd_advance))
    sgd_reset = mo.ui.button(label="↺  Réinitialiser",
                             on_change=lambda _: set_sgd(sgd_reset_state()))

    mo.hstack([sgd_next, sgd_reset])
    return (get_sgd,)


@app.cell(hide_code=True)
def _(bleu, get_sgd, np, orange, plt, rouge, theta_opt_2d, vert, x_2d, y_2d):
    _s = get_sgd()
    _t0, _t1 = _s["theta"]
    _p0, _p1 = _s["theta_prev"]
    _i = _s["point"]
    _step = _s["step"]
    _phase = _s["phase"]

    _y_pred = _t0 + _t1 * x_2d
    _mse = float(np.mean((y_2d - _y_pred) ** 2))
    _xs = np.array([x_2d.min() - 0.1, x_2d.max() + 0.1])
    _gris = "0.55"

    _fig, _ax = plt.subplots(figsize=(7, 4.7))
    _ax.scatter(x_2d, y_2d, color=bleu, s=35, alpha=0.8, zorder=3, label="Données")

    _b0, _b1 = theta_opt_2d
    _ax.plot(_xs, _b0 + _b1 * _xs, color=vert, lw=1.8, ls="--", alpha=0.85,
             zorder=4, label=fr"Optimale : ${_b0:.2f} + {_b1:.2f}\,x$")

    if _phase == "apply":
        _ax.plot(_xs, _p0 + _p1 * _xs, color=_gris, lw=1.6, ls=":", alpha=0.9,
                 zorder=4, label="Droite avant le pas")

    _ax.plot(_xs, _t0 + _t1 * _xs, color=orange, lw=2.6, zorder=5,
             label=fr"Droite courante : ${_t0:.2f} + {_t1:.2f}\,x$")

    if _i is not None:
        _xi, _yi = x_2d[_i], y_2d[_i]
        _y_ref = (_p0 + _p1 * _xi) if _phase == "apply" else (_t0 + _t1 * _xi)
        _res = _yi - _y_ref
        _ax.plot([_xi, _xi], [_yi, _y_ref], color=rouge, lw=2.4, zorder=6,
                 label="Résidu sélectionné")
        _ax.scatter([_xi], [_yi], s=90, facecolor=rouge, edgecolor="black",
                    linewidth=1.0, zorder=7)

    if _phase == "start":
        _titre = "État initial : cliquez sur « Itération suivante » pour sélectionner un point"
    elif _phase == "select":
        _titre = (fr"Itération {_step + 1} · SÉLECTION : point #{_i} ($x={_xi:.2f}$), "
                  fr"résidu $= {_res:+.2f}$" "\n"
                  "▶ cliquez pour appliquer le pas de gradient sur ce résidu")
    else:  # apply
        _dt0, _dt1 = _t0 - _p0, _t1 - _p1
        _titre = (fr"Itération {_step} · MISE À JOUR sur le résidu $= {_res:+.2f}$" "\n"
                  fr"$\Delta\theta_0={_dt0:+.3f}$,  $\Delta\theta_1={_dt1:+.3f}$ "
                  fr"   |   MSE $= {_mse:.3f}$")

    _ax.set_xlim(_xs[0], _xs[1])
    _ax.set_ylim(-1, 13)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_title(_titre, fontsize=10)
    _ax.legend(loc="upper left", fontsize=8)
    _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En raison de sa nature aléatoire, la descente de gradient stochastique est plus hasardeuse : la fonction de coût évolue beaucoup plus irrégulièrement, **elle ne décroit qu'en moyenne.**

    La figure suivante compare l'évolution la MSE au fil des itérations pour une descente batch et une descente stochastique.

    > Plus précisément, on représente l'évolution de $\mathrm{MSE}(\theta_k) - \mathrm{MSE}(\hat{\theta})$. La descente de gradient étant garantie de converger, on s'attend à ce que ces deux courbes se rapprochent de 0.
    """)
    return


@app.cell(hide_code=True)
def _(X_b, gd_opt, np, y):
    _m = len(X_b)
    _n_epochs = 50
    _theta_init = np.random.default_rng(0).standard_normal((2, 1))
    _mse = lambda th: float(((X_b @ th - y) ** 2).mean())
    mse_opt = _mse(gd_opt)                       # MSE irréductible (atteinte en θ̂)

    # Batch : un gradient sur tout le dataset par époque
    _th = _theta_init.copy()
    mse_batch = [_mse(_th)]
    for _ in range(_n_epochs):
        _th = _th - 0.1 * (2 / _m * X_b.T @ (X_b @ _th - y))
        mse_batch.append(_mse(_th))

    # SGD : _m mises à jour par époque sur une instance tirée au hasard
    # learning schedule η(t) = t0 / (t + t1) ; η(0) = 0.1 comme le batch
    _t0, _t1 = 5, 50
    _th = _theta_init.copy()
    _sg = np.random.default_rng(42)
    mse_sgd, ep_axis = [_mse(_th)], [0.0]
    for _epoch in range(_n_epochs):
        for _i in range(_m):
            _j = _sg.integers(_m)
            _xi, _yi = X_b[_j:_j+1], y[_j:_j+1]
            _eta_t = _t0 / (_epoch * _m + _i + _t1)
            _th = _th - _eta_t * (2 * _xi.T @ (_xi @ _th - _yi))
            mse_sgd.append(_mse(_th))
            ep_axis.append((_epoch * _m + _i + 1) / _m)

    mse_batch, mse_sgd, ep_axis = np.array(mse_batch), np.array(mse_sgd), np.array(ep_axis)
    return ep_axis, mse_batch, mse_opt, mse_sgd


@app.cell(hide_code=True)
def _(ep_axis, mse_batch, mse_opt, mse_sgd, plt, rouge, violet):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    _ax1.plot(range(len(mse_batch)), mse_batch - mse_opt, color=violet, lw=1.8)
    _ax1.set_title("Descente de gradient batch")
    _ax1.set_ylabel(r"$\mathrm{MSE}(\theta_k) - \mathrm{MSE}(\hat{\theta})$")

    _ax2.plot(ep_axis, mse_sgd - mse_opt, color=rouge, lw=0.7)
    _ax2.set_title("Descente de gradient stochastique")

    for _ax in (_ax1, _ax2):
        _ax.set_yscale("log")
        _ax.set_xlabel("époque")
        _ax.grid(alpha=0.3, which="both")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut aussi visualiser la trajectoire de la descente. Supposons que l'on compare les deux figures pour un même nombre d'époques :
    - Descente batch : une époque <-> une itération
    - Descente stochastique : une époque <-> $m=200$ itérations

    On ne peut pas faire de comparaison honnête avec cette hypothèse : il vaut mieux fixer un même nombre d'itérations pour les deux figures.
    """)
    return


@app.cell(hide_code=True)
def _(X_b, gd_opt, np, plt, y):
    # --- Trajectoires batch vs SGD sur les lignes de niveau de la MSE ---
    _m          = len(X_b)
    _n_epochs   = 50
    _n_sgd      = 50                                              # nb de pas SGD affichés (ajustable)
    _theta_init = np.random.default_rng(0).standard_normal((2, 1))  # même origine que la cellule précédente

    # Reconstruction des deux chemins (on stocke θ à chaque pas)
    _th = _theta_init.copy(); _batch = [_th.ravel().copy()]
    for _ in range(_n_epochs):
        _th = _th - 0.1 * (2 / _m * X_b.T @ (X_b @ _th - y))
        _batch.append(_th.ravel().copy())
    _batch = np.array(_batch)

    _th = _theta_init.copy(); _sg = np.random.default_rng(42); _sgd = [_th.ravel().copy()]
    for _s in range(_n_sgd):
        _j = _sg.integers(_m); _xi, _yi = X_b[_j:_j+1], y[_j:_j+1]
        _th = _th - (5 / (_s + 50)) * (2 * _xi.T @ (_xi @ _th - _yi))
        _sgd.append(_th.ravel().copy())
    _sgd = np.array(_sgd)

    # Surface MSE : exacte (MSE quadratique) -> MSE(θ) = MSE(θ̂) + (θ-θ̂)ᵀ (XᵀX / m) (θ-θ̂)
    _A   = (1 / _m) * X_b.T @ X_b
    _pts = np.vstack([_batch, _sgd, gd_opt.ravel()])
    _lo, _hi, = _pts.min(0), _pts.max(0); _pad = 0.15 * (_hi - _lo) + 0.2
    _g0 = np.linspace(_lo[0] - _pad[0], _hi[0] + _pad[0], 200)
    _g1 = np.linspace(_lo[1] - _pad[1], _hi[1] + _pad[1], 200)
    _TH0, _TH1 = np.meshgrid(_g0, _g1)
    _d0, _d1   = _TH0 - gd_opt[0, 0], _TH1 - gd_opt[1, 0]
    _Z = float(((X_b @ gd_opt - y) ** 2).mean()) + _A[0,0]*_d0**2 + 2*_A[0,1]*_d0*_d1 + _A[1,1]*_d1**2

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for _ax, _path, _col, _title in [
        (_axes[0], _batch, "navy",    f"Batch — {len(_batch)-1} époques"),
        (_axes[1], _sgd,   "crimson", f"SGD — {len(_sgd)-1} pas"),
    ]:
        _ax.contourf(_TH0, _TH1, _Z, levels=30, cmap="viridis")
        _ax.contour(_TH0, _TH1, _Z, levels=12, colors="white", linewidths=0.4, alpha=0.5)
        _ax.plot(_path[:, 0], _path[:, 1], color=_col, lw=1.4, marker="o", ms=6, zorder=5)
        _ax.scatter(_path[0, 0], _path[0, 1], color="black", marker="s", s=50, zorder=6, label="origine")
        _ax.scatter(gd_opt[0, 0], gd_opt[1, 0], color="gold", marker="*", s=200,
                    edgecolor="black", lw=0.5, zorder=7, label="minimum")
        _ax.set_xlabel(r"$\theta_0$"); _ax.set_title(_title)
        _ax.legend(loc="upper right", fontsize=8)
    _axes[0].set_ylabel(r"$\theta_1$", rotation=0)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Si, pour les fonctions de coût irrégulières, l'aspect "hasardeux" de la descente stochastique lui donne l'**avantage** de pouvoir s'extirper des minima locaux là où la BGD échoue, nous rencontrons ici un **inconvénient** typique de la descente stochastique : l'**impossibilité à converger au voisinage du minimum**.

    La section suivante présente une solution à ce problème.

    ### Learning schedule

    Pour permettre à la SGD de converger, cette approche propose de faire évoluer le learning rate au fil des itérations : $\eta$ devient $\eta_k$.

    $$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta_k\,\nabla\mathrm{MSE}(\boldsymbol{\theta}_k)$$

    Le principe est le suivant :
    1. Initialisation : on initialise $\eta_0$ avec une valeur assez grande, de sorte à ne pas rester coincé dans une crevasse à proximité de $\boldsymbol{\theta}_{0}$.
    2. Itération : on réduit progressivement le pas. La fonction qui fixe la décroissance de $\eta_k$ à chaque itération $k$ s'appelle le **learning schedule**.

    > On trouve aussi l'appellation explicite « learning _rate_ schedule ».

    Quelques points de vigilance :
    - Si le learning rate décroît trop rapidement, la descente risque de s'éterniser.
    - Pour les fonctions de coût non convexes, une décroissance trop rapide / trop lente risque respectivement de coincer l'algorithme dans un minimum local ou de lui faire sursauter le minimum global. Dans les deux cas on peut aboutir à une solution $\hat{\boldsymbol{\theta}}$ sous optimale.

    On décide d'implémenter la fonction de learning schedule suivante :

    $$\eta_k = \frac{1}{\alpha(k + t_0)}$$

    Où $\alpha$ et $t_0$ sont des constantes.

    > Cette fonction correspond au paramètre `learning_rate="optimal"` de `SGDRegressor` et `SGDClassifier`.

    La SGD de la figure mobilise le code-ci dessous (on exécute plus bas une version moins lisible - on y rajoute la génération des droites - mais l'algorithme est le même, vérifiez !).

    ```python
    n_epochs = 50
    t0, t1 = 5, 50  # hyperparamètres du learning schedule

    def learning_schedule(t):
        return t0 / (t + t1)

    _rng = np.random.default_rng(seed=42)
    theta = _rng.standard_normal((2, 1))  # initialisation aléatoire de theta_0

    for epoch in range(n_epochs):
        shuffled_indices = rng.permutation(m) # pour faire des tirages SANS remise
        iteration = 0
        for idx in shuffled_indices:
            xi = X_b[idx:idx+1]
            yi = y[idx:idx+1]
            gradients = 2 * xi.T @ (xi @ theta - yi)  # SGD : on divise par la taille de l'échantillon (=1)
            eta = learning_schedule(epoch * m + iteration) # learning rate
            theta = theta - eta * gradients
            iteration += 1
    ```

    À nouveau, les 20 droites de régression correspondent aux 20 premières itérations de la descente.
    """)
    return


@app.cell(hide_code=True)
def _(X, X_b, X_new, X_new_b, bleu, m, mpl, np, plt, y):
    n_epochs = 50
    t0, t1 = 5, 50  # learning schedule hyperparameters

    def learning_schedule(t):
        return t0 / (t + t1)

    theta_path_sgd = [] 
    _rng = np.random.default_rng(seed=42)    
    theta_sgd = _rng.standard_normal((2, 1)) 

    n_shown = 20
    plt.figure(figsize=(6, 4)) 

    for epoch in range(n_epochs):
        shuffled_indices = _rng.permutation(m)
        iteration = 0
        for idx in shuffled_indices:

            if epoch == 0 and iteration < n_shown:
                _y_predict = X_new_b @ theta_sgd
                color = mpl.colors.rgb2hex(plt.cm.OrRd(iteration / n_shown + 0.15))
                plt.plot(X_new, _y_predict, color=color)

            xi = X_b[idx:idx+1]
            yi = y[idx:idx+1]
            gradients = 2 * xi.T @ (xi @ theta_sgd - yi)  # SGD : on divise par la taille de l'échantillon (=1)
            eta = learning_schedule(epoch * m + iteration) # learning rate
            theta_sgd    = theta_sgd - eta * gradients
            iteration += 1

            theta_path_sgd.append(theta_sgd) 

    plt.plot(X, y, ".", color=bleu)
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([0, 2, 0, 15])
    plt.grid()

    plt.show()
    return (theta_sgd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Le learning schedule n'est pas linéaire, et on voit bien que l'écart entre les droites ne diminue pas linéairement.

    > Pour comparer, jetez un oeil à [la même figure avec $\eta$ constant](#visualisation-learning-rate).

    ### Évaluation - SGD manuelle

    Notre SGD fraîchement implémentée nous fournit un $\hat{\boldsymbol{\theta}}_{\text{SGD}}$ ; comparons-le avec $\hat{\boldsymbol{\theta}}_{\text{BGD}}$.

    > On rappelle que la valeur de $\hat{\boldsymbol{\theta}}_{\text{BGD}}$ coïncide ici au millième près avec l'optimum théorique.
    """)
    return


@app.cell
def _(theta_grad, theta_sgd):
    print("--- Batch Gradient Descent (BGD) ---")
    print(f"Ordonnée à l'origine estimée : {theta_grad[0,0]:.3f} \nPente estimée                : {theta_grad[1,0]:.3f}\n")

    print("--- Stochastic Gradient Descent (SGD) ---")
    print(f"Ordonnée à l'origine estimée : {theta_sgd[0,0]:.3f} \nPente estimée                : {theta_sgd[1,0]:.3f}\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On obtient une estimation plutôt satisfaisante au regard du nombre d'itérations : la BGD a itéré sur **1000 époques** tandis que la SGD n'a itéré que sur **50 époques**.

    ### Implémentation avec Scikit-Learn

    La classe `SGDRegressor` de Scikit implémente la régression linéaire avec descente de gradient stochastique. On présente quelques paramètres utiles :
    - `loss` : fonction d'erreur à minimiser. Par défaut c'est la MSE.
    - `learning_rate` : fixe le learning schedule.
    - `eta0` : valeur initiale $\eta_0$ du learning rate.
    - Itérations : l'algorithme itère `n_iter_no_change` fois. Si à cette échéance la fonction de perte est inférieure à `tol`, l'algorithme s'arrête. Le cas contraire, il continue jusqu'à `max_iter` itérations.

    > Par défaut, `SGDRegressor` effectue des tirages sans remise. À chaque époque, les échantillons sont mélangés aléatoirement, puis traités séquentiellement. C'est pour cela que le paramètre associé s'appelle `shuffle=True` et pas `replacement=True`.

    <a id="adaptive"></a> Le `learning_rate` de `SGDRegressor` supporte plusieurs learning schedules, on en présente ici les 4 principaux. Chaque fonction mobilise des constantes différentes : ce sont aussi des paramètres de `SGDRegressor`.
    - `constant` : $\eta=\eta_0$
    - `optimal`: $\eta_k = \displaystyle\frac{1}{\alpha(k + t_0)}$, c'est le **mode par défaut** et $t_0$ est ajusté automatiquement (pas un paramètre).
    - `invscaling` : $\eta_k = \displaystyle\frac{\eta_0}{k^{\text{power\_t}}}$
    - `adaptive` :  $\eta$ est constant, mais dès que la descente échoue succesivement `n_iter_no_change` fois à réduire la fonction de perte, $\eta$ est divisé par $5$.

    > Le mode par défaut `optimal` offre de bonnes performances dans la plupart des cas.
    """)
    return


@app.cell
def _(X, y):
    from sklearn.linear_model import SGDRegressor

    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01, n_iter_no_change=100, random_state=42)
    sgd_reg.fit(X, y.ravel())

    print("--- SGD Scikit ---")
    print(f"Ordonnée à l'origine estimée : {sgd_reg.intercept_[0]:.3f} \nPente estimée                : {sgd_reg.coef_[0]:.3f}\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On s'approche encore de l'optimum théorique.

    ### La méthode `partial_fit()`

    On a utilisé `fit()` pour entraîner `SGDRegressor`, mais cet estimateur (ce n'est pas le cas de tous) dispose aussi de la méthode `partial_fit()`.

    Chaque appel à `partial_fit()` lance un entraînement sur **une seule époque**, en dépit des contraintes fixées par `max_iter` ou `tol`. On peut ainsi entraîner complètement le modèle en itérant les appels à cette méthode.

    > On y reviendra plus tard, mais il est aussi possible d'exécuter `partial_fit()` sur des instances inédites.

    Cette approche permet d'avoir un bon **contrôle** sur l'entraînement du modèle, éventuellement de **visualiser** l'évolution de la fonction de perte ou des coefficients du modèle.

    > `partial_fit()` ne réinitialise pas le compteur d'itérations (le $k$), sinon un learning schedule comme `optimal` perdrait tout son sens

    ### Le paramètre `warm_start`

    Supposons que nous ayons entraîné notre régresseur et que nous disposions d'un nouveau jeu de données. On voudrait pouvoir alimenter notre modèle **incrémentalement** avec ces nouvelles données, sans devoir le ré-entraîner sur le dataset initial : c'est l'approche **online learning**.

    À cet effet, Scikit propose le paramètre `warm_start`. Notre `SGDRegressor` dispose de cette option, mais ce n'est pas le cas de tous les estimateurs.

    Concrètement, lorsqu'on réalise un `fit()` sur un deuxième dataset :
    - le compteur d'itérations $k$ est réinitialisé
    - le learning schedule est réinitialisé
    - les critères d'arrêt (`tol`, `n_iter_no_change` et `max_iter`) son mémorisés mais les compteurs sont réinitialisés

    La surface de la fonction de coût est remodelée avec les nouvelles données, le `warm_start=True` permet essentiellement d'initialiser ce nouvel entraînement au point $\hat{\boldsymbol{\theta}}_{\text{0}}$ où s'était arrêté le premier.

    « Mais ducoup, si on présente au modèle un nouveau jeu de données beaucoup plus petit et bruité que le premier, **l'optimum peut se déplacer énormément au regard de la représentativité de ce nouveau dataset** ? »

    Et bien oui, et c'est la grosse limite de cette option `warm_start`.

    ### Online-learning

    Pour faire du online-learning correctement, la solution c'est de combiner `partial_fit()` et `warm_start=True`.

    Supposons que notre prédicteur ait `warm_start=True` et qu'on l'ait entraîné sur un premier dataset en ayant exécuté `partial_fit()` un **nombre suffisant de fois**.

    Quand un nouveau paquet de données se présente, on exécute `partial_fit()` sur ces instances inédites. Puisqu'un appel = une seule époque, on ne fait que **quelques pas de gradient** dans la direction suggérée par ce paquet, au lieu de **converger** dessus comme le ferait `fit()`.

    **Cette technique permet de contrôler efficacement l'influence de données entrantes sur la capacité de prédiction du modèle.**

    De sucroît, comme $k$ n'est pas réinitialisé, le learning schedule continue de décroître au fil des appels. Donc $\eta_k$ diminue, et **les données présentées tardivement ont de moins en moins d'influence **: le modèle se stabilise tout seul.

    ### Concept drift

    Il convient toutefois de préciser que cette décroissance n'est bénéfique que si la **distribution est stationnaire**.

    En pratique, il arrive souvent que les données entrantes indiquent un optimum différent non pas parce qu'elles sont trop bruitées, mais parce que la relation entre la variable cible et les prédicteurs évolue _vraiment_ dans le temps : c'est le **concept drift**.

    Dans ces circonstances, un learning schedule décroissant devient un piège : le modèle se fige progressivement et n'arrive plus à suivre un **optimum qui bouge**.

    La solution dans ce cas consiste généralement à imposer un **learning rate constant** (ou [`adaptive`](#adaptive)).

    ---

    ## G. Mini-Batch Gradient Descent

    Cette section présente une deuxième heuristique pour réduire le coût de la descente de gradient. Son principe est exactement celui qu'on a expliqué [ici](#mini-batch) : évaluer le gradient de la fonction de coût sur un **petit paquet d'instances**, un **mini-batch**, plutôt que de le calculer sur tout le training set (descente _batch_) ou sur une seule instance (descente _stochastique_).

    Cette approche bénéficie largement des avantages de la descente stochastique sur la version batch, et elle est même meilleure sur certains aspects :
    1. Puisqu'on mobilise un _petit_ nombre d'instances à chaque itération, on peut **paralléliser** les calculs ; ça rend cette approche compétitive vis-à-vis de la SGD en terme de coût.
    2. On s'affranchit du bruit intrinséque d'une seule instance : l'estimation du gradient a en moyenne une variance inférieure, **la descente est moins hasardeuse** (voir schéma ci-dessous).

    Le seul inconvénient de cette version mini-batch est qu'elle peut rencontrer un peu plus de difficultés à s'échapper d'une crevasse locale que la descente stochastique.
    """)
    return


@app.cell(hide_code=True)
def _(jaune, np, orange, plt, rouge):
    _data_rng = np.random.default_rng(seed=42)
    _m = 200
    _X = 2 * _data_rng.random((_m, 1))
    _y = 4 + 3 * _X + _data_rng.standard_normal((_m, 1))
    _X_b = np.c_[np.ones((_m, 1)), _X]   # équivaut à add_dummy_feature(_X)

    _theta = np.random.default_rng(seed=42).standard_normal((2, 1))
    _theta_path_bgd = []
    for _ in range(1000):
        _grad = 2 / _m * _X_b.T @ (_X_b @ _theta - _y)
        _theta = _theta - 0.1 * _grad
        _theta_path_bgd.append(_theta)
    _theta_path_bgd = np.array(_theta_path_bgd)

    _t0, _t1 = 5, 50
    _sgd_rng = np.random.default_rng(seed=42)
    _theta = _sgd_rng.standard_normal((2, 1))
    _theta_path_sgd = []
    for _epoch in range(50):
        for _iteration in range(_m):
            _ri = _sgd_rng.integers(_m)
            _xi = _X_b[_ri:_ri + 1]
            _yi = _y[_ri:_ri + 1]
            _grad = 2 * _xi.T @ (_xi @ _theta - _yi)        # SGD : pas de division par m
            _eta = _t0 / (_epoch * _m + _iteration + _t1)
            _theta = _theta - _eta * _grad
            _theta_path_sgd.append(_theta)
    _theta_path_sgd = np.array(_theta_path_sgd)

    _minibatch_size = 20
    _n_batches = int(np.ceil(_m / _minibatch_size))
    _T0, _T1 = 200, 1000
    _mgd_rng = np.random.default_rng(seed=42)
    _theta = _mgd_rng.standard_normal((2, 1))
    _theta_path_mgd = []
    for _epoch in range(50):
        _shuffled = _mgd_rng.permutation(_m)
        _X_b_shuffled = _X_b[_shuffled]
        _y_shuffled = _y[_shuffled]
        for _iteration in range(_n_batches):
            _idx = _iteration * _minibatch_size
            _xi = _X_b_shuffled[_idx:_idx + _minibatch_size]
            _yi = _y_shuffled[_idx:_idx + _minibatch_size]
            _grad = 2 / _minibatch_size * _xi.T @ (_xi @ _theta - _yi)
            _eta = _T0 / (_epoch * _n_batches + _iteration + _T1)
            _theta = _theta - _eta * _grad
            _theta_path_mgd.append(_theta)
    _theta_path_mgd = np.array(_theta_path_mgd)

    plt.figure(figsize=(6, 4))
    plt.plot(_theta_path_sgd[:, 0], _theta_path_sgd[:, 1], "-s", color = rouge, linewidth=1, label="Stochastic")
    plt.plot(_theta_path_mgd[:, 0], _theta_path_mgd[:, 1], "-+", color = orange, linewidth=2, label="Mini-batch")
    plt.plot(_theta_path_bgd[:, 0], _theta_path_bgd[:, 1], "-o", color = jaune, linewidth=3, label="Batch")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$   ", rotation=0)
    plt.axis([2.7, 4.5, 2.6, 3.7])
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # III. Régression polynomiale

    ## A. Principe général

    L'implémentation de la **régression polynomiale**, qui consiste à prédire une variable cible à partir d'un polynôme de ses prédicteurs, ne différe pas beaucoup de celle de la régression linéaire.

    En effet, si on dispose de deux prédicteurs $\mathbf{x}_{1}$ et $\mathbf{x}_{1}$, il suffit d'ajouter les vecteurs de features $\mathbf{x}_{1}²$, $\mathbf{x}_{1}³$, $\mathbf{x}_{2}²$, $\mathbf{x}_{2}³$, etc.. à la matrice de design puis de faire une régression linéaire à partir de ces nouvelles features.

    On génère un jeu de données non linéaire à un seul prédicteur, utile pour la suite.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(seed=42)
    _m = 200 

    X_polreg_raw = 6 * _rng.random((_m, 1)) - 3
    y_polreg = 0.5 * X_polreg_raw ** 2 + X_polreg_raw + 2 + _rng.standard_normal((_m, 1))
    return X_polreg_raw, y_polreg


@app.cell(hide_code=True)
def _(X_polreg_raw, bleu, plt, y_polreg):
    plt.figure(figsize=(6, 4))
    plt.plot(X_polreg_raw, y_polreg, ".",color=bleu)
    plt.title("Visualisation du jeu de données")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([-3, 3, 0, 10])
    plt.grid()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## B. Implémentation avec Scikit-Learn

    L'implémentation se fait en deux temps :
    1. `PolynomialFeatures` : On crée des features en mettant au carré, au cube etc.. les features initiales
    2. `LinearRegression` : On entraîne un modèle de régression linéaire classique avec ces nouveaux prédicteurs

    Dans le code ci-dessous, on précise `degree=2` lors de l'appel de `PolynomialFeatures`, de sorte qu'on ajoute seulement la feature $\mathbf{x}_{1}²$ (notre dataset initial n'a qu'un seul prédicteur).

    On a donc :

    $$\hat{\mathbf{y}} = \theta_0 + \theta_1 \mathbf{x}_{1} + \theta_2 \mathbf{x}_{1}² $$

    Ou encore $\hat{\boldsymbol{y}} = \mathbf{X} \boldsymbol{\theta}$, avec :

    $$\mathbf{X} = \begin{bmatrix} 1 & \mathbf{x}_1^{(1)} & \mathbf{x}_2^{(1)} \\ \vdots & \vdots & \vdots \\ 1 & \mathbf{x}_1^{(m)} & \mathbf{x}_2^{(m)} \end{bmatrix} \quad\text{où}\quad \mathbf{x}_{2}^{(i)}=\left( \mathbf{x}_{1}^{(i)} \right)^{2}$$
    """)
    return


@app.cell
def _(X_polreg_raw, mo):
    from sklearn.preprocessing import PolynomialFeatures

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_polreg_poly = poly_features.fit_transform(X_polreg_raw)

    mo.md(rf"$\mathbf{{x}}_1^{{(1)}} = {X_polreg_poly[0][0]:.3f}$ ; $\mathbf{{x}}_2^{{(1)}} = {X_polreg_poly[0][1]:.3f}$ ")
    return PolynomialFeatures, X_polreg_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On a plus qu'à entraîner notre modèle de régression linéaire sur `X_poly`
    """)
    return


@app.cell
def _(LinearRegression, X_polreg_poly, y_polreg):
    lin_pol_reg = LinearRegression()
    lin_pol_reg.fit(X_polreg_poly, y_polreg)

    print(f"Estimation de theta_0 : {lin_pol_reg.intercept_[0]:.3f} \nEstimation de theta_1 : {lin_pol_reg.coef_[0][0]:.3f} \nEstimation de theta_2 : {lin_pol_reg.coef_[0][1]:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > `LinearRegression` : le biais est stocké dans l'attribut `intercept_` tandis que les autres paramètres sont stockés dans l'attribut `.coef_`.

    ### Intégration dans une pipeline

    Le mieux consiste quand même à intégrer `PolynomialFeatures` et `LinearRegression` dans une pipeline, de sorte qu'on **génère automatiquement** les nouvelles features lorsqu'on traite de nouvelles instances pour faire des prédictions.

    Ça a aussi l'avantage de rendre l'intégration plus concise et lisible.
    """)
    return


@app.cell
def _(LinearRegression, PolynomialFeatures, X_polreg_raw, y_polreg):
    from sklearn.pipeline import Pipeline

    poly_reg = Pipeline([
        ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("linear_regression", LinearRegression()),
    ])

    _ = poly_reg.fit(X_polreg_raw, y_polreg)
    return (poly_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pour tracer la courbe $\hat{\boldsymbol{y}} = \mathbf{X} \boldsymbol{\theta}$, on utilise `np.linspace(-3, 3, 100)` pour créer 100 points régulièrement espacés entre -3 et 3. On crée ensuite un vecteur de 100 prédictions, et on trace les 100 points !
    """)
    return


@app.cell
def _(np, poly_reg):
    X_polreg_new = np.linspace(-3, 3, 100).reshape(100, 1)
    y_polreg_new = poly_reg.predict(X_polreg_new)
    return X_polreg_new, y_polreg_new


@app.cell(hide_code=True)
def _(X_polreg_new, X_polreg_raw, bleu, orange, plt, y_polreg, y_polreg_new):
    plt.figure(figsize=(6, 4))
    plt.plot(X_polreg_raw, y_polreg, ".",color=bleu)
    plt.plot(X_polreg_new, y_polreg_new, "-", color=orange, linewidth=2, label="Predictions")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.legend(loc="upper left")
    plt.axis([-3, 3, 0, 10])
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Degré du polynôme de régression

    ### Explosion combinatoire

    Lorsqu'il y a plusieurs features, la régression qu'on a implémentée a la particularité de pouvoir exhiber des relations entre les features elle-même : c'est dû à `PolynomialFeatures`. Si on dispose de deux prédicteurs $x_1$ et $x_2$, `PolynomialFeatures` ajoute

    $$x_1^2 \ , x_2^2 \ , x_1^3 \ , x_2^3 $$

    mais ajoute aussi les **termes d'interaction**

    $$x_1x_2,\ x_1^2x_2,\ x_1x_2^2.$$

    En notant $d$ le degré maximum fixé (option `degree`), et $n$ le nombre de features pré-existantes, un appel à `PolynomialFeatures` crée exactement $\displaystyle \frac{\left(n+d\right)!}{n! \mathpunct{} d!}$ features.

    Attention donc à l'explosion combinatoire du nombre de features, d'autant que celui [intervient au carré dans le coût asymptotique de la SVD](#complexite-svd).

    # IV. Courbes d'apprentissage

    Cette explosion combinatoire n'est pas le seul inconvénient de réaliser une régression polynomiale à haut degré.

    ## A. Overfitting et phénomène de Runge

    La régression polynomiale à haut degré a un autre défaut : de tels modèles tendent à « coller » aux données, les conduisant à **apprendre le bruit** des données et à perdre en capacité de généralisation.

    Concrètement, les polynômes de degré élevé **oscillent violemment** près des bords des données et **en dehors de la plage d'entraînement ils divergent** vers ±∞, dominés par le terme de plus haut degré. C'est le **phénomène de Runge**. On l'observe très nettement sur la figure ci-dessous.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    degree_slider = mo.ui.slider(
        start=2, stop=300, step=1, value=300,
        label=r"Degré du polynôme", show_value=True,
    )
    degree_slider
    return (degree_slider,)


@app.cell(hide_code=True)
def _(
    LinearRegression,
    PolynomialFeatures,
    StandardScaler,
    X_polreg_new,
    X_polreg_raw,
    bleu,
    degree_slider,
    jaune,
    orange,
    plt,
    rouge,
    y_polreg,
):
    from sklearn.pipeline import make_pipeline

    plt.figure(figsize=(6, 4))
    # (format = style+marqueur, couleur, épaisseur, degré)
    for fmt, _color, width, degree in (("-+", rouge, 3, 1),("--", orange, 4, 2),("-",  jaune, 2, degree_slider.value),):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        _lin_reg = LinearRegression()
        polynomial_regression = make_pipeline(polybig_features, std_scaler, _lin_reg)
        polynomial_regression.fit(X_polreg_raw, y_polreg)
        y_newbig = polynomial_regression.predict(X_polreg_new)
        label = f"{degree} degree{'s' if degree > 1 else ''}"
        plt.plot(X_polreg_new, y_newbig, fmt, color=_color, label=label, linewidth=width)

    plt.plot(X_polreg_raw, y_polreg, ".", color=bleu, markersize=6)  # données en gris ardoise foncé
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([-3, 3, 0, 10])
    plt.grid(True, alpha=0.3)  # grille atténuée pour ne pas surcharger
    plt.show()
    return (make_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    À la vue de ce graphique, le lecteur attentif conclura à l'importance du second degré..

    > Cas extrême du phénomène de Runge : il existe même des configurations où l'écart maximal entre la fonction et son polynôme interpolateur augmente indéfiniment avec $n$, degré du polynôme égal au nombre de points d'interpolation.

    ---
    ## B. Trouver le bon degré

    Tandis que la régression linéaire échoue à capturer la structure sous-jacente des données (**underfitting**), les polynômes de degré 20 et + semblent largement sujets à l'**overfitting**.

    Une des difficultés principale de la régression polynomiale consiste donc à trouver le bon degré, le bon **compromis** entre sous-apprentissage et sur-apprentissage.

    > Ici, on a généré nous même les données à partir d'une relation quadratique, donc la courbe en bleue convient sans surprises ; mais la plupart du temps, la fonction de génération est une information dont on ne dispose pas.

    On se propose d'étudier différents outils pour guider la sélection de notre modèle et évaluer sa propension à l'overfitting/underfitting.
    1. Observer la distribution des données
    2. Comparer l'erreur d'entraînement et l'erreur de validation
    3. Utiliser des _learning curves_

    ---

    ## C. Observer la distribution des données

    En régression polynomiale, une pratique utile et peu coûteuse consiste à jeter un oeil à la distribution des données. Ça permet de se donner une idée du degré à fixer, ce qui est appréciable au vu du coût prohibitif du nombre de features.

    On génère quelques distributions bruitées à titre d'exemple.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, vert):
    _rng = np.random.default_rng(seed=42)
    _m = 3000
    _x = 6 * _rng.random(_m) - 3          # x dans [-3, 3]

    _polynomes = {
        "Degré 1 : x + 1":                              [1, 1],
        "Degré 2 : 0.5x² + x + 2":                      [0.5, 1, 2],
        "Degré 3 : 0.3x³ − 0.5x² + x + 2":              [0.3, -0.5, 1, 2],
        "Degré 4 : 0.1x⁴ + 0.2x³ − 0.5x² + x + 1":      [0.1, 0.2, -0.5, 1, 1],
        "Degré 5 : 0.05x⁵ − 0.1x⁴ + 0.2x³ − 0.5x² + x + 2": [0.05, -0.1, 0.2, -0.5, 1, 2],
        "Degré 6 : 0.02x⁶ + 0.05x⁵ − 0.1x⁴ + 0.2x³ − 0.5x² + x + 2": [0.02, 0.05, -0.1, 0.2, -0.5, 1, 2],
    }

    _n_cols = 3
    _n_lignes = 2
    _fig, _axes = plt.subplots(_n_lignes, _n_cols, figsize=(15, 9))
    _axes = _axes.flatten()

    for _i, (_titre, _coeffs) in enumerate(_polynomes.items()):
        _ax = _axes[_i]
        _signal = np.polyval(_coeffs, _x)
        _bruit = _rng.standard_normal(_m)
        _y = _signal + _bruit

        _ax.plot(_x, _y, ".", color=vert, markersize=4, alpha=0.5)
        _ax.set_title(_titre, fontsize=10)
        _ax.set_xlabel("$x_1$")
        _ax.set_ylabel("$y$", rotation=0)
        _ax.grid(True, alpha=0.3)

    _fig.suptitle("Jeux de données polynomiaux bruités (degré 1 à 6)", fontsize=14)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cette approche a ses limites, on constate qu'on a vite fait de confondre un polynôme de degré 6 avec un polynôme de degré 3.

    Mais on ne confondrait pas une droite affine avec un polynôme de degré 2 : puisque cette visualisation ne coûte rien, autant ne pas s'en priver.

    ---

    ## D. Erreur d'entraînement vs. erreur de validation

    ### Intuition

    Le phénomène d'overfitting survient lorsqu'un modèle « colle » trop aux données d'entraînement, au point que sa capacité de prédiction face à des données jamais rencontrées soit dégradée.

    Intuitivement, sa propension à coller aux données d'entraînement s'observe lorsqu'on demande au modèle de prédire la variable cible d'une instance sur laquelle il a été entraîné. C'est l'**erreur d'entraînement**.

    Une erreur d'entraînement faible ne signifie pas forcément que le modèle overfitte ; peut-être que notre modèle est très performant et que le training set est peu bruité. Il faut mettre cette erreur en **perspective** avec la qualité de ses prédictions sur des données qu'il n'a **jamais vues**.

    Cette dernière mesure s'obtient typiquement grâce à un entraînement par validation croisée  (voir chapitre 2), à l'issue duquel on obtient une mesure de la capacité de généralisation du modèle : l'**erreur de validation**.

    On appelle parfois **écart de généralisation** la différence entre les deux erreurs.

    ### Visualisation <a id="visualisation-ecart-generalisation"></a>

    La visualisation interactive ci-dessous illustre ce principe.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    overfit_degree_slider = mo.ui.slider(
        start=1, stop=15, value=1, step=1,
        label="Degré du polynôme",
        show_value=True,
    )
    overfit_degree_slider
    return (overfit_degree_slider,)


@app.cell(hide_code=True)
def _(bleu, gris, np, overfit_degree_slider, plt, rouge):
    import warnings

    # Jeu de données synthétique : fonction sinus
    _rng = np.random.default_rng(0)
    _x_train = np.sort(_rng.uniform(0.0, 1.0, 30))
    _true_fn = lambda _t: np.sin(2.0 * np.pi * _t)
    _y_train = _true_fn(_x_train) + _rng.normal(0.0, 0.25, _x_train.size)

    _degrees = np.arange(1, 16)
    _n_folds = 5

    def _polyfit_quiet(_xx, _yy, _deg):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            return np.polyfit(_xx, _yy, _deg)

    # Erreur d'entraînement : MSE sur l'ensemble du training set
    def _train_mse(_deg):
        _coef = _polyfit_quiet(_x_train, _y_train, _deg)
        return float(np.mean((np.polyval(_coef, _x_train) - _y_train) ** 2)), _coef

    # Erreur de validation : MSE moyenne par validation croisée à 5 plis
    def _val_mse(_deg):
        _shuf = np.random.default_rng(42).permutation(_x_train.size)
        _folds = np.array_split(_shuf, _n_folds)
        _errs = []
        for _i in range(_n_folds):
            _vi = _folds[_i]
            _ti = np.concatenate([_folds[_j] for _j in range(_n_folds) if _j != _i])
            _pred = np.polyval(_polyfit_quiet(_x_train[_ti], _y_train[_ti], _deg), _x_train[_vi])
            _errs.append(np.mean((_pred - _y_train[_vi]) ** 2))
        return float(np.mean(_errs))

    _train_curve = np.array([_train_mse(_d)[0] for _d in _degrees])
    _val_curve = np.array([_val_mse(_d) for _d in _degrees])
    _best_deg = int(_degrees[np.argmin(_val_curve)])

    # Degré sélectionné via slider 
    _d = int(overfit_degree_slider.value)
    _cur_train, _cur_coef = _train_mse(_d)
    _cur_val = _val_curve[_d - 1]

    # Figure : ajustement gauche/droite
    _fig, (_axL, _axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    _grid = np.linspace(0, 1, 400)

    _axL.scatter(_x_train, _y_train, s=34, color="#2b6cb0", zorder=3, label="Données d'entraînement")
    _axL.plot(_grid, _true_fn(_grid), "--", color="#718096", lw=1.8, label="Fonction réelle")
    _axL.plot(_grid, np.clip(np.polyval(_cur_coef, _grid), -3, 3), color="#dd6b20", lw=2.2,
              label=f"Modèle ajusté (degré {_d})")
    _axL.set_ylim(-2.2, 2.2)
    _axL.set_xlabel("x"); _axL.set_ylabel("y")
    _axL.set_title(f"Ajustement du modèle — degré {_d}")
    _axL.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    _axR.semilogy(_degrees, _train_curve, "o-", color="#2b6cb0", lw=1.8, ms=5, label="Erreur d'entraînement")
    _axR.semilogy(_degrees, _val_curve, "s-", color="#c53030", lw=1.8, ms=5, label="Erreur de validation")
    _axR.axvline(_best_deg, color="#38a169", ls="--", lw=1.4, alpha=0.8)
    _axR.text(_best_deg + 0.2, _val_curve.min() * 1.7, "meilleur\ncompromis",
              color="#38a169", fontsize=8, va="bottom", ha="left")

    # Écart de généralisation au degré courant
    _axR.plot([_d, _d], [_cur_train, _cur_val], color=gris, lw=6, alpha=0.55, solid_capstyle="round")
    _axR.scatter([_d, _d], [_cur_train, _cur_val], color=[bleu, rouge], s=70, zorder=5, ec="white")
    _axR.set_xlabel("Degré du polynôme")
    _axR.set_ylabel("MSE (échelle log)")
    _axR.set_title(f"Écart de généralisation au degré {_d} : {_cur_val - _cur_train:.3f}")
    _axR.set_xticks(_degrees)
    _axR.legend(loc="upper left", fontsize=8.5)
    _fig.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Deux critères permettent de déterminer le meilleur compromis :
    - L'écart de généralisation
    - La moyenne des erreurs

    En fixant un degré de 1 ou de 2, notre modèle ne réalise ni sur-apprentissage ni sous-apprentissage, mais ses prédictions sont comparativement assez **mauvaises**. On considère donc ici que la régression polynomiale de degré 3 est la meilleure, bien que son écart de généralisation ne soit pas le plus petit de tous.

    ### Implémentation

    La façon la plus concise de calculer ces erreurs en gérant les deux types d'entraînements consiste à utiliser `cross_validate`.

    L'implémentation suivante mobilise le jeu de données et la pipeline précédemment créés.
    """)
    return


@app.cell
def _(X_polreg_raw, np, poly_reg, y_polreg):
    from sklearn.model_selection import cross_validate

    res = cross_validate(poly_reg, X_polreg_raw, y_polreg, cv=5, scoring="neg_root_mean_squared_error", return_train_score=True)

    tab_mse_train_error = -res["train_score"]
    tab_mse_validation_error = -res["test_score"]

    print(f"Erreur d'entraînement : {np.mean(tab_mse_train_error):.3f}\nErreur de validation : {np.mean(tab_mse_validation_error):.3f} ")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Formalisation mathématique des erreurs

    Pour mieux comprendre la structure mathématique des notions présentées dans la section suivante, les _learning curves_, il peut être utile de formaliser les concepts que l'on a vus jusqu'ici.

    /// details | Formalisation mathématique des erreurs - (pour aller plus loin)

    Reprenons le cadre de génération utilisé depuis le début du chapitre : le label s'écrit comme une fonction **déterministe** des prédicteurs, corrompue par un bruit additif.

    $$y = f(\mathbf{x}) + \epsilon, \qquad \mathbb{E}[\epsilon] = 0, \quad \mathbb{V}(\epsilon) = \sigma^2$$

    La fonction $f$ est la **structure sous-jacente** que l'on cherche à approcher (la _fonction réelle_ qui génère les distributions de points) ; $\epsilon$ est un bruit aléatoire de moyenne nulle.

    L'entraînement, lui, produit un modèle à partir d'un training set. Mais ce jeu n'est qu'un **tirage** parmi tous les training sets possibles : avec d'autres données, on aurait obtenu d'autres paramètres $\hat{\boldsymbol{\theta}}$, donc un autre modèle. Par conséquent la** fonction hypothèse** apprise est **aléatoire** ; pour alléger, on note $\hat{f}(\mathbf{x}_0)$ sa prédiction en une instance $\mathbf{x}_0$ fixée :

    $$\hat{f}(\mathbf{x}_0) := h_{\hat{\boldsymbol{\theta}}}(\mathbf{x}_0)$$

    > Puisque $\hat{f}$ est fonction de plusieurs variables (le training set $\mathcal{D}$, l'instance $\mathbf{x}_0$ en laquelle est elle évaluée, etc.), on notera en indice la grandeur sur laquelle on moyenne dans l'expression de l'espérance. Par exemple, $\mathbb{E}_{\mathcal{D}}\big[\hat{f}\big]$ est la fonction moyenne obtenue à partir de tous les training sets possibles.

    #### Erreur de généralisation

    Pour un modèle donné déjà entraîné, on définit l'**erreur de généralisation** comme la moyenne des écarts au carrés entre $y$ et $\hat{f}$ sur l'ensemble des instances de l'univers (sauf celles utilisées pendant l'entraînement). Elle quantifie l'exacte capacité de généralisation du modèle (d'où son nom).

    $$R(\hat f) = \mathbb{E}_{(\mathbf{x},y)}\big[(y - \hat f(\mathbf{x}))^2\big]$$

    C'est un idéal théorique, il faudrait toute la distribution pour la calculer.

    #### Erreur de validation

    Cette erreur est simplement l'estimateur empirique de $R(\hat f)$ sur un échantillon $V$ de $p$ instances inédites (i.e. issues du test set).

    $${\widehat R_V(\hat f)}_1 = \frac{1}{p}\sum_{j=1}^{p}\big(y^{(j)} - \hat f(\mathbf{x}^{(j)})\big)^2$$

    NB : On peut prouver facilement que $\mathbb{E}_{V} \big[\widehat R_V(\hat f)- R(\hat f)\big]=0$ c'est-à-dire que cet estimateur est **sans biais**.

    Rigoureusement, cette mesure correspond à la RMSE que l'on calcule sur le test set une fois notre modèle définitivement fixé. Pour avoir une idée de la capacité de généralisation sans induire un _data snooping bias_, on l'approche par validation croisée avec `cross_validate` en faisant la moyenne des erreurs sur les différents folds.

    #### Erreur d'entraînement

    L'erreur d'entraînement ${\widehat R_V(\hat f)}_2$ est un estimateur **biaisé** de $R(\hat f)$.

    $${\widehat R_V(\hat f)}_2 = \frac{1}{p}\sum_{j=1}^{p}\big(y^{(j)} - \hat f(\mathbf{x}^{(j)})\big)^2$$

    Il ressemble à l'erreur de validation, mais son estimation est réalisée sur un échantillon $V'$ d'instances **non inédites**, issues du train set.

    #### Écart de généralisation

    C'est la différence entre l'erreur de validation et l'erreur d'entraînement. Son nom est trompeur, attention à ne pas le confondre avec l'_erreur_ de généralisation.

    ///

    ---

    ## E. Learning curves

    Si l'écart de généralisation constitue déjà un bon indicateur, il peut être plus instructif d'étudier, pour un modèle donné, son évolution au cours de son entraînement.

    Concrètement, il s'agit de mesurer l'erreur d'entraînement et l'erreur de validation à chaque itération de `partial_fit()`, puis de tracer cette évolution. Les deux courbes que l'on obtient sont appelées **learning curves** (courbes d'entraînement).

    > Si le modèle ne supporte pas l'apprentissage incréméntal - i.e. ne dispose pas nativement de la méthode `partial_fit()` - , il suffit de le ré-entraîner complètement sur des sous-ensemble du training set de tailles croissantes.

    ### Implémentation Scikit <a id="implementation-scikit"></a>

    Scikit-Learn facilite l'implémentation avec `learning_curve()` : la fonction itère sur différentes tailles de training set, et renvoie trois tableaux dont les éléments correspondent chacun à une itération.
    1. `train_sizes_abs` : les tailles successives des training subsets
    2. `train_scores` : l'erreur d'entraînement sur le subset courant
    3. `test_scores` : les $k$ erreurs de validation sur les $k$ splits du subset courant

    > L'option `exploit_incremental_learning=True` permet d'entraîner les modèles incrémentalement.

    On reprend nos données générées avec une équation quadratique, on entraîne deux modèles (une régression linéaire, une régression polynomiale de degré 10) et on trace les learnings curves associées. On s'attend à observer pour les deux schémas, respectivement de l'underfitting et de l'overfitting.
    """)
    return


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    X_polreg_raw,
    make_pipeline,
    np,
    y_polreg,
):
    from sklearn.model_selection import learning_curve

    # Régression polynomiale de degré 1 (droite affine)
    uf_polynomial_regression = make_pipeline(PolynomialFeatures(degree=1, include_bias=False), LinearRegression())

    # Régression polynomiale de degré 10 
    of_polynomial_regression = make_pipeline(PolynomialFeatures(degree=10, include_bias=False), LinearRegression())


    uf_train_sizes, uf_train_scores, uf_valid_scores = learning_curve(
        uf_polynomial_regression, X_polreg_raw, y_polreg, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
        scoring="neg_root_mean_squared_error")

    of_train_sizes, of_train_scores, of_valid_scores = learning_curve(
        of_polynomial_regression, X_polreg_raw, y_polreg, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
        scoring="neg_root_mean_squared_error")


    uf_train_errors = -uf_train_scores.mean(axis=1)
    uf_valid_errors = -uf_valid_scores.mean(axis=1)

    of_train_errors = -of_train_scores.mean(axis=1)
    of_valid_errors = -of_valid_scores.mean(axis=1)
    return (
        of_train_errors,
        of_train_sizes,
        of_valid_errors,
        uf_train_errors,
        uf_train_sizes,
        uf_valid_errors,
    )


@app.cell(hide_code=True)
def _(
    bleu,
    of_train_errors,
    of_train_sizes,
    of_valid_errors,
    plt,
    rouge,
    uf_train_errors,
    uf_train_sizes,
    uf_valid_errors,
):
    _, _axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    _axes[0].plot(uf_train_sizes, uf_train_errors, "-+", color=rouge, linewidth=2, label="entraînement")
    _axes[0].plot(uf_train_sizes, uf_valid_errors, "-", color=bleu, linewidth=3, label="validation")
    _axes[0].legend(loc="upper right")
    _axes[0].set_xlabel("Taille du training set")
    _axes[0].set_ylabel("RMSE")
    _axes[0].grid()
    _axes[0].axis([0, 160, 0, 2.5])
    _axes[0].set_title("Learning curve : plot n°1") # Tu peux changer ce titre

    _axes[1].plot(of_train_sizes, of_train_errors, "-+", color=rouge, linewidth=2, label="entraînement")
    _axes[1].plot(of_train_sizes, of_valid_errors, "-", color=bleu, linewidth=3, label="validation")
    _axes[1].legend(loc="upper right")
    _axes[1].set_xlabel("Taille du training set")
    _axes[1].grid()
    _axes[1].axis([0, 160, 0, 2.5])
    _axes[1].set_title("Learning curve : plot n°2")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Après avoir introduit quelques notions utiles, la section [Interpréter les learning curves](#interpreter-learning-curve) présente des critères pratiques pour analyser ces courbes.

    ---

    ## F. La décomposition biais-variance-erreur irréductible

    Un examen approfondi de l'erreur de généralisation peut nous permettre de mieux comprendre ce qui affecte la capacité de généralisation du modèle. En l'espèce, on se propose de montrer que cette erreur est la somme d'un terme de **biais**, d'un terme de **variance** et d'une **erreur irréductible**.

    /// details | La décomposition biais-variance-erreur irréductible (pour aller plus loin)

    On rappelle que la variable cible s'écrit comme une fonction $f$ des prédicteurs, que notre modèle va tenter d'approcher par une fonction $\hat{f}$.

    $$y = f(\mathbf{x}) + \epsilon$$

    Lors de l'entraînement, on produit un modèle à partir d'un training set. Si on considère les choix de modélisation **fixés**, la fonction $\hat{f}$ apprise est entièrement déterminée par le jeu d'entraînement. Et puisque ce jeu n'est qu'un **tirage** parmi tous les training sets possibles, le modèle est lui-même le fruit d'un tirage, dont on peut étudier les **propriétés statistiques** : $\mathbb{V}_{\mathcal{D}}\big[\hat{f} \big]$, $\mathbb{E}_{\mathcal{D}}\big[\hat{f} \big]$, etc..

    Formellement, on note
    $$\hat f = \mathcal{A}(\mathcal{D})$$

    où $\mathcal{A}$ est l'**algorithme d'apprentissage**, qui empaquette tous les choix de modélisation : la classe d'hypothèses $\mathcal{H}$ (linéaire ? polynomiale de degré $d$ ? SVM ?), la standardisation, la fonction de coût, la régularisation, etc..

    L'erreur de généralisation $R(\hat f)$ quantifiant l'exacte capacité de généralisation du modèle **pour un training set donné**, on va étudier son espérance pour tous les tirages possibles. Cela revient à étudier $R(\mathcal{A})$ :

    $$\mathbb{E}_{\mathcal{D}}\big[R(\hat f)\big] =\mathbb{E}_{\mathcal{D}}\,\mathbb{E}_{(\mathbf{x}, y)}\big[(y - \hat f(\mathbf{x}))^2\big] $$

    <a id="etape-demo"></a> Soit $\mathcal P$ la distribution du test set, tirer $(\mathbf{x},y)\sim\mathcal P$ revient à tirer $\mathbf{x}_0 \sim \mathcal P_{\mathbf x}$ et $\epsilon \sim \mathcal P_{\mathbf \epsilon}$ et à fixer ensuite $y = f(\mathbf{x}_0)+\epsilon$. D'autre part, l'intégrande étant positive on peut permuter les espérances en vertu du théorème de Fubini :

    $$\underbrace{\mathbb{E}_{\mathcal{D}}\,\mathbb{E}_{(\mathbf{x}, y)}\big[(y - \hat f(\mathbf{x}))^2\big]}_{\text{erreur de généralisation, moyennée sur } \mathcal{D}} = \mathbb{E}_{\mathbf{x_0}}\Big[\,\underbrace{\mathbb{E}_{\mathcal{D},\,\epsilon}\big[(y - \hat f(\mathbf{x}_0))^2\big]}_{\text{objet de la démonstration}}\,\Big]$$

    Ainsi $\mathbb{E}_{\mathcal{D},\,\epsilon}\big[(y - \hat f(\mathbf{x}_0))^2\big]$ mesure l'erreur quadratique en $\mathbf{x}_0$, moyennée à la fois sur le bruit $\epsilon$ de l'instance test et sur le tirage du training set. En notant $\bar{f}(\mathbf{x}_0) = \mathbb{E}[\hat{f}(\mathbf{x}_0)]$ la prédiction du **modèle moyen** (celui qu'on obtiendrait en moyennant une infinité de modèles entraînés), cette erreur se décompose en **trois termes** :

    $$\mathbb{E}_{\mathcal{D},\,\epsilon}\big[(y - \hat f(\mathbf{x}_0))^2\big] = \underbrace{\big(f(\mathbf{x}_0) - \bar{f}(\mathbf{x}_0)\big)^2}_{\text{biais}^2} \;+\; \underbrace{\mathbb{E}\!\left[\big(\hat{f}(\mathbf{x}_0) - \bar{f}(\mathbf{x}_0)\big)^2\right]}_{\text{variance}} \;+\; \underbrace{\sigma^2}_{\text{erreur irréductible}}$$

    /// details | Décomposition biais-variance - démonstration (pour aller _encore_ loin)

    **Étape 0 - contexte et notations**

    On suppose que le bruit $\epsilon$ de l'instance test est :
    1. **indépendant** du training set $\mathcal{D}$
    2. **centré** ($\mathbb{E}[\epsilon]=0$)
    3. de variance $\sigma^2 = \mathbb{V}(\epsilon)$.

    Pour alléger, on omet la dépendance en $\mathbf{x}_0$ et on note :

    $$f = f(\mathbf{x}_0), \quad \hat f = \hat f(\mathbf{x}_0), \quad \bar f = \bar f(\mathbf{x}_0) = \mathbb{E}_{\mathcal{D}}[\hat f]$$

    On rappelle que $y = f + \epsilon$ et que notre objectif est de décomposer $\mathbb{E}_{\mathcal{D},\,\epsilon}\big[(y - \hat f)^2\big]$.

    **Étape 1 — isoler le bruit.** En écrivant $y - \hat f = (f - \hat f) + \epsilon$ et en développant le carré :

    $$(y - \hat f)^2 = (f - \hat f)^2 + 2\,\epsilon\,(f - \hat f) + \epsilon^2$$

    On prend l'espérance sur $\mathcal{D}$ et $\epsilon$. Le terme croisé se **factorise** car $\epsilon \perp \mathcal{D}$, puis **s'annule** car $\mathbb{E}[\epsilon]=0$ :

    $$\mathbb{E}_{\mathcal{D},\epsilon}\big[2\,\epsilon\,(f - \hat f)\big] = 2\,\underbrace{\mathbb{E}[\epsilon]}_{=\,0}\;\mathbb{E}_{\mathcal{D}}\big[f - \hat f\big] = 0$$

    La quantité $f - \hat f$ ne dépendant que de $\mathcal{D}$, et $\mathbb{E}_\epsilon[\epsilon^2] = \mathbb{V}(\epsilon) = \sigma^2$ (le bruit est centré), il reste :

    $$\mathbb{E}_{\mathcal{D},\epsilon}\big[(y - \hat f)^2\big] = \underbrace{\mathbb{E}_{\mathcal{D}}\big[(f - \hat f)^2\big]}_{\text{erreur d'estimation}} + \;\sigma^2$$

    **Étape 2 — faire apparaître le modèle moyen.** On introduit $\bar f$ dans le premier terme via $f - \hat f = (f - \bar f) + (\bar f - \hat f)$ :

    $$(f - \hat f)^2 = (f - \bar f)^2 + 2\,(f - \bar f)(\bar f - \hat f) + (\bar f - \hat f)^2$$

    On prend l'espérance sur $\mathcal{D}$. De nouveau le terme croisé s'annule : $f - \bar f$ est une constante indépendante de $\mathcal{D}$ ($f$ est le modèle théorique et $\bar f$ est déjà moyenné sur $\mathcal{D}$), et $\mathbb{E}_{\mathcal{D}}[\bar f - \hat f] = \bar f - \mathbb{E}_{\mathcal{D}}[\hat f] = \bar f - \bar f = 0$ par **définition** de $\bar f$ :

    $$\mathbb{E}_{\mathcal{D}}\big[2\,(f - \bar f)(\bar f - \hat f)\big] = 2\,(f - \bar f)\,\underbrace{\mathbb{E}_{\mathcal{D}}\big[\bar f - \hat f\big]}_{=\,0} = 0$$

    Il vient :

    $$\mathbb{E}_{\mathcal{D}}\big[(f - \hat f)^2\big] = \underbrace{(f - \bar f)^2}_{\text{biais}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat f - \bar f)^2\big]}_{\text{variance}}$$

    On reconnaît la formule de la variance $\mathbb{V}_{\mathcal{D}}\big(\hat f(\mathbf{x}_0)\big) = \mathbb{E}_{\mathcal{D}}\big[(\hat f - \bar f)^2\big]$, puisque $\bar f = \mathbb{E}_{\mathcal{D}}[\hat f]$.

    **Conclusion.** En réinjectant ce résultat dans celui de l'étape 1 :

    $$\mathbb{E}_{\mathcal{D},\,\epsilon}\big[(y - \hat f(\mathbf{x}_0))^2\big] = \underbrace{\big(f(\mathbf{x}_0) - \bar f(\mathbf{x}_0)\big)^2}_{\text{biais}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat f(\mathbf{x}_0) - \bar f(\mathbf{x}_0))^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{erreur irréductible}}$$

    ///

    Par linéarité de l'espérance, c'est l'erreur de généralisation elle-même qui se décompose en **trois contributions**.

    ///

    ### Analyse des 3 contributions

    1. L'**erreur irréductible** $\sigma^2=\mathbb{V}(\epsilon)$ porte bien son nom : elle ne dépend ni du modèle ni du training set, donc même le modèle parfait $\hat{f} = f$ la subirait, puisqu'aucun apprentissage ne peut deviner le bruit $\epsilon$.
    2. Le **biais** $\big(f - \bar{f}\big)^2$ mesure l'écart entre la structure réelle $f$ et le modèle moyen $\bar{f}$. Il est élevé quand la famille de modèles est **trop rigide** pour épouser $f$. C'est la signature de l'**underfitting**.
    3. La **variance** $\mathbb{V}(\hat{f})$ mesure la sensibilité du modèle au training set : de combien $\hat{f}$ fluctue-t-il lorsqu'on change les données ? Elle est élevée quand le modèle est **trop flexible** et épouse le bruit propre à chaque jeu. C'est la signature de l'**overfitting**.

    C'est ce vocabulaire, **biais élevé** pour l'underfitting, **variance élevée** pour l'overfitting, que l'on mobilise pour lire les learning curves dans la section suivante.

    ### Visualisation

    La décomposition précédente est un énoncé sur le **tirage du training set** $\mathcal{D}$ : elle moyenne l'erreur sur tous les jeux d'entraînement possibles. La visualisation suivante est générée en tirant $K$ training sets issus du même $f+\epsilon$ et en ajustant sur chacun un polynôme dont on règle le degré (la **complexité**).

    Ce graphique nourrit l'intuition de compréhension deux objets : le **modèle moyen** $\bar f = \mathbb{E}_{\mathcal{D}}[\hat f]$ (« celui qu'on obtiendrait en moyennant une infinité de modèles ») et la **variance** comprise comme *dispersion de $\hat f$ d'un tirage à l'autre*.

    Quelques précisions sur ces trois graphes :

    1. **Graphe 1**, haut-gauche. Le faisceau orange réunit les $K$ modèles $\hat f$ ; la courbe rouge est le modèle moyen $\bar f$, la verte la vraie fonction $f$, la bande grise le bruit $\pm\sigma$. En l'abscisse $x_0$, le **segment noir mesure le biais $f(x_0)-\bar f(x_0)$ : l'écart entre la vraie fonction et le modèle moyen**.

    2. **Graphe 2**, haut-droit. Chaque point violet correspond à la **dispersion $(\hat f_i - \bar f)^2$** d'un des $K$ modèles ; le losange orange est leur moyenne - c'est précisément la **variance**. La courbe est tracée en fonction du degré : on voit la variance augmenter à mesure que les modèles se mettent à « coller » au bruit (overfit).

    3. **Graphe 3**, bas-gauche. Les trois contributions, cette fois **moyennées sur tout le domaine** des $x$, tracées en fonction de la complexité. Ce sont littéralement les composantes de l'**erreur de généralisation**, tracée en violet. Son creux marque la complexité optimale.

    > Dans le panneau de variance (en haut à droite), le losange (la moyenne) est tiré vers le **haut** du nuage : les contributions $(\hat f_i-\bar f)^2$ sont fortement asymétriques, si bien que quelques modèles aberrants suffisent à tirer la variance vers le haut.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    bv_true_fn = lambda t: np.sin(2.0 * np.pi * t)
    bv_sigma = 0.4 
    bv_m = 20 
    bv_K = 400
    bv_degmax = 7
    bv_degrees = np.arange(1, bv_degmax + 1)
    bv_grid = np.linspace(0.0, 1.0, 300)
    bv_test_pts = np.linspace(0.12, 0.88, 80)

    def bv_polyfit(x, y, deg):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            return np.polyfit(x, y, deg)

    _bv_rng = np.random.default_rng(seed=7)
    bv_datasets = []
    for _ in range(bv_K):
        _x = np.sort(_bv_rng.uniform(0.0, 1.0, bv_m))
        _y = bv_true_fn(_x) + _bv_rng.normal(0.0, bv_sigma, bv_m)
        bv_datasets.append((_x, _y))

    _bv_ft = bv_true_fn(bv_test_pts)
    bv_bias2, bv_var = [], []
    for _d in bv_degrees:
        _fits = [bv_polyfit(_x, _y, _d) for (_x, _y) in bv_datasets]
        _P = np.array([np.polyval(_c, bv_test_pts) for _c in _fits])   # K x len(test_pts)
        _P = np.where(np.isfinite(_P), _P, np.nan)
        _bar = np.nanmean(_P, axis=0)
        bv_bias2.append(np.nanmean((_bar - _bv_ft) ** 2))
        bv_var.append(np.nanmean(np.nanmean((_P - _bar) ** 2, axis=0)))
    bv_bias2 = np.array(bv_bias2)
    bv_var = np.array(bv_var)
    bv_total = bv_bias2 + bv_var + bv_sigma ** 2
    return (
        bv_bias2,
        bv_datasets,
        bv_degmax,
        bv_degrees,
        bv_grid,
        bv_polyfit,
        bv_sigma,
        bv_test_pts,
        bv_total,
        bv_true_fn,
        bv_var,
    )


@app.cell(hide_code=True)
def _(bv_degmax, mo):
    bv_degree_slider = mo.ui.slider(
        start=1, stop=bv_degmax, step=1, value=1,
        label="Degré du polynôme", show_value=True,
    )
    bv_x0_slider = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.80,
        label=r"$x_0$ (point de lecture du biais)", show_value=True,
    )
    mo.vstack([bv_degree_slider, bv_x0_slider])
    return bv_degree_slider, bv_x0_slider


@app.cell(hide_code=True)
def _(bv_datasets, bv_degree_slider, bv_grid, bv_polyfit, bv_test_pts, np):
    bv_d = int(bv_degree_slider.value)
    bv_fits = [bv_polyfit(_x, _y, bv_d) for (_x, _y) in bv_datasets]
    bv_curves = np.array([np.polyval(_c, bv_grid) for _c in bv_fits])
    bv_mean = bv_curves.mean(axis=0)

    _Pt = np.array([np.polyval(_c, bv_test_pts) for _c in bv_fits])
    _Pt = np.where(np.isfinite(_Pt), _Pt, np.nan)
    _bart = np.nanmean(_Pt, axis=0)
    bv_vcloud = np.nanmean((_Pt - _bart) ** 2, axis=1)
    bv_vcloud = bv_vcloud[np.isfinite(bv_vcloud) & (bv_vcloud > 0)]
    bv_var_d = float(np.mean(bv_vcloud))     # = bv_var[bv_d - 1], la moyenne du nuage
    return bv_curves, bv_d, bv_fits, bv_mean, bv_var_d, bv_vcloud


@app.cell(hide_code=True)
def _(
    bleu,
    bv_bias2,
    bv_curves,
    bv_d,
    bv_degmax,
    bv_degrees,
    bv_fits,
    bv_grid,
    bv_mean,
    bv_sigma,
    bv_total,
    bv_true_fn,
    bv_var,
    bv_var_d,
    bv_vcloud,
    bv_x0_slider,
    gris,
    np,
    orange,
    plt,
    rouge,
    vert,
    violet,
):
    _x0 = float(bv_x0_slider.value)
    _ylo, _yhi = -2.2, 2.2
    _clip = lambda a: np.clip(a, _ylo, _yhi)

    _fig, _axes = plt.subplots(2, 2, figsize=(13.5, 9))
    _A, _B, _C = _axes[0, 0], _axes[0, 1], _axes[1, 0]
    _axes[1, 1].axis("off")

    for _c in bv_curves:
        _A.plot(bv_grid, _clip(_c), color=orange, lw=0.8, alpha=0.05)
    _A.plot([], [], color=orange, lw=1.4, alpha=0.5, label=r"modèles $\hat f$")
    _A.fill_between(bv_grid, bv_true_fn(bv_grid) - bv_sigma, bv_true_fn(bv_grid) + bv_sigma,
                    color=gris, alpha=0.12, label=r"bruit $\pm\sigma$")
    _A.plot(bv_grid, bv_true_fn(bv_grid), "--", color=vert, lw=2.4, label=r"vraie fonction $f$")
    _A.plot(bv_grid, _clip(bv_mean), color=rouge, lw=2.6, label=r"modèle moyen $\bar f$")

    _p0 = np.array([np.polyval(_c, _x0) for _c in bv_fits])
    _p0 = _p0[np.isfinite(_p0)]
    _m0, _f0 = _p0.mean(), float(bv_true_fn(_x0))
    _mc = float(_clip(np.array([_m0]))[0])
    _A.axvline(_x0, color=gris, ls=":", lw=1.0, alpha=0.7)
    _A.plot([_x0, _x0], [_f0, _m0], color="black", lw=2.6, zorder=6)
    _A.annotate("biais", xy=(_x0, (_f0 + _m0) / 2), xytext=(_x0 - 0.20, (_f0 + _m0) / 2),
                fontsize=11, va="center", arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    _A.scatter([_x0], [_f0], marker="*", s=210, color=vert, ec="white", lw=0.6, zorder=7)
    _A.scatter([_x0], [_mc], s=95, color=rouge, ec="white", lw=0.6, zorder=7)
    _A.set_xlim(0, 1); _A.set_ylim(_ylo, _yhi)
    _A.set_xlabel("$x$"); _A.set_ylabel("$y$", rotation=0)
    _A.set_title(fr"Les $K$ modèles (degré {bv_d}) — biais en $x_0={_x0:.2f}$", fontsize=12)
    _A.legend(loc="upper right", fontsize=9, framealpha=0.9)

    _B.plot(bv_degrees, bv_var, "-", color=orange, lw=1.0, alpha=0.30)   
    _B.plot(bv_degrees[:bv_d], bv_var[:bv_d], "-o", color=orange, lw=2.4, ms=5) 
    _jr = np.random.default_rng(0)
    _jit = _jr.uniform(-0.20, 0.20, bv_vcloud.size)
    _B.scatter(bv_d + _jit, bv_vcloud, s=10, color=violet, alpha=0.22, zorder=3,
               label=r"$(\hat f_i-\bar f)^2$ par modèle")
    _B.scatter([bv_d], [bv_var_d], marker="D", s=95, color=orange, ec="black", lw=0.7, zorder=5,
               label=r"variance $=$ moyenne")
    _B.set_yscale("log")
    _B.set_xlim(0.5, bv_degmax + 0.5); _B.set_xticks(bv_degrees)
    _B.set_xlabel("Degré du polynôme"); _B.set_ylabel("Contribution à la variance (log)")
    _B.set_title(r"Variance : dispersion des $K$ modèles", fontsize=12)
    _B.legend(loc="upper left", fontsize=9, framealpha=0.9)

    _C.semilogy(bv_degrees, bv_bias2, "o-", color=bleu, lw=2.0, ms=6, label=r"biais$^2$")
    _C.semilogy(bv_degrees, bv_var, "s-", color=orange, lw=2.0, ms=6, label="variance")
    _C.semilogy(bv_degrees, bv_total, "^-", color=violet, lw=2.4, ms=6, label="erreur de généralisation")
    _C.axhline(bv_sigma ** 2, color=gris, ls="--", lw=1.4, label=r"erreur irréductible $\sigma^2$")
    _C.axvline(bv_d, color=rouge, lw=1.3, alpha=0.7)
    _C.set_xlabel("Degré du polynôme"); _C.set_ylabel("Erreur (échelle log)")
    _C.set_xticks(bv_degrees)
    _C.set_title("Décomposition de l'erreur de généralisation", fontsize=12)
    _C.legend(loc="lower left", fontsize=9, framealpha=0.9)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Le compromis biais-variance

    On parle souvent de compromis biais-variance (il y a même une [page Wikipédia](https://fr.wikipedia.org/wiki/Dilemme_biais-variance) dédiée), mais même en étudiant la démonstration précédente et l'expression des trois contributions, il n'est pas trivial d'établir l'**antagonisme** entre ces termes.

    En fait, on peut démontrer que si l'on évalue ces deux composantes le long d'une famille de complexité croissante, **leurs dérivées par rapport à la complexité sont de signes opposés** (le biais décroît, la variance croît). C'est d'ailleurs _exactement_ ce que l'on observe sur le graphique 3 un peu plus haut !

    En conclusion, cet antagonisme n'a de sens que si l'on considère biais et variance comme **fonctions de la complexité** du modèle.

    On gardera à l'esprit l'intuition suivante : complexifier le modèle (augmenter le degré, par exemple) réduit le biais mais gonfle la variance, et inversement. On ne peut annuler les deux simultanément, et l'erreur totale est minimale pour une complexité **intermédiaire** : exactement ce que l'on observe aussi sur ce même schéma.

    > Tout ceci à **taille de training set fixée**. Faire varier cette taille, c'est précisément ce que tracent les learning curves : biais et variance évoluent avec le nombre d'instances, ce qui explique le mouvement des deux courbes.

    ### Interpréter les learning curves <a id="interpreter-learning-curve"></a>

    Pour faire le lien entre nos 3 contributions et les learnings curves, il faut s'intéresser à la façon dont ces termes se comportent au fur et à mesure que le training set grossit (i.e. que $m$ grossit) :

    1. **$\sigma^2$ : plat.** Le bruit ne dépend ni du modèle ni des données.
    2. **biais : essentiellement plat.** Il est fixé par la _famille_ de modèles, pas par la quantité de données. Quand $m \to \infty$, le modèle moyen $\bar f$ tend vers le meilleur modèle de la classe $f^\star$, donc le biais tend vers $f - f^\star - \text{constante}$. La constante est éventuellement nulle si la classe peut représenter $f$.
    3. **variance : décroissante et tend vers $0$**. Plus de données $\Rightarrow$ $\hat f$ se stabilise d'un training set à l'autre $\Rightarrow$ les fluctuations s'éteignent.

    En conséquence, la hauteur du plateau où se rejoignent les courbes correspond à la somme du biais et du bruit. La variance s'observe sur l'écart entre les courbes.

    /// details | Interprétation graphique - démonstration (pour aller _vraiment_ plus loin)

    On se propose de démontrer ici le lien entre les composantes de l'erreur de généralisation et l'interprétation des learnings curves : « _la hauteur du plateau où se rejoignent les courbes correspond à la somme du biais et du bruit. La variance s'observe sur l'écart entre les courbes _» .

    **Rappels :**

    - Erreur de généralisation : $R(\hat f) = \mathbb{E}_{(\mathbf{x},y)}\big[(y - \hat f(\mathbf{x}))^2\big]$.

    - Erreur de validation : ${\widehat R_{V_{\text{test}}}(\hat f)}_1 = \frac{1}{p}\sum_{j=1}^{p}\big(y^{(j)} - \hat f(\mathbf{x}^{(j)})\big)^2$ sans biais donc $\mathbb{E}_V\big[{\widehat R_{V_{\text{test}}}(\hat f)}_1\big] = R(\hat f)$

    - Erreur d'entraînement : ${\widehat R_{V_{\text{train}}}(\hat f)}_2 = \frac{1}{p}\sum_{j=1}^{p}\big(y^{(j)} - \hat f(\mathbf{x}^{(j)})\big)^2$

    **Notations :**

    - On note $\hat f_i := \hat f(\mathbf{x}^{(i)})$ et $f_i := f(\mathbf{x}^{(i)})$ , où $\mathbf{x}^{(i)}$ est l'instance d'indice $i$ du test set.

    **Objectif :**

    - On étudie l'espérance des deux courbes sur le tirage du training set $\mathcal{D}$, à taille $m$ fixée.

    #### Courbe de l'erreur de validation

    En moyennant sur $\mathcal{D}$ et $V_{\text{test}}$ et en réinjectant la décomposition biais-variance (intégrée sur $\mathbf{x}_0$, comme on l'a fait [ici](#etape-demo)) :

    $$\mathbb{E}_{\mathcal{D},V}\big[{\widehat R_V(\hat f)}_1\big] = \mathbb{E}_{\mathcal{D}}\big[R(\hat f)\big] = \mathbb{E}_{\mathbf{x}_0}\Big[\,\underbrace{(f - \bar f)^2}_{\text{biais}^2} + \underbrace{\mathbb{V}_{\mathcal{D}}(\hat f)}_{\text{variance}}\,\Big] + \sigma^2$$

    On peut montrer grâce au théorème central limite que la matrice de covariance des paramètres estimés $\hat{\theta}$ est proportionnelle à $\frac{1}{m}$. Par conséquent, quand $m \to \infty$, la variance tend vers $0$ : **la courbe de validation tend vers le plateau $\text{biais}^2 + \sigma^2$**.

    #### Courbe de l'erreur d'entraînement

    On prend pour $V'$ le training set lui-même ($p = m$). Problème : chaque $(\mathbf{x}^{(i)}, y^{(i)})$ a servi à fabriquer $\hat f$, donc $\hat f_i$ n'est plus indépendant de $y^{(i)}$ ; le terme croisé qui s'annulait à l'étape 1 de la décomposition ne s'annule plus.

    Pour **mesurer cet écart**, on introduit en chaque point la quantité ${y'}^{(i)} = f_i + {\epsilon'}^{(i)}$, de même loi mais avec ${\epsilon'}^{(i)} \perp \mathcal{D}$. On appelle **optimisme** l'écart en espérance entre l'erreur sur ces quantités et l'erreur d'entraînement :

    $$\text{optimisme}(m) = \mathbb{E}_{\mathcal{D},\,\epsilon'}\Big[\tfrac{1}{m}\sum_{i=1}^{m} ({y'}^{(i)} - \hat f_i)^2\Big] - \mathbb{E}_{\mathcal{D}}\big[{\widehat R_V(\hat f)}_2\big]$$

    En développant les deux carrés, les termes $\mathbb{E}[{y'}^2] = \mathbb{E}[y^2]$ et $\hat f_i^2$ se compensent ; il reste, en chaque point, $-2\,\mathbb{E}[\hat f_i\,{y'}^{(i)}] + 2\,\mathbb{E}[\hat f_i\,y^{(i)}]$. Or ${y'}^{(i)} \perp \hat f_i$ donne $\mathbb{E}[\hat f_i\,{y'}^{(i)}] = f_i\,\mathbb{E}[\hat f_i]$, tandis que $\mathbb{E}[\hat f_i\,y^{(i)}] = \mathrm{Cov}(\hat f_i, y^{(i)}) + f_i\,\mathbb{E}[\hat f_i]$. Les termes en $f_i\,\mathbb{E}[\hat f_i]$ se télescopent :

    $$\text{optimisme}(m) = \frac{2}{m}\sum_{i=1}^{m} \mathrm{Cov}\big(\hat f_i,\, y^{(i)}\big) \;\ge\; 0$$

    Cette covariance mesure de combien la prédiction $\hat f_i$ suit son propre label : c'est la signature de la **complexité** du modèle, qui alimente la variance.

    L'identité ci-dessus se réécrit, le membre de gauche partageant **le même plateau** que la validation (même décomposition biais-variance, aux points $\mathbf{x}^{(i)}$) :

    $$ \mathbb{E}_{\mathcal{D}}\big[\widehat R_V(\hat f)_2\big] + \underbrace{\text{optimisme}(m)\vphantom{\mathbb{E}_{\mathcal{D},\,\epsilon'}\Big[\tfrac{1}{m}\sum_{i=1}^{m}({y'}^{(i)}-\hat f_i)^2\Big]}}_{\xrightarrow{\,m\to\infty\,}0} = \underbrace{\mathbb{E}_{\mathcal{D},\,\epsilon'}\Big[\tfrac{1}{m}\sum_{i=1}^{m}({y'}^{(i)}-\hat f_i)^2\Big]}_{\xrightarrow{\,m\to\infty\,}\text{biais}^2+\sigma^2} $$

    #### Conclusion

    Les deux courbes rejoignent donc théoriquement le même plateau $\text{biais}^2 + \sigma^2$ (variance et optimisme tendent vers $0$) : c'est **la hauteur du plateau qui donne biais + bruit**. D'autre part, **l'écart résiduel à $m$ fini est gouverné par la complexité du modèle (optimisme), donc par la variance**. On retrouve exactement les deux règles de lecture.

    ///

    En pratique, on se contente de retenir les règles suivantes, qui permettent généralement d'évaluer correctement la capacité de généralisation du modèle, indépendamment de sa nature (régression linéaire, polynomiale, SVM, forêt aléatoire etc..).

    - **Underfitting** (biais elevé $\Rightarrow$ plateau élevé et variance faible (donc faible écart))
      - Les deux courbes convergent vers un plateau associé à une erreur élevée
      - Faible écart asymptotique

    - **Overfitting** (variance élevée $\Rightarrow$ écart marqué et biais faible (plateau bas))
      - Erreur d'entraînement basse
      - Écart asymptotique important

    Par conséquent, un bon ajustement serait marqué par une faible erreur et un faible écart. Et si les deux courbes bougent encore beaucoup au bord droit, c'est probablement qu'on a pas assez de données.

    > Certains aspects de ces courbes restent largement imputables à la nature du modèle, notamment le fait que l'erreur d'entraînement augmente / reste proche de 0, ou que l'écart se résorbe / persiste avec les données.

    On en déduit, [comme prévu](#implementation-scikit) :
    1. _Learning curve : plot n°1_ exhibe un phénomène d'underfitting (plateau + faible écart)
    2. _Learning curve : plot n°2_ illustre un phénomène d'overfitting (erreur faible + écart marqué)
    """)
    return


if __name__ == "__main__":
    app.run()
