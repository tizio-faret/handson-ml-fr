import marimo

__generated_with = "0.24.0"
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
    from copy import deepcopy

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import (
        ElasticNet,
        ElasticNetCV,
        Lasso,
        LassoCV,
        LinearRegression,
        Ridge,
        RidgeCV,
        SGDRegressor,
    )
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import RepeatedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    def make_poly_pipe(model, degree):
        return make_pipeline(
            PolynomialFeatures(degree, include_bias=False), StandardScaler(), model
        )

    def rmse(model, X, y):
        return np.sqrt(mean_squared_error(y, model.predict(X)))

    rouge = "#c53030"
    orange = "#dd6b20"
    gris = "#718096"
    bleu = "#2b6cb0"
    vert = "#2f855a"
    violet = "#6b46c1"
    jaune = "#fcbf49"
    return (
        ElasticNet,
        ElasticNetCV,
        Lasso,
        LassoCV,
        LinearRegression,
        PolynomialFeatures,
        RepeatedKFold,
        Ridge,
        RidgeCV,
        SGDRegressor,
        StandardScaler,
        bleu,
        cross_val_score,
        deepcopy,
        gris,
        make_pipeline,
        make_poly_pipe,
        mo,
        np,
        orange,
        plt,
        rmse,
        rouge,
        vert,
        violet,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # I. Modèles linéaires régularisés

    Régulariser un modèle d'apprentissage automatique consiste à réduire ses degrés de liberté en vue de réduire sa tendance à l'overfitting ou de le stabiliser. Pour les modèles de régression linéaire, on utilise souvent un système de pénalités (Lasso, Ridge ou Elastic Net - c'est le sujet des section suivantes).

    ## A. Régression Ridge

    ### Formalisation

    On rappelle que la régression linéaire classique consiste à chercher un vecteur $\boldsymbol{\theta}$ qui **minimise la fonction de perte** RMSE.

    Avec la régression Ridge, on opte pour une fonction de perte $J(\boldsymbol{\theta})$ un peu différente. En notant $\mathbf{w}$ le vecteur des paramètres $\boldsymbol{\theta}$ auquel on a retiré le biais $\theta_0=1$, la fonction $J(\boldsymbol{\theta})$ s'écrit :

    $$J(\boldsymbol{\theta}) = \mathrm{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{m} \sum_{i=1}^{n} {\theta_i}^2 = \mathrm{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{m} {{\| \mathbf{w}\|}_2}^{2}$$

    > Sommer sur les $n$ features et diviser par le nombre $m$ d'instances peut sembler surprenant. Sans entrer dans les détails, diviser par $m$ permet de rendre $\alpha$ indépendant de la taille $m$ du training set.

    ### Intuition

    En minimisant **conjointement** deux termes, l'objectif est double :
    1. MSE : adapter le modèle à la distribution, comme la régression linéaire classique.
    2. Terme de pénalisation : limiter la taille des poids du modèle

    > L'importance de ce deuxième effet est contrôlé par le paramètre $\alpha$.

    ### Visualisations <a id="learning-curves"></a>

    Dans le graphique ci-dessous, on balaie $\alpha$ et on trace les deux RMSE : l'**écart entre l'erreur d'entraînement et de test**, caractéristique de l'overfitting. Le $\alpha$ retenu (pointillés) minimise l'erreur de **validation croisée sur le training set**.

    On reporte dans le titre du graphique deux métriques pour comparer Ridge, Lasso et Elastic-Net en terme de **qualité des prédictions** et de **surapprentissage** :

    1. Calcul de la MSE sur le test set
    2. Calcul de l'écart de généralisation $\Delta_{\text{gén}}$ (erreur de test $-$ erreur d'entraînement)

    > L'écart de généralisation Ridge est mis en perspective avec l'écart de généralisation d'une régression sans régularisation, de sorte qu'on puisse observer la capacité des modèles régularisés à réduire le surapprentissage.
    """)
    return


@app.cell(hide_code=True)
def _(
    RepeatedKFold,
    bleu,
    cross_val_score,
    gris,
    make_poly_pipe,
    np,
    plt,
    poly_X,
    poly_Xte,
    poly_max_degree,
    poly_rmse_ols,
    poly_rmse_oracle,
    poly_y,
    poly_yte,
    rmse,
    rouge,
):
    def courbe_regularisation(make_model, alphas, couleur, nom):
        _cv_split = RepeatedKFold(n_splits=5, n_repeats=30, random_state=42)
        _train, _test, _cv = [], [], []
        for _a in alphas:
            _pipe = make_poly_pipe(make_model(_a), poly_max_degree)
            _cv.append(
                -cross_val_score(
                    _pipe,
                    poly_X,
                    poly_y,
                    cv=_cv_split,
                    scoring="neg_root_mean_squared_error",
                ).mean()
            )
            _pipe.fit(poly_X, poly_y)
            _train.append(rmse(_pipe, poly_X, poly_y))
            _test.append(rmse(_pipe, poly_Xte, poly_yte))
        _train, _test = np.array(_train), np.array(_test)
        _ibest = int(np.argmin(_cv))
        _a_best, _test_best = alphas[_ibest], _test[_ibest]

        _train_best = _train[_ibest]
        _gap = _test_best - _train_best
        _gap0 = poly_rmse_ols[1] - poly_rmse_ols[0]

        _fig, _ax = plt.subplots(figsize=(8.5, 4.6))
        _ax.fill_between(
            alphas,
            _train,
            _test,
            color=couleur,
            alpha=0.12,
            label="écart train/test = overfitting",
        )
        _ax.plot(alphas, _train, "-", color=rouge, lw=2.2, label="RMSE entraînement")
        _ax.plot(alphas, _test, "-", color=bleu, lw=2.4, label="RMSE test")
        _ax.axhline(
            poly_rmse_oracle, color=gris, ls="--", lw=1.3, label="oracle (vrai degré 3)"
        )
        _ax.axvline(_a_best, color=couleur, lw=1.4, ls=":")
        _ax.scatter(
            [_a_best],
            [_test_best],
            s=75,
            color=couleur,
            ec="white",
            lw=0.8,
            zorder=6,
            label=rf"$\alpha$ choisi par CV = {_a_best:.3g}",
        )
        _ax.set_xscale("log")
        _ax.set_xlabel(r"$\alpha$ (échelle log)")
        _ax.set_ylabel("RMSE")
        _ax.set_title(
            f"RMSE test {_test_best:.3f} | "
            f"$\\Delta_{{\\text{{gén}}}}$ sans régularisation {_gap0:.2f} | $\\Delta_{{\\text{{gén}}}}$ Ridge {_gap:.2f}",
            fontsize=11.5,
        )
        _ax.legend(fontsize=8, loc="upper center", framealpha=0.92)
        for _s in ("top", "right"):
            _ax.spines[_s].set_visible(False)
        _fig.tight_layout()
        plt.close(_fig)
        return _fig

    return (courbe_regularisation,)


@app.cell(hide_code=True)
def _(Ridge, courbe_regularisation, np, vert):
    courbe_regularisation(
        lambda _a: Ridge(alpha=_a), np.logspace(-3, 3, 60), vert, "Ridge"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plus $\alpha$ est proche de 0, plus la pénalisation est faible et plus le modèle tend à se comporter comme une régression classique.

    Visuellement, sur une régression linéaire, augmenter $\alpha$ conduit à des prédictions plus lisses, ce qui réduit la variance du modèle mais augmente son biais (on l'observe d'ailleurs très bien sur le graphique ci-dessus).
    """)
    return


@app.cell(hide_code=True)
def _(LinearRegression, Ridge, bleu, make_poly_pipe, np, plt, rouge, vert):
    _rng = np.random.default_rng(seed=42)
    _m = 20
    X_regularization_demo = 3 * _rng.random((_m, 1))
    y_regularization_demo = (
        1 + 0.5 * X_regularization_demo + _rng.standard_normal((_m, 1)) / 1.5
    )
    X_new_regularization_demo = np.linspace(0, 3, 100).reshape(100, 1)

    _ridge_reg = Ridge(alpha=0.1, solver="cholesky")
    _ = _ridge_reg.fit(X_regularization_demo, y_regularization_demo)

    def _plot_model(model_class, polynomial, alphas, **model_kwargs):
        plt.plot(
            X_regularization_demo,
            y_regularization_demo,
            marker=".",
            linestyle="none",
            color=bleu,
            linewidth=3,
        )
        line_colors = (bleu, vert, rouge)
        line_styles = (":", "--", "-")
        for alpha, color, style in zip(alphas, line_colors, line_styles):
            if alpha > 0:
                model = model_class(alpha, **model_kwargs)
            else:
                model = LinearRegression()
            if polynomial:
                model = make_poly_pipe(model, 10)
            model.fit(X_regularization_demo, y_regularization_demo)
            y_new_regul = model.predict(X_new_regularization_demo)
            plt.plot(
                X_new_regularization_demo,
                y_new_regul,
                linestyle=style,
                color=color,
                linewidth=2,
                label=rf"$\alpha = {alpha}$",
            )
        plt.legend(loc="upper left")
        plt.xlabel("$x_1$")
        plt.axis([0, 3, 0, 3.5])
        plt.grid()

    plt.figure(figsize=(8.5, 4.6))
    _plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$  ", rotation=0)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Genèse du terme de pénalisation

    Pourquoi est-ce que réduire la norme 2 des paramètres empêcherait _nécessairement_ le modèle de trop coller au données ?

    En notant $\mathbf{w}^\star$ le vrai vecteur des paramètres (sans le biais) issu de la fonction de génération, on peut montrer que :

    $$\;\mathbb{E}\big[\|\hat{\mathbf{w}}\|_2^{\,2}\big] \;=\; \|\mathbf{w}^\star\|_2^{\,2} \;+\operatorname{tr}\left(\operatorname{Cov}(\hat{\mathbf{w}})\right)  \quad ; \quad \operatorname{tr}\left(\operatorname{Cov}(\hat{\mathbf{w}})\right)>0$$

    En moyenne sur le bruit, la norme du modèle d'estimation dépasse toujours celle de la vraie fonction, et l'excès vaut exactement la **variance des paramètres estimés**. Et cette variance des paramètres intervient justement dans l'expression de la variance des prédictions, celle-là même de la décomposition biais-variance de la première partie. Et on sait déjà que variance élevée $\Longrightarrow$ overfitting.

    > Ce résultat est vrai en espérance sur le tirage du bruit, ce n'est donc pas une régle absolue. On pourrait trouver une distribution bruitée pour laquelle un modèle qui surapprend a une norme inférieure à la fonction de génération.

    ### Standardisation

    Cette régression étant particulièrement sensible aux poids du modèle, il devient d'autant plus important de standardiser au préalable les features (via `StandardScaler` par exemple) de sorte à ce que les poids des modèles soient seulement proportionnels à leur importance, et non plus à leur ordre de grandeur.

    C'est d'ailleurs vrai pour **tous les modèles de régularisation linéaires** : Ridge, Lasso et Elastic-Net.

    ### Implémentation

    On peut implémenter la régression ridge avec une solution en** forme fermée** :

    $$ \hat{\boldsymbol{\theta}} = \left( \mathbf{X}^{\mathsf T}\mathbf{X} + \alpha\mathbf{D} \right)^{-1} \mathbf{X}^{\mathsf T}\mathbf{y}$$

    Où $\mathbf{D}$ est la matrice identité avec un 0 en haut à gauche.

    /// details | Solution en forme fermée - démonstration (pour aller plus loin)
    On note $\boldsymbol{\theta} = \begin{pmatrix} \theta_0 \\ \mathbf{w} \end{pmatrix}$ et $\mathbf{h} = \begin{pmatrix} h_0 \\ \mathbf{h}_{\mathbf{w}} \end{pmatrix}$. On rappelle l'expression de la fonction de perte : $J(\boldsymbol{\theta}) = \frac{1}{m} \lVert \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \rVert_2^2 + \frac{\alpha}{m} \lVert \mathbf{w} \rVert_2^2$

    On obtient l'expression de $\nabla J(\boldsymbol{\theta})$ en passant par le développement de Taylor à l'ordre 1. On identifie ensuite le gradient via un produit scalaire :

    $$\begin{aligned}
    J(\boldsymbol{\theta} + \mathbf{h}) &= \frac{1}{m} \lVert \mathbf{X}(\boldsymbol{\theta} + \mathbf{h}) - \mathbf{y} \rVert_2^2 + \frac{\alpha}{m} \lVert \mathbf{w} + \mathbf{h}_{\mathbf{w}} \rVert_2^2 \\
    &= \frac{1}{m} \left( \lVert \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \rVert_2^2 + 2 \langle \mathbf{X}^{\mathsf T}(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}), \mathbf{h} \rangle + \lVert \mathbf{X}\mathbf{h} \rVert_2^2 \right) + \frac{\alpha}{m} \left( \lVert \mathbf{w} \rVert_2^2 + 2\langle \mathbf{w}, \mathbf{h}_{\mathbf{w}} \rangle + \lVert \mathbf{h}_{\mathbf{w}} \rVert_2^2 \right)
    \end{aligned}$$

    En posant $\mathbf{D} = \begin{pmatrix} 0 & 0 & \dots & 0 \\ 0 & 1 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & 1 \end{pmatrix}$, on a $\mathbf{D}\boldsymbol{\theta} = \begin{pmatrix} 0 \\ \mathbf{w} \end{pmatrix}$ et par conséquent : $\langle \mathbf{w}, \mathbf{h}_{\mathbf{w}} \rangle = \langle \mathbf{D}\boldsymbol{\theta}, \mathbf{h} \rangle$.

    Finalement :

    $$J(\boldsymbol{\theta} + \mathbf{h}) = J(\boldsymbol{\theta}) + \underbrace{\left\langle \frac{2}{m}\mathbf{X}^{\mathsf T}(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) + \frac{2\alpha}{m}\mathbf{D}\boldsymbol{\theta}, \mathbf{h} \right\rangle}_{dJ_{\boldsymbol{\theta}}(\mathbf{h})} + o(\lVert \mathbf{h} \rVert)$$

    Puisque $dJ_{\boldsymbol{\theta}}(\mathbf{h}) = \langle \nabla J(\boldsymbol{\theta}), \mathbf{h} \rangle$, on conclut :
    $$\nabla J(\boldsymbol{\theta}) = \frac{2}{m} \left( \mathbf{X}^{\mathsf T}(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) + \alpha \mathbf{D}\boldsymbol{\theta} \right)$$

    La fonction de perte restant convexe, tout point critique est un minimum global :

    $$\begin{aligned}
    \nabla J(\hat{\boldsymbol{\theta}}) = \mathbf{0} &\implies \left( \mathbf{X}^{\mathsf T}\mathbf{X} + \alpha\mathbf{D} \right)\hat{\boldsymbol{\theta}} = \mathbf{X}^{\mathsf T}\mathbf{y} \\
    &\implies \hat{\boldsymbol{\theta}} = \left( \mathbf{X}^{\mathsf T}\mathbf{X} + \alpha\mathbf{D} \right)^{-1} \mathbf{X}^{\mathsf T}\mathbf{y}
    \end{aligned}$$
    ///
    <a id="implementation-ridge-scikit"></a>
    Comme vu avec la régression classique, Scikit-Learn ne calcule pas $\hat{\boldsymbol{\theta}}$ directement. Le choix de l'algorithme se fait selon la **nature des données **(denses, creuses, nombre d'observations, etc.)

    L'implémentation se fait très facilement avec le prédicteur `Ridge` :
    """)
    return


@app.cell(hide_code=True)
def _(np):
    rng = np.random.default_rng(seed=42)
    m = 20
    X = 3 * rng.random((m, 1))
    y = 1 + 0.5 * X + rng.standard_normal((m, 1)) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)
    return X, m, y


@app.cell
def _(Ridge, X, y):
    ridge_reg = Ridge(alpha=0.1, solver="auto")
    _ = ridge_reg.fit(X, y)
    return (ridge_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut s'amuser à comparer l'estimation de $\hat{\boldsymbol{\theta}}$ produite par Scikit et celle que l'on obtiendrait avec notre équation fraîchement démontrée.
    """)
    return


@app.cell
def _(X, m, np, ridge_reg, y):
    alpha = 0.1
    A = np.array([[0.0, 0.0], [0.0, 1.0]])
    X_b = np.c_[np.ones(m), X]
    theta = np.linalg.inv(X_b.T @ X_b + alpha * A) @ X_b.T @ y

    print("--- Calcul matriciel brut ---")
    print(f"theta_0 : {theta[0, 0]:.20f}")
    print(f"theta_1 : {theta[1, 0]:.20f}\n")
    print("--- Estimation Scikit ---")
    print(f"theta_0 : {ridge_reg.intercept_[0]:.20f}")
    print(f"theta_1 : {ridge_reg.coef_[0]:.20f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dans ce cas particulier, les estimations coïncident presque parfaitement. On a bon espoir que dans le cas général, la prédiction de Scikit soit **meilleure** et **plus rapide**.

    On peut aussi utiliser la **descente de gradient stochastique** pour calculer $\hat{\boldsymbol{\theta}}$ ; les avantages / inconvénients sont ceux qu'on a abordé dans la première partie. L'implémentation se fait de nouveau via `SGDRegressor` :
    """)
    return


@app.cell
def _(SGDRegressor, X, m, y):
    sgd_reg = SGDRegressor(
        penalty="l2",
        alpha=2 * 0.1 / m,
        tol=None,
        max_iter=1000,
        eta0=0.01,
        random_state=42,
    )
    _ = sgd_reg.fit(X, y.ravel())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Petite subtilité ici ! En précisant `penalty="l2"`, Scikit ajoute $\alpha {{\| \mathbf{w}\|}_2}^{2}$ à la MSE. Puisqu'on veut plutôt ajouter $\displaystyle \frac{\alpha}{m} {{\| \mathbf{w}\|}_2}^{2}$, on doit imposer `alpha=0.1 / m` en divisant bien par $m$.

    ### RidgeCV

    On présente une dernière optimisation pour notre régression Ridge. Elle est semblable au [`Ridge`](#implementation-ridge-scikit) que l'on vient de voir, mais intègre un fine-tuning de l'hyper-paramètre $\alpha$ par validation croisée.

    Pourquoi ne pas se contenter du classique `Ridge` + `GridSearchCV` ?

    Très bonne question, et je vous remercie de l'avoir posée. En fait, `RidgeCV` fait exactement la même chose mais de façon **optimisée pour la régression Ridge** et s'exécute donc **plus rapidement**.

    > Scikit propose de telles optimisations pour de nombreux autres estimateurs. Par exemple `LassoCV` et `ElasticNetCV` existent - on y reviendra en temps voulu.
    """)
    return


@app.cell
def _(RidgeCV, X, np, y):
    ridgecv_reg = RidgeCV(alphas=np.logspace(-3, 3, 100))
    _ = ridgecv_reg.fit(X, y)

    print(f"alpha choisi par RidgeCV : {ridgecv_reg.alpha_:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On impose une grille de `alphas` à parcourir, comme on l'aurait fait avec `GridSearchCV`.

    NB : `logspace` est une alternative logarithmique à `linspace` : `np.logspace(-3, 3, 100)` génère 100 valeurs régulièrement espacées entre 10⁻³ et 10³.

    ---
    ## B. Régression Lasso

    Lasso est une autre version régularisée de la régression linéaire. Sa fonction de perte mobilise la norme 1 :

    $$J(\boldsymbol{\theta}) = \mathrm{MSE}(\boldsymbol{\theta}) + 2 \alpha \sum_{i=1}^{n} | \theta_i | = \mathrm{MSE}(\boldsymbol{\theta}) + 2 \alpha  {{\| \mathbf{w}\|}_1}$$

    Contrairement à Ridge, qui réduit progressivement tous les coefficients sans jamais les annuler exactement, la régression Lasso peut **annuler des poids** : elle effectue automatiquement de la **feature selection**.

    ### Effets de la régularisation

    La régularisation **déforme la fonction de perte**.

    - Elle **déplace le minimum** : il se rapproche de l'origine quand on augmente $\alpha$
    - Elle **modifie la trajectoire de la descente** de gradient.

    La descente se déplace orthogonalement aux lignes de niveau : dans le cas de $\ell_1$, celles-ci sont des **losanges** ($|\theta_1| + |\theta_2| = \text{cste}$) et le gradient vaut $(\pm1, \pm1)$. Les deux coefficients diminuent donc de la **même quantité** à chaque pas (déplacement en diagonale). Le coefficient déjà le plus proche de 0 l'atteint le premier ; une fois nul, la trajectoire longe l'axe et **le coefficient y reste**, car le gradient $\ell_1$ continue de le pousser vers 0 avec une amplitude constante. C'est ainsi que Lasso parvient à imposer des coefficients exactement nuls.

    Côté Ridge, les lignes de niveau de $\ell_2$ sont des **cercles** : le gradient pointe radialement vers l'origine, la descente suit une ligne droite et rétrécit tous les coefficients proportionnellement, sans jamais les annuler.

    Le type de pénalisation influence également la **convergence** :

    - **$\ell_1$ (Lasso)** : oscille un peu autour de l'optimum, car la partie $\nabla \ell_1$ de $\nabla J(\boldsymbol{\theta})$ ne s'approche jamais de 0 (elle vaut $-1$ ou $+1$ pour chaque paramètre).
    - **$\ell_2$ (Ridge)** : les gradients rapetissent à mesure qu'on approche de l'optimum, donc la descente ralentit naturellement. Cela limite les oscillations et fait converger Ridge plus vite que Lasso.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5

    t1s = np.linspace(t1a, t1b, 500)
    t2s = np.linspace(t2a, t2b, 500)
    t1, t2 = np.meshgrid(t1s, t2s)
    T = np.c_[t1.ravel(), t2.ravel()]
    Xr = np.array([[1, 1], [1, -1], [1, 0.5]])
    yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]

    J = (1 / len(Xr) * ((T @ Xr.T - yr.T) ** 2).sum(axis=1)).reshape(t1.shape)

    N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)
    N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)

    t_min_idx = np.unravel_index(J.argmin(), J.shape)
    t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]

    t_init = np.array([[0.25], [-1]])

    def bgd_path(theta, X, y, l1, l2, core=1, eta=0.05, n_iterations=200):
        path = [theta]
        for iteration in range(n_iterations):
            gradients = (
                core * 2 / len(X) * X.T @ (X @ theta - y)
                + l1 * np.sign(theta)
                + l2 * theta
            )
            theta = theta - eta * gradients
            path.append(theta)
        return np.array(path)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 7))

    for i, N, l1, l2, title in ((0, N1, 2.0, 0, "Lasso"), (1, N2, 0, 2.0, "Ridge")):
        JR = J + l1 * N1 + l2 * 0.5 * N2**2

        tr_min_idx = np.unravel_index(JR.argmin(), JR.shape)
        t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

        levels = np.exp(np.linspace(0, 1, 20)) - 1
        levelsJ = levels * (J.max() - J.min()) + J.min()
        levelsJR = levels * (JR.max() - JR.min()) + JR.min()
        levelsN = np.linspace(0, N.max(), 10)

        path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
        path_JR = bgd_path(t_init, Xr, yr, l1, l2)
        path_N = bgd_path(
            theta=np.array([[2.0], [0.5]]),
            X=Xr,
            y=yr,
            l1=np.sign(l1) / 3,
            l2=np.sign(l2),
            core=0,
        )
        ax = axes[i, 0]
        ax.grid()
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.contourf(t1, t2, N / 2.0, levels=levelsN)
        ax.plot(path_N[:, 0], path_N[:, 1], "y--")
        ax.plot(0, 0, "ys")
        ax.plot(t1_min, t2_min, "ys")
        ax.set_title(rf"$\ell_{i + 1}$ penalty")
        ax.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$", rotation=0)

        ax = axes[i, 1]
        ax.grid()
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
        ax.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
        ax.plot(path_N[:, 0], path_N[:, 1], "y--")
        ax.plot(0, 0, "ys")
        ax.plot(t1_min, t2_min, "ys")
        ax.plot(t1r_min, t2r_min, "rs")
        ax.set_title(title)
        ax.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            ax.set_xlabel(r"$\theta_1$")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Pour permettre à Lasso de converger, on le combine généralement à un learning schedule décroissant.

    ### Sous-gradient

    L'implémentation de la régression Lasso **par descente de gradient** se heurte à un problème majeur : la norme $\ell_1$ n'est pas différentiable dès qu'une coordonnée est nulle.

    Une solution typique consiste à remplacer $\nabla \ell_1$ par un **vecteur** $\partial\|\boldsymbol\theta\|_{1}$ **de sous-gradient** :
    $$\partial\|\boldsymbol\theta\|_{1} = \begin{pmatrix} \operatorname{sign}(\theta_1) \\ \operatorname{sign}(\theta_2) \\ \vdots \\ \operatorname{sign}(\theta_n) \end{pmatrix} \quad \quad \text{avec} \ \operatorname{sign}(\theta_i)= \begin{cases} -1 & \text{si } \theta_i < 0, \\ 0 & \text{si } \theta_i = 0, \\ +1 & \text{si } \theta_i > 0. \end{cases}$$

    En plus d'étendre la définition de $\nabla \ell_1$ en tout point ayant une composante nulle, les deux vecteurs $\partial\|\boldsymbol\theta\|_{1}$ et $\nabla \ell_1$ **coïncident lorsqu'aucune composante n'est nulle**.

    /// details| L'intuition du sous-gradient (pour aller plus loin)
    Le sous-gradient est une **généralisation du gradient aux fonctions non différentiables**. On présente l'intuition derrière cette notion avec le cas de la norme $\ell_1$. Considérons une seule variable :
    $$f(\theta)=|\theta|.$$

    Sa dérivée vaut :
    - $+1$ si $\theta>0$,
    - $-1$ si $\theta<0$,
    - elle **n'existe pas** en $\theta=0$.

    Pourtant, on sait intuitivement quoi faire :
    - si $\theta>0$, il faut aller vers la gauche (vers 0) ;
    - si $\theta<0$, il faut aller vers la droite (vers 0).

    C'est exactement ce que traduit

    $$\operatorname{sign}(\theta)=
    \begin{cases}
    -1 & \text{si } \theta<0,\\
    0  & \text{si } \theta=0,\\
    1  & \text{si } \theta>0.
    \end{cases}$$

    Le sous-gradient indique simplement **de quel côté se trouve le minimum**.

    D'autre part, on décide de fixer $\operatorname{sign}(0)=0$  de sorte que **le paramètre reste inchangé lorsqu'il est déjà nul**. Il s'agit d'une convention propre au machine learning : mathématiquement, le sous gradient de $|\theta|$ en 0 vaut $\partial |\theta|_{,\theta=0}=[-1,1]$ (n'importe quelle pente comprise entre -1 et +1 est valable).

    ///

    L'expression en tout point de la fonction de perte Lasso devient :
    $$\mathbf{g}(\boldsymbol\theta)=\nabla_{\boldsymbol\theta}\,\mathrm{MSE}(\boldsymbol\theta)+2\alpha\,\partial\|\boldsymbol\theta\|_{1}.$$

    ### Visualisation

    On s'intéresse, dans le cadre de la régression Lasso, à l'évolution de l'erreur d'entraînement et de validation, [déjà tracés pour la régression Ridge](#learning-curves).

    Par rapport à Ridge  :
    1. L'erreur sur le test set est un peu meilleure : réduction de $0.007$
    2. Moins d'overfitting : réduction de l'écart de généralisation de $0.03$
    """)
    return


@app.cell(hide_code=True)
def _(Lasso, courbe_regularisation, np, orange):
    courbe_regularisation(
        lambda _a: Lasso(alpha=_a, max_iter=300_000),
        np.logspace(-3, 0.5, 60),
        orange,
        "Lasso",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implémentation

    L'implémentation de Lasso avec Scikit est très similaire à celle de la régression Ridge.
    """)
    return


@app.cell
def _(Lasso, X, y):
    lasso_reg = Lasso(alpha=0.1)
    _ = lasso_reg.fit(X, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Cette implémentation mobilise en silence un algorithme de **coordinate descent**. J'invite le lecteur curieux à se renseigner à son sujet.

    Version descente de gradient stochastique :
    """)
    return


@app.cell
def _(SGDRegressor, X, m, y):
    sgd_lasso_reg = SGDRegressor(
        penalty="l1", alpha=0.1 / m, tol=None, max_iter=1000, eta0=0.01, random_state=42
    )
    _ = sgd_lasso_reg.fit(X, y.ravel())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Version fine-tuning intégré de $\alpha$ par validation croisée :
    """)
    return


@app.cell
def _(LassoCV, X, np, y):
    lassocv_reg = LassoCV(alphas=np.logspace(-3, 3, 100))
    _ = lassocv_reg.fit(X, y.ravel())

    print(f"alpha choisi par LassoCV : {lassocv_reg.alpha_:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## <a id="regularisation-modele-polynomiaux"></a>C. Régularisation des modèles polynomiaux

    La régularisation d'un modèle de régression polynomiale peut se faire simplement en **réduisant son degré maximal** (option `degree` de `PolynomialFeatures`).

    Mais puisque l'implémentation de cette régression polynomiale avec Scikit-Learn se fait via une régression linéaire classique (`LinearRegression`) à partir de variables polynomiales (`PolynomialFeatures`), on peut aussi envisager de la régulariser par **pénalisation** : il suffit de substituer `LinearRegression` par son équivalent Lasso ou Ridge, plutôt que de réduire son degré.

    ### Les avantages de la pénalisation

    Avec Ridge ou Lasso, les termes de haut degré restent présents mais leurs coefficients sont **poussés vers zéro** si les données ne justifient pas leur utilisation.

    On bénéficie du meilleur des deux mondes :

    * On exploite les termes complexes quand ils sont utiles
    * On les ignore lorsqu'ils ne le sont pas.

    ### Visualisation

    On compare trois modèles de régression de degré 7 : **polynomial classique**, **Ridge** et **Lasso**.

    Pour que la régularisation ait un intérêt visible, on se place dans un cas où le degré 7 sur-apprend : seulement 20 points bruités, générés par un polynôme de degré 3.

    > Les trois graphes du bas montrent les poids signés appris, degré par degré (features standardisées). Sur chacun, un **trait noir** marque la *vraie* valeur du poids : on voit ainsi si l'estimation est trop haute ou trop basse.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    alpha_ridge_slider = mo.ui.slider(
        steps=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        value=0.03,
        show_value=True,
        label=r"$\alpha_1$ (Ridge)",
    )
    alpha_lasso_slider = mo.ui.slider(
        steps=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
        value=0.02,
        show_value=True,
        label=r"$\alpha_2$ (Lasso)",
    )

    montre_vrai = mo.ui.checkbox(value=False, label="vrai polynôme")
    montre_ols = mo.ui.checkbox(value=True, label="sans pénalité")
    montre_ridge = mo.ui.checkbox(value=False, label="Ridge")
    montre_lasso = mo.ui.checkbox(value=True, label="Lasso")
    return (
        alpha_lasso_slider,
        alpha_ridge_slider,
        montre_lasso,
        montre_ols,
        montre_ridge,
        montre_vrai,
    )


@app.cell(hide_code=True)
def _(
    Lasso,
    LinearRegression,
    PolynomialFeatures,
    Ridge,
    make_poly_pipe,
    np,
    rmse,
):
    poly_ref_degree = 3
    poly_max_degree = 7
    poly_n_train = 20
    poly_noise = 1.2
    poly_lo, poly_hi = -2.5, 2.5
    poly_true_coefs = np.array([0.5, -1.6, -1.1, 1.0])  # 0.5 -1.6x -1.1x^2 +1.0x^3

    def f(x):
        return sum(poly_true_coefs[_k] * x**_k for _k in range(len(poly_true_coefs)))

    _rng = np.random.default_rng(0)
    _xtr = _rng.uniform(poly_lo, poly_hi, poly_n_train)
    poly_X = _xtr.reshape(-1, 1)
    poly_y = f(_xtr) + _rng.normal(0.0, poly_noise, poly_n_train)
    _xte = _rng.uniform(poly_lo, poly_hi, 2000)
    poly_Xte = _xte.reshape(-1, 1)
    poly_yte = f(_xte) + _rng.normal(0.0, poly_noise, 2000)

    def fit_poly(model, degree=poly_max_degree):
        return make_poly_pipe(model, degree).fit(poly_X, poly_y)

    poly_model_ols = fit_poly(LinearRegression())
    poly_coefs_ols = poly_model_ols[-1].coef_
    poly_rmse_ols = (
        rmse(poly_model_ols, poly_X, poly_y),
        rmse(poly_model_ols, poly_Xte, poly_yte),
    )
    poly_rmse_oracle = rmse(
        fit_poly(LinearRegression(), poly_ref_degree), poly_Xte, poly_yte
    )

    _sigma = (
        PolynomialFeatures(poly_max_degree, include_bias=False)
        .fit_transform(poly_X)
        .std(axis=0)
    )
    poly_true_std = (
        np.array(
            [
                poly_true_coefs[1],
                poly_true_coefs[2],
                poly_true_coefs[3],
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        * _sigma
    )

    poly_wmax_ols = 1.1 * np.abs(poly_coefs_ols).max()
    poly_wmax_reg = 1.15 * max(
        np.abs(fit_poly(Ridge(alpha=0.01))[-1].coef_).max(),
        np.abs(fit_poly(Lasso(alpha=0.01, max_iter=300_000))[-1].coef_).max(),
        np.abs(poly_true_std).max(),
    )
    return (
        f,
        fit_poly,
        poly_X,
        poly_Xte,
        poly_hi,
        poly_lo,
        poly_max_degree,
        poly_model_ols,
        poly_n_train,
        poly_rmse_ols,
        poly_rmse_oracle,
        poly_true_std,
        poly_wmax_ols,
        poly_wmax_reg,
        poly_y,
        poly_yte,
    )


@app.cell
def _(Lasso, Ridge, alpha_lasso_slider, alpha_ridge_slider, fit_poly):
    aR = alpha_ridge_slider.value
    aL = alpha_lasso_slider.value
    poly_ridge = fit_poly(Ridge(alpha=aR))
    poly_lasso = fit_poly(Lasso(alpha=aL, max_iter=300_000))
    return aL, aR, poly_lasso, poly_ridge


@app.cell(hide_code=True)
def _(
    aL,
    aR,
    alpha_lasso_slider,
    alpha_ridge_slider,
    f,
    gris,
    mo,
    montre_lasso,
    montre_ols,
    montre_ridge,
    montre_vrai,
    np,
    orange,
    plt,
    poly_X,
    poly_hi,
    poly_lasso,
    poly_lo,
    poly_model_ols,
    poly_n_train,
    poly_ridge,
    poly_y,
    rouge,
    vert,
):
    _courbes = [
        (montre_vrai.value, None, "#1a202c", "vrai polynôme (deg 3)", "--"),
        (montre_ols.value, poly_model_ols, rouge, "sans pénalité", "-"),
        (montre_ridge.value, poly_ridge, vert, f"Ridge ($\\alpha_1$={aR:g})", "-"),
        (montre_lasso.value, poly_lasso, orange, f"Lasso ($\\alpha_2$={aL:g})", "-"),
    ]

    _fig1, _ax = plt.subplots(figsize=(8.5, 4.6))
    _gx = np.linspace(poly_lo - 0.7, poly_hi + 0.7, 500).reshape(-1, 1)
    _ax.axvspan(poly_lo, poly_hi, color=gris, alpha=0.07, zorder=0)
    _ax.text(
        0.5,
        0.03,
        "zone grisée = entraînement · au-delà = extrapolation",
        transform=_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=gris,
    )
    for _visible, _m, _col, _lab, _ls in _courbes:
        if not _visible:
            continue
        if _m is None:
            _ax.plot(
                _gx.ravel(),
                f(_gx.ravel()),
                _ls,
                color=_col,
                lw=1.6,
                label=_lab,
                zorder=2,
            )
        else:
            _ax.plot(
                _gx, _m.predict(_gx), _ls, color=_col, lw=2.2, label=f"{_lab}", zorder=3
            )
    _ax.scatter(
        poly_X.ravel(),
        poly_y,
        s=26,
        color="#2d3748",
        zorder=4,
        label=f"{poly_n_train} points",
    )
    _ax.set_ylim(poly_y.min() - 10, poly_y.max() + 10)
    _ax.set_xlim(poly_lo - 0.7, poly_hi + 0.7)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")

    _ax.legend(fontsize=8, loc="lower right")
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    _fig1.tight_layout()
    plt.close(_fig1)

    mo.vstack(
        [
            mo.hstack(
                [alpha_ridge_slider, alpha_lasso_slider], justify="center", gap=2
            ),
            mo.hstack(
                [montre_vrai, montre_ols, montre_ridge, montre_lasso],
                justify="center",
                gap=1,
            ),
            _fig1,
        ],
        gap=0.8,
    )
    return


@app.cell(hide_code=True)
def _(
    aL,
    aR,
    np,
    orange,
    plt,
    poly_Xte,
    poly_lasso,
    poly_max_degree,
    poly_model_ols,
    poly_ridge,
    poly_true_std,
    poly_wmax_ols,
    poly_wmax_reg,
    poly_yte,
    rmse,
    rouge,
    vert,
):
    _panneaux = [
        ("Régression polynomiale classique", poly_model_ols, rouge, poly_wmax_ols),
        (f"Ridge ($\\alpha_1$={aR:g})", poly_ridge, vert, poly_wmax_reg),
        (f"Lasso ($\\alpha_2$={aL:g})", poly_lasso, orange, poly_wmax_reg),
    ]

    _degres = np.arange(1, poly_max_degree + 1)
    _fig2, _axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for _ax2, (_nom, _m, _col, _wm) in zip(_axes, _panneaux):
        _c = _m[-1].coef_
        _ax2.axhline(0, color="#cbd5e0", lw=1, zorder=1)
        _ax2.bar(
            _degres,
            _c,
            width=0.62,
            color=_col,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
        for _d, _tv in zip(_degres, poly_true_std):
            _ax2.plot(
                [_d - 0.34, _d + 0.34],
                [_tv, _tv],
                color="black",
                lw=2.4,
                zorder=6,
                solid_capstyle="round",
            )
        _ax2.set_title(
            f"{_nom}\nRMSE test {rmse(_m, poly_Xte, poly_yte):.2f}",
            fontsize=10.5,
            pad=8,
        )
        _ax2.set_xticks(_degres)
        _ax2.set_xlabel("degré du terme", fontsize=10)
        _ax2.set_xlim(0.4, poly_max_degree + 0.6)
        _ax2.set_ylim(-_wm, _wm)
        _ax2.grid(axis="y", color="#edf2f7", lw=0.8, zorder=0)
        for _s in ("top", "right"):
            _ax2.spines[_s].set_visible(False)
    _axes[0].set_ylabel("poids signé (base standardisée)", fontsize=10)
    _axes[0].plot([], [], color="black", lw=2.4, label="vrai poids")
    _axes[0].legend(fontsize=8, loc="upper left")
    _fig2.tight_layout()
    plt.close(_fig2)

    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lasso vs. Ridge

    Lasso a un avantage théorique important : il peut mettre des poids à $0$ **exactement** (essayez en imposant $\alpha_2 = 0,02$). Par conséquent, un vecteur de paramètres $\boldsymbol{\theta}$ généré par un "Polynôme degré 3" peut théoriquement être retrouvé par "Polynôme degré 7 + Lasso", alors qu'un "Polynôme degré 7 + Ridge" conserverait des poids - éventuellement faibles, mais non nuls - pour les termes de degré > 3.

    Toutefois, lorsque certaines variables sont fortement corrélées (ce qui est typiquement le cas de $x, x^2, x^3,\ldots$) le Lasso peut être **instable** :

    * il choisit parfois arbitrairement un terme plutôt qu'un autre (essayez en imposant $\alpha_2 = 0,05$)
    * de petites variations des données peuvent modifier la sélection

    C'est une des raisons pour lesquelles **on préfère souvent Ridge pour les bases polynomiales**.

    ### Faut-il cesser de plafonner le choix du degré ?

    Non. Même lorsqu'on utilise Ridge ou Lasso, on **limite généralement le degré maximal** pour des raisons qu'on connaît bien :

    1. Explosion combinatoire : on rappelle qu'en notant $d$ le degré maximum fixé et $n$ le nombre initial de prédicteurs, un appel à `PolynomialFeatures` crée exactement $\frac{\left(n+d\right)!}{n! \ d!}$ features.
    2. Instabilité : Les termes de degré élevés $x^{10}, x^{15}, x^{20}$ sont souvent très corrélés entre eux. Cela augmente :
        * la variance des estimations
        * les problèmes de conditionnement
        * le coût de calcul

    Même Ridge ne compense pas toujours complètement ces difficultés.

    ### La pratique la plus courante

    * On fixe un degré maximal raisonnable (par exemple 3 à 10 selon le problème)
    * On applique une régularisation (Ridge, Lasso ou Elastic Net)
    * Les deux hyperparamètres (`degree` et `alpha`) sont choisis par validation croisée

    ---

    ## D. Régression Elastic Net

    La régression Elastic Net est un compromis entre Ridge et Lasso : le terme de pénalisation qu'elle introduit est une **combinaison convexe** des termes de pénalisation des deux autres.

    $$J(\boldsymbol{\theta}) = \mathrm{MSE}(\boldsymbol{\theta}) + r \left( 2\alpha \sum_{i=1}^{n} |\theta_i| \right) + (1 - r) \left( \frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2 \right) = \mathrm{MSE}(\boldsymbol{\theta}) + 2r\alpha \|\mathbf{w}\|_1 + (1 - r)\frac{\alpha}{m} \|\mathbf{w}\|_2^2$$

    ### Visualisation

    On reprend la même visualisation déjà tracée deux fois.

    Par rapport à Lasso :
    1. L'erreur sur le test set est un peu moins bonne : augmentation de $0.003$
    2. Davantage d'overfitting : augmentation de l'écart de généralisation de $0.02$
    """)
    return


@app.cell(hide_code=True)
def _(ElasticNet, courbe_regularisation, np, violet):
    courbe_regularisation(
        lambda _a: ElasticNet(alpha=_a, l1_ratio=0.5, max_iter=300_000),
        np.logspace(-3, 0.5, 60),
        violet,
        "Elastic-Net",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implémentation

    L'implémentation d'Elastic Net avec Scikit est très similaire à celle de la régression Ridge.
    """)
    return


@app.cell
def _(ElasticNet, X, y):
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    _ = elastic_net.fit(X, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Version fine-tuning intégré de $\alpha$ et $r$ par validation croisée :
    """)
    return


@app.cell
def _(ElasticNetCV, X, np, y):
    elasticnetcv_reg = ElasticNetCV(
        l1_ratio=np.linspace(0.1, 0.9, 9), alphas=np.logspace(-3, 3, 100)
    )
    _ = elasticnetcv_reg.fit(X, y.ravel())

    print(f"alpha choisi par ElasticNetCV : {elasticnetcv_reg.alpha_:.4f}")
    print(f"r choisi par ElasticNetCV : {elasticnetcv_reg.l1_ratio_:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## E. Comparatif des méthodes de régularisation par pénalisation

    | Méthode | Pénalisation | Effet principal | Annule des coefficients ? | Cas d'usage privilégié | Contraintes |
    |:-|:-:|:-|:-:|:-|:-|
    |**Régression linéaire**|Aucune|Ajuste uniquement les données|Non|Données simples, peu de risque d'overfitting|Surapprentissage fréquent|
    |**Ridge**|$\ell_2$|Réduit tous les coefficients de façon progressive|Non|**Choix par défaut** ; variables nombreuses ou corrélées|Ne réalise pas de sélection de variables|
    |**Lasso**|$\ell_1$|Réduit les coefficients et peut les annuler|Oui|Peu de variables réellement utiles ; sélection automatique de features|Peut être instable si les variables sont fortement corrélées ou si $n>m$|
    |**Elastic Net**|$\ell_1+\ell_2$|Combine réduction des coefficients et sélection de variables|Oui|Variables corrélées tout en souhaitant effectuer une sélection|Hyperparamètre supplémentaire (`l1_ratio`) à régler|
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## F. Early stopping

    On présente une méthode de régularisation **sans pénalisation**. Elle s'applique aux modèles dont l'apprentissage est itératif, comme les algorithmes de régression linéaire utilisant la descente de gradient.

    Le principe de l'early stopping est simple : on **cesse d'itérer** dès lors que **l'erreur de validation atteint un minimum**.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, train_errors, val_errors):
    _best_epoch = int(np.argmin(val_errors))
    _best_valid_rmse = val_errors[_best_epoch]
    _n_epochs = len(val_errors)

    plt.figure(figsize=(6, 4))
    plt.annotate(
        "Meilleur modèle",
        xy=(_best_epoch, _best_valid_rmse),
        xytext=(_best_epoch, _best_valid_rmse + 0.5),
        ha="center",
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    plt.plot([0, _n_epochs], [_best_valid_rmse, _best_valid_rmse], "k:", linewidth=2)
    plt.plot(val_errors, "b-", linewidth=3, label="Validation set")
    plt.plot(_best_epoch, _best_valid_rmse, "bo")
    plt.plot(train_errors, "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right")
    plt.xlabel("Époques")
    plt.ylabel("RMSE")
    plt.axis([0, _n_epochs, 0, 3.5])
    plt.grid()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Descente double

    Il arrive que l'erreur de validation atteigne un premier minimum, puis augmente avant de redescendre jusqu'à une valeur inférieure à ce premier minimum. S'arrêter systématiquement au premier minimum peut donc conduire à des configurations sous-optimales.

    Ce phénomène, fréquent lors de l'entraînement des réseaux de neurones, est appelé **double descent**.

    Un bon moyen de s'en affranchir consiste à entraîner le modèle un grand nombre d'époques, et à ne garder ultimement que la version associée à la **plus faible erreur de validation**. Cela nécessite d'enregistrer régulièrement l'état du modèle, ce que l'on fait avec `deepcopy()` dans l'implémentation ci-dessous.

    > Cette implémentation **avec mémoire** se montre également utile dans le cas de descentes stochastiques ou mini-batch, pour lesquelles il est souvent difficile d'identifier un minimum, la courbe de l'erreur d'entraînement étant généralement très irrégulière.

    ### Implémentation

    Le code qui suit propose une implémentation simple de l'early stopping. On n'introduit pas de nouvel objet Python, sinon la méthode `partial_fit()` déjà présentée dans la première partie de ce chapitre.

    La régularisation est appliquée ici à une descente de gradient stochastique (mais on aurait pu choisir une descente batch ou mini-batch) sur une régression polynomiale de degré 90.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    _rng = np.random.default_rng(seed=42)
    _m = 200
    _X = 6 * _rng.random((_m, 1)) - 3
    _y = 0.5 * _X**2 + _X + 2 + _rng.standard_normal((_m, 1))
    X_train, y_train = _X[: _m // 2], _y[: _m // 2, 0]
    X_valid, y_valid = _X[_m // 2 :], _y[_m // 2 :, 0]
    return X_train, X_valid, y_train, y_valid


@app.cell
def _(
    PolynomialFeatures,
    SGDRegressor,
    StandardScaler,
    X_train,
    X_valid,
    deepcopy,
    make_pipeline,
    rmse,
    y_train,
    y_valid,
):
    _preprocessing = make_pipeline(
        PolynomialFeatures(degree=90, include_bias=False), StandardScaler()
    )
    _X_train_prep = _preprocessing.fit_transform(X_train)
    # Pas de `fit()` ici : on réutilise les moyennes et écarts-types calculés sur X_train
    _X_valid_prep = _preprocessing.transform(X_valid)

    _sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
    _n_epochs = 500
    # On initialise la meilleure RMSE de validation à +∞
    # ainsi la première vraie RMSE obtenue sera nécessairement meilleure
    _best_valid_rmse = float("inf")

    # Deux listes dans lesquelles on va stocker l'évolution des erreurs
    # c'est avec ces deux listes qu'est tracée la figure un peu plus haut
    train_errors, val_errors = [], []
    best_model = deepcopy(_sgd_reg)

    for _epoch in range(_n_epochs):
        _sgd_reg.partial_fit(_X_train_prep, y_train)  # On itère une seule époque
        _val_error = rmse(
            _sgd_reg, _X_valid_prep, y_valid
        )  # Mesure performance actuelle sur jeu de validation
        if _val_error < _best_valid_rmse:
            _best_valid_rmse = _val_error
            best_model = deepcopy(_sgd_reg)
        _train_error = rmse(
            _sgd_reg, _X_train_prep, y_train
        )  # Mesure performance actuelle sur jeu entraînement
        val_errors.append(_val_error)
        train_errors.append(_train_error)
    return train_errors, val_errors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # II. Régression logistique

    Après avoir étudié la régression linéaire, les différentes façons de l'implémenter, de l'évaluer et de la régulariser, on se propose maintenant d'étudier la **régression logistique**.

    Comme son nom ne l'indique _pas_, la régression logistique est bien un algorithme de **classification binaire** : il sert à déterminer si une instance appartient ou non à une classe donnée.

    ## A. Estimation des probabilités

    Si l'on parle de régression logistique, c'est parce que le modèle repose d'abord sur une combinaison linéaire des prédicteurs.

    Cette quantité pouvant prendre n'importe quelle valeur réelle, elle ne peut pas être directement interprétée comme une probabilité. On lui applique donc la **fonction logistique**, aussi appelée **sigmoïde logistique**, fonction bijective de $\mathbb{R}$ dans $\left[0,1\right]$ :

    $$\sigma : t \longmapsto \frac{1}{1+\exp(-t)}$$

    On note alors $\hat{p}$ la probabilité qu'une instance $\mathbf{x}$ appartienne à la classe étudiée.

    $$\hat{p} = \sigma\left(\boldsymbol{\theta}^{\top}\mathbf{x}\right)$$

    Le classifieur étant **binaire**, notre modèle doit renvoyer $0$ ou $1$ selon si l'instance est classée positivement ou négativement. Il suffit pour cela de fixer un seuil, par exemple $0.5$ :

    $$ \hat{y} = \begin{cases} 0 & \text{si } \hat{p} < 0.5 \\ 1 & \text{si } \hat{p} \geq 0.5 \end{cases}$$

    ---

    ## B. Entraînement et fonction de coût

    ### Principe

    Il nous faut maintenant entraîner notre modèle, c'est-à-dire déterminer $\boldsymbol{\theta}$ à partir du training set.

    Pour trouver un estimateur de $\boldsymbol{\theta}$, une méthode possible consiste à utiliser l'**estimateur du maximum de vraisemblance** $\hat{\boldsymbol{\theta}}_{\text{EMV}}$ :

    $$\hat{\boldsymbol{\theta}}_{\text{EMV}} = \arg\max_{\boldsymbol{\theta}} \ \ln \mathcal{L}(\boldsymbol{\theta} ; \mathbf{y})$$

    1. On peut alors essayer d'obtenir une **solution en forme fermée**, c'est-à-dire une expression explicite de $\hat{\boldsymbol{\theta}}_{\text{EMV}}$.
    2. Si l'on n'y parvient pas (ce n'est pas toujours possible), on se contente d'expliciter une fonction de coût $J(\boldsymbol{\theta})$. En effet :

    $$\begin{aligned} \hat{\boldsymbol{\theta}}_{\text{EMV}} &= \arg\max_{\boldsymbol{\theta}} \ \ln \mathcal{L}(\boldsymbol{\theta} ; \mathbf{y}) \\ &= \arg\min_{\boldsymbol{\theta}} \ \underbrace{-\ln \mathcal{L}(\boldsymbol{\theta} ; \mathbf{y})}_{J(\boldsymbol{\theta})} \end{aligned}$$

    > Si $J(\boldsymbol{\theta})$ est convexe, la descente de gradient est d'ailleurs garantie de converger (pour un pas $\eta$ ni trop petit, ni trop grand).

    ### Notations

    On note, pour la $i$-ème instance du training set :

    - $\mathbf{X}^{(i)}$ la variable aléatoire associée au vecteur des **attributs** (features) de la $i$-ème instance, et $\mathbf{x}^{(i)}$ sa réalisation.
    - $Y^{(i)} \in \{0,1\}$ la variable aléatoire associée à son **étiquette** (label), et $y^{(i)}$ sa réalisation.
    - $\hat{p}_{i}= \sigma\left(\boldsymbol{\theta}^{\top} \mathbf{x}^{(i)}\right)$ la probabilité estimée par le modèle.

    > $\mathbf{X}^{(i)}$ désigne ici la variable aléatoire dont $\mathbf{x}^{(i)}$ est une réalisation, à ne pas confondre avec la matrice de conception.

    ### Construction de la vraisemblance

    Comment procéder ensuite ? Quelles variables aléatoires considérer ? Eh bien **pour construire la vraisemblance, on doit chercher quelle loi des données observées dépend du paramètre** $\boldsymbol{\theta}$.

    /// details | Pour aller plus loin

    La section précédente nous dit deux choses :

    - La probabilité qu'une instance $\mathbf{x}$ appartienne à la classe étudiée vaut $\hat{p}$.
    - $\hat{y} = \begin{cases} 0 & \text{si } \hat{p} < 0.5 \\ 1 & \text{si } \hat{p} \geq 0.5 \end{cases}$

    Le second énoncé correspond à la **règle de décision** du modèle, propre aux problèmes de classification : c'est elle qui convertit une probabilité en une classe prédite.

    Or la vraisemblance correspond à la probabilité que le modèle attribue aux données effectivement observées : la maximiser revient à chercher les $\boldsymbol{\theta}$ qui rendent les classes réellement observées les plus **probables**, et non ceux qui rapprochent les **prédictions** de la réalité.

    C'est donc le premier énoncé, **le seul des deux à décrire une loi**, qui répond à notre question.

    ///

    On sait d'après la section précédente que la probabilité qu'une instance $\mathbf{x}$ appartienne à la classe étudiée vaut $\hat{p}$.

    Ainsi, pour tout $i \in \llbracket 1, m \rrbracket$ :

    $$\begin{cases} \mathbb{P}_{\boldsymbol{\theta}}\left(Y^{(i)} = 1 \mid \mathbf{X}^{(i)} = \mathbf{x}^{(i)}\right) &= \hat{p}_{i}\\ \mathbb{P}_{\boldsymbol{\theta}}\left(Y^{(i)} = 0 \mid \mathbf{X}^{(i)} = \mathbf{x}^{(i)}\right) &= 1 - \hat{p}_{i} \end{cases}$$

    On reconnaît la loi de Bernoulli :

    $$Y^{(i)} \mid \mathbf{X}^{(i)} = \mathbf{x}^{(i)}\sim \mathcal{B}\left(\hat{p}_{i}\right)$$

    > Dans un souci de lisibilité, on s'affranchira par la suite de la notation conditionnelle.

    En supposant les instances du training set indépendantes, les variables $Y^{(i)}$ sont conditionnellement indépendantes sachant les vecteurs d’attributs $\mathbf{x}^{(i)}$ observés. On peut donc écrire :

    $$\begin{aligned} \mathcal{L}\left(\boldsymbol{\theta} ; y^{(1)},\ldots,y^{(m)} \right) &= \prod_{i=1}^{m} \mathbb{P}_{\boldsymbol{\theta}}\left(Y^{(i)} = y^{(i)} \right) \\ &= \prod_{i=1}^{m} \left(\hat{p}_{i}\right)^{y^{(i)}}\left(1-\hat{p}_{i}\right)^{1-y^{(i)}} \end{aligned}$$

    Par conséquent,

    $$\ln \mathcal{L}\left(\boldsymbol{\theta} ; y^{(1)},\ldots,y^{(m)} \right) =  \sum_{i=1}^{m} \left[ y^{(i)} \ln \hat{p}_{i}(\boldsymbol{\theta}) + \left(1-y^{(i)}\right) \ln\left(1 - \hat{p}_{i}(\boldsymbol{\theta})\right) \right]$$

    Donc,

    $$\begin{aligned} \hat{\boldsymbol{\theta}}_{\text{EMV}} &= \arg\max_{\boldsymbol{\theta}} \ \ln \mathcal{L}\left(\boldsymbol{\theta} ; y^{(1)},\ldots,y^{(m)} \right) \\ &= \arg\max_{\boldsymbol{\theta}} \ \sum_{i=1}^{m} \left[ y^{(i)} \ln \hat{p}_{i}(\boldsymbol{\theta}) + \left(1-y^{(i)}\right) \ln\left(1 - \hat{p}_{i}(\boldsymbol{\theta})\right) \right] \\ &= \arg\min_{\boldsymbol{\theta}} \ \underbrace{-\sum_{i=1}^{m} \left[ y^{(i)} \ln \hat{p}_{i}(\boldsymbol{\theta}) + \left(1-y^{(i)}\right) \ln\left(1 - \hat{p}_{i}(\boldsymbol{\theta})\right) \right]}_{J(\boldsymbol{\theta})} \end{aligned}$$

    ### La log loss

    On rencontre plus fréquemment cette fonction de coût sous sa forme **moyennée**, appelée **log loss**  :

    $$J(\boldsymbol{\theta}) = -\frac{1}{m}\sum_{i=1}^{m} \left[ y^{(i)} \ln \hat{p}_{i}(\boldsymbol{\theta}) + \left(1-y^{(i)}\right) \ln\left(1 - \hat{p}_{i}(\boldsymbol{\theta})\right) \right]$$

    Ce facteur $\frac{1}{m}$ n'est pas une nécessité mathématique : pour tout $\lambda > 0$, les fonctions $J$ et $\lambda J$ atteignent leur minimum au même endroit, donc $\hat{\boldsymbol{\theta}}_{\text{EMV}}$ est inchangé. Il s'agit d'une **convention**, mais qui a son intérêt :

    1. **Comparer deux erreurs.** Moyennée, la log loss s'interprète comme une erreur **par instance**, exactement comme la MSE. Sans le $\frac{1}{m}$, l'erreur d'entraînement (somme sur $m$ instances) et l'erreur de validation (somme sur $p$ instances) ne vivraient plus sur la même échelle : ni l'écart de généralisation ni les learning curves n'auraient de sens.
    2. **Conserver un taux d'apprentissage.** Le gradient est moyenné lui aussi. Sans cette normalisation, $\lVert \nabla J \rVert$ croîtrait proportionnellement à $m$ : un $\eta$ réglé sur un training set donné serait à re-régler dès qu'on en change la taille, et une descente stochastique (un seul terme) n'opérerait plus du tout à la même échelle qu'une descente batch ($m$ termes).
    3. **Régulariser.** C'est l'argument déjà rencontré avec Ridge : dans une fonction de coût pénalisée, seul le **rapport** entre le terme d'ajustement et le terme de pénalisation détermine le compromis. Les deux doivent donc être normalisés de la même façon pour que $\alpha$ garde le même sens quelle que soit la taille du training set.

    ---

    ## C. Frontières de décision

    ### Le Palmer Penguins Dataset
    Plusieurs datasets intégrés à `sklearn.datasets` permettent de mettre en pratique des modèles de classification  (Wine, Breast Cancer Wisconsin, Iris, ..).

    On décide toutefois d'utiliser le **Palmer Penguins Dataset**, bien que non nativement intégré à `sklearn.datasets`, pour la simple et bonne raison que les manchots, c'est quand même sympa.
    """)
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    penguins = fetch_openml(name="penguins", version=1, as_frame=True)
    penguins.frame.head(3)
    return (penguins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ce dataset concentre différentes caractéristiques (île d'origine, longueur et hauteur du bec, longueur de la nageoire, masse et sexe) de 344 manchots issues de trois espèces différentes (Adélie, Chinstrap et Gentoo).
    """)
    return


@app.cell
def _(penguins, plt):
    y_logreg = (penguins.target == "Adelie")

    plt.figure(figsize=(4, 3))
    y_logreg.value_counts().rename(index={True: "Adélie", False: "Non Adélie"}).plot.bar(rot=0, grid=True)
    plt.xlabel(None)
    plt.ylabel("Nombre de manchots")
    plt.show()
    return (y_logreg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Préparation du modèle

    On va essayer d'entraîner un modèle de régression logistique capable de **prédire si un manchot est un Pygoscelis adeliae** (manchot Adélie) à partir de certaines de ses **caractéristiques physiques**.

    Pour simplifier l'application pédagogique de notre modèle, on décide de s'affranchir des variables catégorielles (sexe et île d'origine).

    > Si on envisageait de les laisser, il faudrait transformer l'île d'origine en variable numérique par un procédé de **one-hot encoding**, comme expliqué au chapitre 2.
    """)
    return


@app.cell
def _(penguins):
    penguins.frame.isna().any(axis=1).sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    10 instances du dataset comportent **des données manquantes** (NaN). Plutôt que de les supprimer, on décide de les **inférer** avec `KNNImputer`. Concrètement, chaque valeur manquante sera remplacée par **la moyenne des $k$ plus proches voisins**. La proximité entre instances sera calculée à partir des autres features numériques.

    Toutefois, `body_mass_g` risque de peser énormément dans la notion de proximité simplement parce que ses valeurs sont numériquement beaucoup plus grandes. On résoud simplement ce problème en **standardisant les données**.
    """)
    return


@app.cell
def _(StandardScaler, make_pipeline):
    from sklearn.impute import KNNImputer
    from sklearn.linear_model import LogisticRegression

    log_reg = make_pipeline(
        StandardScaler(),
        KNNImputer(
            n_neighbors=5,         # k = 5
            weights="uniform",     # les k voisins contribuent tous de manière égale
            metric="nan_euclidean"),
        LogisticRegression(random_state=42)
    )
    return (log_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Régression logistique (un seul prédicteur)

    On essaye d'abord d'entraîner notre modèle avec **un seul prédicteur**, par exemple la longueur du bec (`culmen_length_mm`).
    """)
    return


@app.cell
def _(log_reg, penguins, y_logreg):
    from sklearn.model_selection import train_test_split

    X_longueur_bec = penguins.data[["culmen_length_mm"]]

    # Le ratio par défaut de train_test_split() est de 75% train et 25% test
    X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(
        X_longueur_bec,
        y_logreg,
        test_size=0.25,
        random_state=42,
        stratify=y_logreg
    )

    _ = log_reg.fit(X_train_1d, y_train_1d)
    return X_longueur_bec, X_train_1d, train_test_split, y_train_1d


@app.cell(hide_code=True)
def _(
    X_longueur_bec,
    X_train_1d,
    afficher_metriques,
    log_reg,
    np,
    plt,
    seuil_proba,
    y_train_1d,
):
    import pandas as pd
    from matplotlib.lines import Line2D

    _feature = "culmen_length_mm"

    _x_min = X_longueur_bec[_feature].min()
    _x_max = X_longueur_bec[_feature].max()
    _marge = 0.05 * (_x_max - _x_min)
    _X_new = pd.DataFrame(
        np.linspace(_x_min - _marge, _x_max + _marge, 1000),
        columns=[_feature],
    )

    _y_proba = log_reg.predict_proba(_X_new)
    _p_adelie = _y_proba[:, 1]
    _x_grille = _X_new.to_numpy().ravel()

    # Frontière = endroit où (p - seuil) change de signe
    _au_dessus = _p_adelie >= seuil_proba.value
    _croisements = np.flatnonzero(np.diff(_au_dessus.astype(int)) != 0)
    frontiere = (
        0.5 * (_x_grille[_croisements[0]] + _x_grille[_croisements[0] + 1])
        if _croisements.size else None
    )

    _sens = 1 if _p_adelie[-1] > _p_adelie[0] else -1

    # --- NOUVEAU : métriques au seuil courant, sur les points affichés ---
    _pred = log_reg.predict_proba(X_train_1d)[:, 1] >= seuil_proba.value
    _vrai = y_train_1d.to_numpy()
    _vp = np.sum(_pred & _vrai)                     # vrais positifs
    _fp = np.sum(_pred & ~_vrai)                    # faux positifs
    _fn = np.sum(~_pred & _vrai)                    # faux négatifs
    _precision = _vp / (_vp + _fp) if (_vp + _fp) else float("nan")
    _rappel = _vp / (_vp + _fn) if (_vp + _fn) else float("nan")

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(_X_new, _y_proba[:, 0], "b--", linewidth=2, label="Non Adélie")
    _ax.plot(_X_new, _p_adelie, "g-", linewidth=2, label="Adélie")

    if frontiere is not None:
        _dx = 0.06 * (_x_max - _x_min)
        _ax.plot([frontiere, frontiere], [0, 1], "k:", linewidth=2,
                 label="Frontière de décision")
        _ax.arrow(frontiere, 0.92, _sens * _dx, 0,
                  head_width=0.05, head_length=_dx / 2, fc="g", ec="g")
        _ax.arrow(frontiere, 0.08, -_sens * _dx, 0,
                  head_width=0.05, head_length=_dx / 2, fc="b", ec="b")

    _x_faux = X_train_1d.loc[~y_train_1d, _feature]
    _x_vrai = X_train_1d.loc[y_train_1d, _feature]
    _ax.plot(_x_faux, np.zeros(len(_x_faux)), "bs", alpha=0.3)
    _ax.plot(_x_vrai, np.ones(len(_x_vrai)), "g^", alpha=0.3)

    _ax.set_xlabel("Longueur du bec (mm)")
    _ax.set_ylabel("Probabilité")
    _ax.axis([_x_min - _marge, _x_max + _marge, -0.02, 1.02])
    _ax.grid(True)

    # --- NOUVEAU : légende augmentée si la case est cochée ---
    _handles, _labels = _ax.get_legend_handles_labels()
    if afficher_metriques.value:
        _vide = Line2D([], [], linestyle="none")     # poignée invisible
        _handles += [_vide, _vide]
        _labels += [f"Précision : {_precision:.1%}", f"Rappel : {_rappel:.1%}"]
    _ax.legend(_handles, _labels, loc="center left")

    _fig
    return frontiere, pd


@app.cell
def _(frontiere):
    frontiere
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On lit sur le graphique que les manchots Adélie (triangles) ont un bec dont la longueur s'étale d'environ $32$ à $46$ mm, quand celui des autres espèces (carrés) mesure plutôt entre $41$ et $60$ mm. Les deux distributions se chevauchent : c'est précisément dans cette zone que le modèle doit trancher.

    On distingue trois régimes:
    1. En dessous de $38$ mm : le modèle attribue une probabilité élevée à la classe « Adélie ».
    2. Au-dessus de $48$ mm : la probabilité bascule vers la classe « Non Adélie ».
    3. Entre les deux : les deux probabilités sont comparables, le modèle est incertain.

    Là où `predict_proba()` restitue $\hat p$ telle quelle, `predict()` applique **la règle de décision au seuil de 50%** : il renvoie simplement la classe la plus probable. Ainsi, un manchot dont le bec mesure $42{,}9$ mm sera classé « Adélie » avec exactement la même assurance apparente qu'un manchot à $33$ mm. La règle de décision **écrase toute l'information de confiance** contenue dans $\hat p$

    ### Du modèle à la réalité métier

    « _La règle de décision écrase toute l'information de confiance contenue dans $\hat p$._ »

    Il y a donc un fort enjeu à la régler correctement et à le faire **à la lumière des enjeux métier **: un faux négatif en dépistage médical ne coûte pas ce que coûte un faux positif. C'est exactement le **compromis précision/recall** du chapitre 3.

    Le slider ci-dessous déplace la frontière de décision tracée plus haut.. Il permet de changer le seuil de probabilité à partir duquel une instance est classée positivement. Il était initialement fixé à 50%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    seuil_proba = mo.ui.slider(start=0, stop=1, step=0.001, value=0.500, label=r"Règle de décision (probabilité)")

    afficher_metriques = mo.ui.checkbox(value=False, label="Afficher précision et rappel")

    mo.vstack([seuil_proba, afficher_metriques])
    return afficher_metriques, seuil_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Précision, recall et F1-score

    Le chapitre 3 détaille la difficulté d'évaluer les performances des modèles de classification. On décide ici d'implémenter une méthode (parmi d'autres !) pour optimiser le seuil de décision.

    Quelques rappels sur les métriques d'évaluation des classifieurs :

    - La précision d'un classifieur quantifie la proportion de **prédictions positives correctes**.
    - Le recall quantifie la proportion d'**instances positives** qui sont **classées comme telles** par le modèle.
    - Le F1-score correspond à la **moyenne harmonique **de la précision et du recall

    Le code suivant entraîne l'estimateur `TunedThresholdClassifierCV` par validation croisée. Il traite le seuil de décision comme un hyperparamètre pour maximiser le F1-score.

    > À défaut de pouvoir maximiser simultanément précision et recall et en l'absence d'exigences métier évidentes (dépistage médical par exemple), on se contente souvent de maximiser le F1-score.
    """)
    return


@app.cell
def _(X_train_1d, log_reg, y_train_1d):
    from sklearn.model_selection import TunedThresholdClassifierCV
    from sklearn.metrics import f1_score

    tuned_log_reg = TunedThresholdClassifierCV(log_reg, scoring="f1")
    _ = tuned_log_reg.fit(X_train_1d, y_train_1d)

    print(f"Seuil choisi par validation croisée (F1) : {tuned_log_reg.best_threshold_:.4f}")
    print(f"F1-score correspondant : {tuned_log_reg.best_score_:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > NB : `seuil_proba` n'est pas un hyperparamètre à strictement parler. En effet, `predict()` applique la règle de décision au seuil de 50%, et ce seuil n'est pas nativement modifiable ; on est obligé de le faire manuellement. `TunedThresholdClassifierCV` n'est pas donc pas un fine-tuner classique et doit recréer en interne ce paramètre. C'est pourquoi le code ci-dessus ne précise pas la localisation d'un hyper-paramètre dans la pipeline, comme on l'avait fait au chapitre 2.

    ### Régression logistique (deux prédicteurs)

    On entraîne maintenant notre modèle avec **deux prédicteurs** : la longueur du bec (`culmen_length_mm`) et sa profondeur (`culmen_depth_mm`).

    Le code est sensiblement le même.
    """)
    return


@app.cell
def _(log_reg, penguins, train_test_split, y_logreg):
    import copy

    X_2d = penguins.data[["culmen_length_mm", "culmen_depth_mm"]]

    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d,
        y_logreg,
        test_size=0.25,
        random_state=42,
        stratify=y_logreg
    )

    log_reg_2 = copy.deepcopy(log_reg)
    _ = log_reg_2.fit(X_train_2d, y_train_2d)
    return X_2d, X_train_2d, log_reg_2, y_train_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sur la figure ci-dessous, la ligne noire en pointillés marque les points où le modèle estime $\hat p = 0{,}5$ : c'est la frontière de décision.

    > La frontière de décision est linéaire. C'est logique, $\{\mathbf x : \boldsymbol\theta^\top\mathbf x = 0\}$ est une droite dans le plan (longueur du bec, hauteur du bec).

    Chacune des lignes parallèles représente les points où le modèle affiche une probabilité donnée, de $0.90$  à $0.15$. Tous les manchots situés au-delà de la ligne $0.90$ ont, selon le modèle, plus de $90\%$ de chances d'être des Adélie.
    """)
    return


@app.cell(hide_code=True)
def _(X_2d, X_train_2d, log_reg_2, np, pd, plt, y_train_2d):
    _feature_x, _feature_y = "culmen_length_mm", "culmen_depth_mm"

    _x_min, _x_max = X_2d[_feature_x].min(), X_2d[_feature_x].max()
    _y_min, _y_max = X_2d[_feature_y].min(), X_2d[_feature_y].max()
    _marge_x = 0.05 * (_x_max - _x_min)
    _marge_y = 0.05 * (_y_max - _y_min)

    _x0, _x1 = np.meshgrid(
        np.linspace(_x_min - _marge_x, _x_max + _marge_x, 300),
        np.linspace(_y_min - _marge_y, _y_max + _marge_y, 300),
    )
    _X_new = pd.DataFrame(
        np.c_[_x0.ravel(), _x1.ravel()], columns=[_feature_x, _feature_y]
    )

    _y_proba = log_reg_2.predict_proba(_X_new)
    _zz = _y_proba[:, 1].reshape(_x0.shape)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _x_faux = X_train_2d.loc[~y_train_2d]
    _x_vrai = X_train_2d.loc[y_train_2d]
    _ax.plot(_x_faux[_feature_x], _x_faux[_feature_y], "bs")
    _ax.plot(_x_vrai[_feature_x], _x_vrai[_feature_y], "g^")

    _contour = _ax.contour(_x0, _x1, _zz, cmap=plt.cm.brg)
    _ax.clabel(_contour, inline=1)

    _ax.contour(_x0, _x1, _zz, levels=[0.5], colors="k", linestyles="--", linewidths=3)

    _ax.text(
        _x_min + 0.05 * (_x_max - _x_min), _y_min + 0.5 * (_y_max - _y_min),
        "Adélie", color="g", ha="center",
    )
    _ax.text(
        _x_max - 0.15 * (_x_max - _x_min), _y_max - 0.45 * (_y_max - _y_min),
        "Non Adélie", color="b", ha="center",
    )
    _ax.set_xlabel("Longueur du bec (mm)")
    _ax.set_ylabel("Hauteur du bec (mm)")
    _ax.axis([_x_min - _marge_x, _x_max + _marge_x, _y_min - _marge_y, _y_max + _marge_y])
    _ax.grid()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Régularisation

    Comme les autres modèles linéaires, la régression logistique peut être régularisée avec une pénalité $\ell_1$ ou $\ell_2$. Scikit-Learn ajoute par défaut une pénalité $\ell_2$.

    L'hyperparamètre qui contrôle la force de régularisation d'une telle `LogisticRegression` Scikit-Learn n'est pas $\alpha$ (comme pour les autres modèles linéaires), mais **son inverse** : $C$. Plus la valeur de $C$ est élevée, moins le modèle est régularisé.

    ---

    ## D. Régression softmax

    ### Contexte

    Le modèle de régression logistique, qui est un classifieur binaire, peut être généralisé pour devenir un classifieur multi-classes. On parle alors de régression logistique multinomiale, ou de **régression softmax**.

    > Attention ! Un classifieur multiclasse prédit une seule classe à la fois, ce n'est pas un classifieur multi-label. Il doit donc être déployé dans un contexte où les classes sont mutuellement exclusives.

    ### Principe de fonctionnement

    On considère un problème de classification avec $K$ classes différentes.

    Pour une instance $\mathbf{x}$, la régression softmax calcule **sa probabilité d'appartenance à chaque classe** : $\hat{p}_{1}(\mathbf{x})$, $\hat{p}_{2}(\mathbf{x})$, .., $\hat{p}_{K}(\mathbf{x})$. On a donc :

    $$\sum_{k=1}^{K}{\hat{p}_{k}(\mathbf{x})} = 1$$

    Comme avec la régression logistique, ces probabilités correspondent à l'image d'un produit scalaire par une fonction $\sigma_k$ :

    $$\hat{p}_k(\mathbf{x} ; \boldsymbol{\Theta}) = \sigma_k\left({\mathbf{x}}^{\top} \boldsymbol{\theta}^{(k)}\right)$$

    > Puisqu'il y autant de vecteurs de paramètres $\boldsymbol{\theta}^{(j)}$ qu'il y a de classes, on les combine généralement dans une matrice $\boldsymbol{\Theta}$ sous forme de vecteur lignes.

    La classe prédite est celle associée à la probabilité la plus haute :

    $$\hat{y} = \underset{k}{\arg\max}\; \hat{p}_{k}(\mathbf{x})$$

    ### Fonction softmax

    La régression logistique mobilisait la **fonction logistique** :

    $$\sigma : t \longmapsto \frac{1}{1+\exp(-t)}$$

    La régression softmax, qui utilise la **fonction softmax**, fait les choses un peu différement.

    1. Cette fonction que l'on applique dépend elle-même de la classe d'indice $k$ dont on estime la probabilité d'appartenance. On la note donc plus volontiers $\sigma_k$.

    2. Puisque les probabilités d'appartenance doivent sommer à 1, cette fonction dépend également des autres _scores_ $s_j(\mathbf{x})={\mathbf{x}}^{\top} \boldsymbol{\theta}^{(j)}$ (pas seulement celui de la classe d'indice $k$).

    Son expression est la suivante :

    $$\hat{p}_k = \sigma_k(\mathbf{x} ; \boldsymbol{\Theta}) = \frac{\exp({\mathbf{x}}^{\top} \boldsymbol{\theta}^{(k)})}{\sum_{j=1}^{K}\exp({\mathbf{x}}^{\top} \boldsymbol{\theta}^{(j)})}$$

    ### Log-odds

    On retrouve souvent la notation $s_k(\mathbf{x})={\mathbf{x}}^{\top} \boldsymbol{\theta}^{(k)}$. Ces scores $s_k$ sont appelés **logits**. Les probabilités $\hat{p}_k$ deviennent alors :

    $$\hat{p}_k = \sigma_k(\mathbf{s}(\mathbf{x})) = \frac{\exp(s_k)} {\sum_{j=1}^K\exp(s_j)}$$

    C'est la façon **rigoureuse** de définir $\sigma_k$.

    > Puisqu'il y autant de scores $s_k(\mathbf{x})$ qu'il y a de classes, on les combine généralement dans un vecteur $\mathbf{s}(\mathbf{x})$.

    /// details | Genèse du softmax (pour aller plus loin)

    Dans le contexte de la régression logistique classique, on avait justifié le choix de la fonction logistique avec l'argument qu'elle effectuait une bijection de $\mathbb{R}$ dans $]0,1[$.

    En réalité, l'idée fondamentale de la régression logistique, c'est que les logarithmes des rapports de probabilités d'appartenances aux classes doivent dépendre linéairement des prédicteurs.

    Ce principe suffit à construire la fonction softmax et la fonction logistique :

    $$\ln\left( \frac{p_k(\mathbf x)} {p_j(\mathbf x)} \right) = \mathbf x^\top \left( \theta^{(k)}-\theta^{(j)} \right).$$

    En posant $s_k(\mathbf x)=\mathbf x^\top\theta^{(k)}$, on obtient :

    $$\ln\left(\frac{p_k}{p_j}\right)=s_k-s_j.$$

    Donc :

    $$\frac{p_k}{p_j} = e^{s_k-s_j} = \frac{e^{s_k}}{e^{s_j}}.$$

    Les probabilités doivent donc être proportionnelles à $e^{s_k}$. On peut écrire :

    $$p_k=\lambda e^{s_k}$$

    Or :

    $$\sum_{k=1}^K p_k = 1.$$

    Donc :

    $$\lambda \sum_{k=1}^K e^{s_k} = 1$$

    et ainsi :

    $$\lambda = \frac{1}{\sum_{j=1}^K e^{s_j}}.$$

    Finalement :

    $$p_k = \frac{e^{s_k}}{\sum_{j=1}^K e^{s_j}}$$

    Et voilà la softmax. On peut même montrer que la fonction logistique en est un cas particulier.

    Dans le cadre de la régression logistique on a $K=2$ classes :

    $$p_1 = \frac{e^{s_1}}{e^{s_1}+e^{s_2}}= \frac{1}{1+e^{s_2-s_1}} = \frac{1}{1+e^{-(s_1-s_2)}}$$

    Et puisque $s_1-s_2 = \mathbf x^\top (\theta^{(1)}-\theta^{(2)})$, on retrouve exactement une régression logistique en posant $\theta=\theta^{(1)}-\theta^{(2)}$ .

    ///

    ### Cross-entropy

    La loi probabiliste du modèle ayant changé (par rapport à la régression logistique), il est fort probable que l'estimateur du maximum de vraisemblance conduise à définir une fonction de coût différente. C'est ce que l'on va essayer de démontrer.

    On rappelle la définition de l'EMV, et son lien avec la fonction de coût.

    $$\begin{aligned} \hat{\boldsymbol{\theta}}_{\text{EMV}} &= \arg\max_{\boldsymbol{\theta}} \ \ln \mathcal{L}(\boldsymbol{\theta} ; \mathbf{y}) \\ &= \arg\min_{\boldsymbol{\theta}} \ \underbrace{-\ln \mathcal{L}(\boldsymbol{\theta} ; \mathbf{y})}_{J(\boldsymbol{\theta})} \end{aligned}$$

    Pour calculer la vraisemblance, on doit chercher **quelle loi des données observées dépend du paramètre** $\boldsymbol{\Theta}$.

    Dans la démonstration précédente, on avait noté « $Y^{(i)} \in \{0,1\}$ la variable aléatoire associée à son étiquette, et $y^{(i)}$ sa réalisation ». Puisque désormais on a $K$ classes possibles, on a deux possibilités :
    1. On fait évoluer $Y^{(i)}$ dans $\{0,1, .., K\}$
    2. On fait évoluer $\mathbf{Y}^{(i)}$ dans  $\{\mathbf e_1,\dots,\mathbf e_K\}$, base canonique de $\mathbb{R}^K$

    On choisis la seconde option :

    $$ \mathbf{Y} = \begin{pmatrix} Y_1 \\ \vdots \\ Y_K \end{pmatrix} $$

    $$ \forall k \in \llbracket 1, K \rrbracket, \qquad \mathbb{P}(Y_k = 1 \mid \mathbf{X} = \mathbf{x}) = \hat{p}_k $$

    > Toutes les probabilités qui suivent sont implicitement conditionnées par $\{\mathbf{X} = \mathbf{x}\}$. Dans un souci de lisibilité, on s'affranchira désormais de la notation conditionnelle et l'on écrira simplement $\mathbb{P}(\cdot)$ pour $\mathbb{P}(\cdot \mid \mathbf{X} = \mathbf{x})$.

    Les classes étant mutuellement exclusives :

    $$ \mathbb{P}(Y_k = 0) = \mathbb{P}\left( \bigcup_{\substack{j=1 \\ j \neq k}}^{K} \{Y_j = 1\} \right)= \sum_{\substack{j=1 \\ j \neq k}}^{K} \mathbb{P}(Y_j = 1) = \sum_{\substack{j=1 \\ j \neq k}}^{K} \hat{p}_j = \left( \sum_{j=1}^{K} \hat{p}_j \right) - \hat{p}_k = 1 - \hat{p}_k $$

    Par conséquent :

    $$Y_k \sim \mathcal{B}(\hat{p}_k)$$

    On trouve la loi de probabilité de $\mathbf{Y}$ grâce à celles de ses composantes $Y_k$.

    $$ \forall i \in \llbracket 1, K \rrbracket, \qquad \mathbb{P}(\mathbf{Y} = \mathbf{e}_i) = \mathbb{P}\left( \{Y_i = 1\} \cap \left( \bigcap_{\substack{k=1 \\ k \neq i}}^{K} \{Y_k = 0\} \right) \right) = \mathbb{P}\left(Y_i = 1\right) = \hat{p}_i $$

    En effet, par mutuelle exclusivité des classes :

    $$ \{\mathbf{Y} = \mathbf{e}_i\} = \{Y_i = 1\} \quad \text{car} \quad \{Y_i = 1\} \subseteq \bigcap_{k \neq i} \{Y_k = 0\}. $$

    On a entièrement définir la loi de $\mathbf{Y}$. Il ne reste plus qu'à écrire _astucieusement_ le terme $\mathbb{P}_{\boldsymbol{\Theta}}\left(\mathbf{Y}^{(i)} = \mathbf{y}^{(i)}\right)$ dans l'expression de la vraisemblance :


    $$ \; \forall \mathbf{y} \in \{\mathbf e_1,\dots,\mathbf e_K\}, \qquad \mathbb{P}(\mathbf{Y} = \mathbf{y}) = \prod_{k=1}^{K} \left(\hat{p}_k\right)^{y_k} \; $$

    La suite de la démonstration est connue. En supposant les instances du training set indépendantes, les vecteurs aléatoires $\mathbf{Y}^{(i)}$ sont conditionnellement indépendants sachant les vecteurs d'attributs $\mathbf{x}^{(i)}$ observés :

    $$ \begin{aligned} \mathcal{L}\left(\boldsymbol{\Theta} ; \mathbf{y}^{(1)},\ldots,\mathbf{y}^{(m)}\right) &= \prod_{i=1}^{m} \mathbb{P}_{\boldsymbol{\Theta}}\left(\mathbf{Y}^{(i)} = \mathbf{y}^{(i)}\right) \\[2mm] &= \prod_{i=1}^{m} \prod_{k=1}^{K} \left(\hat{p}_{k}^{(i)}\right)^{y_k^{(i)}} \end{aligned} $$

    Où $\hat{p}_{k}^{(i)}$ est la probabilité que la $i$-ème instance appartienne à la classe d'indice $k$.

    La log-vraisemblance ensuite :

    $$ \begin{aligned} \ln \mathcal{L}\left(\boldsymbol{\Theta} ; \mathbf{y}^{(1)},\ldots,\mathbf{y}^{(m)}\right) &= \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \ln \hat{p}_{k}^{(i)} \end{aligned} $$

    Finalement :

    $$ \begin{aligned} \hat{\boldsymbol{\Theta}}_{\text{EMV}} &= \arg\max_{\boldsymbol{\Theta}} \ \ln \mathcal{L}\left(\boldsymbol{\Theta} ; \mathbf{y}^{(1)},\ldots,\mathbf{y}^{(m)}\right) \\[2mm] &= \arg\max_{\boldsymbol{\Theta}} \ \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \ln \hat{p}_{k}^{(i)} \\[2mm] &= \arg\min_{\boldsymbol{\Theta}} \ \underbrace{-\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \ln \hat{p}_{k}^{(i)}}_{J(\boldsymbol{\Theta})} \end{aligned} $$

    On obtient ainsi la **Categorical Cross-Entropy** :

    $$ \; J(\boldsymbol{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \ln \hat{p}_{k}^{(i)} \; $$

    ### Solution en forme fermée ou descente de gradient

    L'annulation du gradient de la Cross-Entropy **n'admet pas de solution en forme fermée** (on l'admet, on déjà assez fait de calculs comme ça !)

    La bonne nouvelle, c'est que la Cross-Entropy est **convexe**. On procède donc par descente de gradient (batch, stochastic ou mini-batch) pour trouver le vecteur $\boldsymbol{\Theta}$ qui minimise la fonction de coût.
    """)
    return


if __name__ == "__main__":
    app.run()
