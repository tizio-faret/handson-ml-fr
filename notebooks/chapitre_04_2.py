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
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score, RepeatedKFold

    rouge = "#c53030"
    orange = "#dd6b20"
    gris = "#718096"
    bleu = "#2b6cb0"
    vert = "#2f855a"
    violet = "#6b46c1"
    jaune = "#fcbf49"
    return (
        ElasticNet,
        Lasso,
        LinearRegression,
        PolynomialFeatures,
        RepeatedKFold,
        Ridge,
        RidgeCV,
        SGDRegressor,
        StandardScaler,
        bleu,
        cross_val_score,
        gris,
        make_pipeline,
        mean_squared_error,
        mo,
        np,
        orange,
        plt,
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

    ### Visualisations

    Dans le graphique ci-dessous, on balaie $\alpha$ et on trace les deux RMSE : l'**écart entre l'erreur d'entraînement et de test** est caractéristique de l'overfitting, comme vu en première partie, c'est donc lui qu'on cherche à refermer. Le $\alpha$ retenu (pointillés) minimise l'erreur de **validation croisée sur le training set** ; la MSE est ensuite calculée sur le test set puis reportée sur le titre du graphique, de sorte à obtenir un chiffre honnête et comparable entre Ridge, Lasso et Elastic-Net.
    """)
    return


@app.cell(hide_code=True)
def _(
    PolynomialFeatures,
    RepeatedKFold,
    StandardScaler,
    bleu,
    cross_val_score,
    gris,
    make_pipeline,
    mean_squared_error,
    np,
    plt,
    poly_X,
    poly_Xte,
    poly_max_degree,
    poly_rmse_oracle,
    poly_y,
    poly_yte,
    rouge,
):
    def courbe_regularisation(make_model, alphas, couleur, nom):
        _cv_split = RepeatedKFold(n_splits=5, n_repeats=30, random_state=42) 
        _train, _test, _cv = [], [], []
        for _a in alphas:
            _pipe = make_pipeline(PolynomialFeatures(poly_max_degree, include_bias=False),
                                  StandardScaler(), make_model(_a))
            _cv.append(-cross_val_score(_pipe, poly_X, poly_y, cv=_cv_split,
                                        scoring="neg_root_mean_squared_error").mean())
            _pipe.fit(poly_X, poly_y)
            _train.append(np.sqrt(mean_squared_error(poly_y,   _pipe.predict(poly_X))))
            _test.append( np.sqrt(mean_squared_error(poly_yte, _pipe.predict(poly_Xte))))
        _train, _test = np.array(_train), np.array(_test)
        _ibest = int(np.argmin(_cv))        
        _a_best, _test_best = alphas[_ibest], _test[_ibest]

        _fig, _ax = plt.subplots(figsize=(8.5, 4.6))
        _ax.fill_between(alphas, _train, _test, color=couleur, alpha=0.12,
                         label="écart train/test = overfitting")
        _ax.plot(alphas, _train, "-", color=rouge, lw=2.2, label="RMSE entraînement")
        _ax.plot(alphas, _test,  "-", color=bleu,  lw=2.4, label="RMSE test")
        _ax.axhline(poly_rmse_oracle, color=gris, ls="--", lw=1.3, label="oracle (vrai degré 3)")
        _ax.axvline(_a_best, color=couleur, lw=1.4, ls=":")
        _ax.scatter([_a_best], [_test_best], s=75, color=couleur, ec="white", lw=0.8,
                    zorder=6, label=fr"$\alpha$ choisi par CV = {_a_best:.3g}")
        _ax.set_xscale("log")
        _ax.set_xlabel(r"$\alpha$ (échelle log)"); _ax.set_ylabel("RMSE")
        _ax.set_title(f"{nom} | RMSE test {_test_best:.3f}", fontsize=11.5)
        _ax.legend(fontsize=8, loc="upper center", framealpha=0.92)
        for _s in ("top", "right"): _ax.spines[_s].set_visible(False)
        _fig.tight_layout(); plt.close(_fig)
        return _fig

    return (courbe_regularisation,)


@app.cell(hide_code=True)
def _(Ridge, courbe_regularisation, np, vert):
    courbe_regularisation(lambda _a: Ridge(alpha=_a), np.logspace(-3, 3, 60), vert, "Ridge")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plus $\alpha$ est proche de 0, plus la pénalisation est faible et plus le modèle tend à se comporter comme une régression classique.

    Visuellement, sur une régression linéaire, augmenter $\alpha$ conduit à des prédictions plus lisses, ce qui réduit la variance du modèle mais augmente son biais.
    """)
    return


@app.cell(hide_code=True)
def _(
    LinearRegression,
    PolynomialFeatures,
    Ridge,
    StandardScaler,
    bleu,
    make_pipeline,
    np,
    plt,
    rouge,
    vert,
):
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
        plt.plot(X_regularization_demo, y_regularization_demo,
                 marker=".", linestyle="none", color=bleu, linewidth=3)
        line_colors = (bleu, vert, rouge)
        line_styles = (":", "--", "-")
        for alpha, color, style in zip(alphas, line_colors, line_styles):
            if alpha > 0:
                model = model_class(alpha, **model_kwargs)
            else:
                model = LinearRegression()
            if polynomial:
                model = make_pipeline(
                    PolynomialFeatures(degree=10, include_bias=False),
                    StandardScaler(),
                    model)
            model.fit(X_regularization_demo, y_regularization_demo)
            y_new_regul = model.predict(X_new_regularization_demo)
            plt.plot(X_new_regularization_demo, y_new_regul,
                     linestyle=style, color=color, linewidth=2,
                     label=fr"$\alpha = {alpha}$")
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

    Pourquoi est-ce que réduire la norme 2 des paramètres empêcherait _nécessairement_ le modèle de trop coller au données ? Ne pourrait-on pas trouver une distribution bruitée pour laquelle un modèle qui surapprend a une norme inférieure à la fonction de génération ?

    En effet, un tel contre-exemple existe bel et bien. La justification est que l'implication

    $$ \text{overfitting} \Longrightarrow {{\| \mathbf{w}\|}_2} \text{ élevée} $$

    est vraie **en espérance** sur le tirage du bruit. En notant $\mathbf{w}^\star$ le vrai vecteur des paramètres (sans le biais) issu de la fonction de génération, on peut montrer que :

    $$\;\mathbb{E}\big[\|\hat{\mathbf{w}}\|_2^{\,2}\big] \;=\; \|\mathbf{w}^\star\|_2^{\,2} \;+\operatorname{tr}\left(\operatorname{Cov}(\hat{\mathbf{w}})\right)  \quad ; \quad \operatorname{tr}\left(\operatorname{Cov}(\hat{\mathbf{w}})\right)>0$$

    En moyenne sur le bruit, la norme du modèle d'estimation dépasse toujours celle de la vraie fonction, et l'excès vaut exactement la **variance des paramètres estimés**. Et cette variance des paramètres, elle intervient justement dans l'expression de la variance des prédictions, celle-là même de la décomposition biais-variance de la première partie. Et on sait déjà que variance élevée $\Longrightarrow$ overfitting.

    En résumé, l'intuition à garder est que « **réduire la norme pénalise les configurations où le modèle surapprend** » et non « tout modèle qui surapprend a une grande norme ». C'est aussi pourquoi $\alpha$ reste un hyperparamètre : le bon dosage dépend de la vraie fonction, inconnue, donc on le choisit par validation croisée.


    ### Standardisation

    Cette régression étant particulièrement sensible aux poids du modèle, il devient d'autant plus important de standardiser au préalable les features (via `StandardScaler` par exemple) de sorte à ce que les poids des modèles soient seulement proportionnels à leur importance, et non plus à leur ordre de grandeur.

    ### Implémentation

    On peut implémenter la régression ridge avec une solution en** forme close** :

    $$ \hat{\boldsymbol{\theta}} = \left( \mathbf{X}^{\mathsf T}\mathbf{X} + \alpha\mathbf{D} \right)^{-1} \mathbf{X}^{\mathsf T}\mathbf{y}$$

    Où $\mathbf{D}$ est la matrice identité avec un 0 en haut à gauche.

    /// details | Solution en forme close - démonstration (pour aller plus loin)
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
    A = np.array([[0., 0.], [0., 1.]])
    X_b = np.c_[np.ones(m), X]
    theta = np.linalg.inv(X_b.T @ X_b + alpha * A) @ X_b.T @ y

    print("--- Calcul matriciel brut ---")
    print(f"theta_0 : {theta[0,0]:.20f}")
    print(f"theta_1 : {theta[1,0]:.20f}\n")
    print("--- Estimation Scikit ---")
    print(f"theta_0 : {ridge_reg.intercept_[0]:.20f}")
    print(f"theta_1 : {ridge_reg.coef_[0]:.20f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dans ce cas particulier, les estimations coincident presque parfaitement. On a bon espoir que dans le cas général, la prédiction de Scikit soit **meilleure** et **plus rapide**.

    On peut aussi utiliser la **descente de gradient stochastique** pour calculer $\hat{\boldsymbol{\theta}}$ ; les avantages / inconvénients sont ceux qu'on a abordé dans la première partie. L'implémentation se fait de nouveau via `SGDRegressor` :
    """)
    return


@app.cell
def _(SGDRegressor, X, m, y):
    sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None, max_iter=1000, eta0=0.01, random_state=42)
    _ = sgd_reg.fit(X, y.ravel()) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Petite subtilité ici ! En précisant `penalty="l2"`, on ajoute $\alpha {{\| \mathbf{w}\|}_2}^{2}$ à la fonction de perte. Puisqu'on veut plutôt ajouter $\displaystyle \frac{\alpha}{m} {{\| \mathbf{w}\|}_2}^{2}$, on doit imposer `alpha=0.1 / m` en divisant bien par $m$.

    ### RidgeCV

    On présente une dernière optimisation pour notre régression Ridge. Elle est très semblable au [`Ridge`](#implementation-ridge-scikit) que l'on vient de voir, mais intègre une optimisation de l'hyper-paramètre $\alpha$ par validation croisée.

    Pourquoi ne pas se contenter du classique `Ridge` + `GridSearchCV` ?

    Très bonne question, et je vous remercie de l'avoir posée. En fait, `RidgeCV` fait exactement la même chose mais de façon **optimisée pour la régression Ridge** et s'exécute donc beaucoup **plus rapidement**.

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

    Lasso est
    """)
    return


@app.cell(hide_code=True)
def _(Lasso, courbe_regularisation, np, orange):
    courbe_regularisation(lambda _a: Lasso(alpha=_a, max_iter=300_000), np.logspace(-3, 0.5, 60), orange, "Lasso")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## C. Régularisation des modèles polynomiaux

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
        value=0.03, show_value=True, label=r"$\alpha_1$ (Ridge)",
    )
    alpha_lasso_slider = mo.ui.slider(
        steps=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
        value=0.02, show_value=True, label=r"$\alpha_2$ (Lasso)",
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
    StandardScaler,
    make_pipeline,
    mean_squared_error,
    np,
):
    poly_ref_degree = 3
    poly_max_degree = 7
    poly_n_train = 20
    poly_noise = 1.2
    poly_lo, poly_hi = -2.5, 2.5
    poly_true_coefs = np.array([0.5, -1.6, -1.1, 1.0])   # 0.5 -1.6x -1.1x^2 +1.0x^3

    def f(x):
        return sum(poly_true_coefs[_k] * x**_k for _k in range(len(poly_true_coefs)))

    _rng = np.random.default_rng(0)
    _xtr = _rng.uniform(poly_lo, poly_hi, poly_n_train)
    poly_X = _xtr.reshape(-1, 1)
    poly_y = f(_xtr) + _rng.normal(0.0, poly_noise, poly_n_train)
    _xte = _rng.uniform(poly_lo, poly_hi, 2000)
    poly_Xte = _xte.reshape(-1, 1)
    poly_yte = f(_xte) + _rng.normal(0.0, poly_noise, 2000)

    def _pipe(_deg, _model):
        return make_pipeline(PolynomialFeatures(_deg, include_bias=False),
                             StandardScaler(), _model).fit(poly_X, poly_y)
    def _rmse(_m, _X, _y):
        return np.sqrt(mean_squared_error(_y, _m.predict(_X)))

    poly_model_ols = _pipe(poly_max_degree, LinearRegression())
    poly_coefs_ols = poly_model_ols[-1].coef_
    poly_rmse_ols = (_rmse(poly_model_ols, poly_X, poly_y), _rmse(poly_model_ols, poly_Xte, poly_yte))
    poly_rmse_oracle = _rmse(_pipe(poly_ref_degree, LinearRegression()), poly_Xte, poly_yte)

    _sigma = PolynomialFeatures(poly_max_degree, include_bias=False).fit_transform(poly_X).std(axis=0)
    poly_true_std = np.array([poly_true_coefs[1], poly_true_coefs[2], poly_true_coefs[3],
                              0.0, 0.0, 0.0, 0.0]) * _sigma

    poly_wmax_ols = 1.1 * np.abs(poly_coefs_ols).max()
    poly_wmax_reg = 1.15 * max(
        np.abs(_pipe(poly_max_degree, Ridge(alpha=0.01))[-1].coef_).max(),
        np.abs(_pipe(poly_max_degree, Lasso(alpha=0.01, max_iter=300_000))[-1].coef_).max(),
        np.abs(poly_true_std).max(),
    )
    return (
        f,
        poly_X,
        poly_Xte,
        poly_hi,
        poly_lo,
        poly_max_degree,
        poly_model_ols,
        poly_n_train,
        poly_rmse_oracle,
        poly_true_std,
        poly_wmax_ols,
        poly_wmax_reg,
        poly_y,
        poly_yte,
    )


@app.cell(hide_code=True)
def _(
    Lasso,
    PolynomialFeatures,
    Ridge,
    StandardScaler,
    alpha_lasso_slider,
    alpha_ridge_slider,
    f,
    gris,
    make_pipeline,
    mean_squared_error,
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
    poly_lo,
    poly_max_degree,
    poly_model_ols,
    poly_n_train,
    poly_y,
    rouge,
    vert,
):
    _aR = alpha_ridge_slider.value
    _aL = alpha_lasso_slider.value

    def _pipe(_deg, _model):
        return make_pipeline(PolynomialFeatures(_deg, include_bias=False),
                             StandardScaler(), _model).fit(poly_X, poly_y)
    def _rmse(_m, _X, _y):
        return np.sqrt(mean_squared_error(_y, _m.predict(_X)))

    _ridge = _pipe(poly_max_degree, Ridge(alpha=_aR))
    _lasso = _pipe(poly_max_degree, Lasso(alpha=_aL, max_iter=300_000))

    _courbes = [
        (montre_vrai.value,  None,            "#1a202c", "vrai polynôme (deg 3)", "--"),
        (montre_ols.value,   poly_model_ols,  rouge,    "sans pénalité",         "-"),
        (montre_ridge.value, _ridge,          vert,      f"Ridge ($\\alpha_1$={_aR:g})", "-"),
        (montre_lasso.value, _lasso,          orange,    f"Lasso ($\\alpha_2$={_aL:g})", "-"),
    ]

    _fig1, _ax = plt.subplots(figsize=(8.5, 4.6))
    _gx = np.linspace(poly_lo - 0.7, poly_hi + 0.7, 500).reshape(-1, 1)
    _ax.axvspan(poly_lo, poly_hi, color=gris, alpha=0.07, zorder=0)
    _ax.text(0.5, 0.03, "zone grisée = entraînement · au-delà = extrapolation",
             transform=_ax.transAxes, ha="center", va="bottom", fontsize=8, color=gris)
    for _visible, _m, _col, _lab, _ls in _courbes:
        if not _visible:
            continue
        if _m is None:
            _ax.plot(_gx.ravel(), f(_gx.ravel()), _ls, color=_col, lw=1.6, label=_lab, zorder=2)
        else:
            _ax.plot(_gx, _m.predict(_gx), _ls, color=_col, lw=2.2,
                     label=f"{_lab}", zorder=3)
    _ax.scatter(poly_X.ravel(), poly_y, s=26, color="#2d3748", zorder=4, label=f"{poly_n_train} points")
    _ax.set_ylim(poly_y.min() - 10, poly_y.max() + 10)
    _ax.set_xlim(poly_lo - 0.7, poly_hi + 0.7)
    _ax.set_xlabel("x"); _ax.set_ylabel("y")

    _ax.legend(fontsize=8, loc="lower right")
    for _s in ("top", "right"): _ax.spines[_s].set_visible(False)
    _fig1.tight_layout(); plt.close(_fig1)

    mo.vstack([
        mo.hstack([alpha_ridge_slider, alpha_lasso_slider], justify="center", gap=2),
        mo.hstack([montre_vrai, montre_ols, montre_ridge, montre_lasso], justify="center", gap=1),
        _fig1,
    ], gap=0.8)
    return


@app.cell(hide_code=True)
def _(
    Lasso,
    PolynomialFeatures,
    Ridge,
    StandardScaler,
    alpha_lasso_slider,
    alpha_ridge_slider,
    make_pipeline,
    mean_squared_error,
    np,
    orange,
    plt,
    poly_X,
    poly_Xte,
    poly_max_degree,
    poly_model_ols,
    poly_true_std,
    poly_wmax_ols,
    poly_wmax_reg,
    poly_y,
    poly_yte,
    vert,
):
    _rouge = "#c53030"
    _aR = alpha_ridge_slider.value
    _aL = alpha_lasso_slider.value

    def _pipe(_deg, _model):
        return make_pipeline(PolynomialFeatures(_deg, include_bias=False),
                             StandardScaler(), _model).fit(poly_X, poly_y)
    def _rmse(_m, _X, _y):
        return np.sqrt(mean_squared_error(_y, _m.predict(_X)))

    _ridge = _pipe(poly_max_degree, Ridge(alpha=_aR))
    _lasso = _pipe(poly_max_degree, Lasso(alpha=_aL, max_iter=300_000))
    _panneaux = [
        ("Régression polynomiale classique", poly_model_ols, _rouge, poly_wmax_ols),
        (f"Ridge ($\\alpha_1$={_aR:g})", _ridge, vert, poly_wmax_reg),
        (f"Lasso ($\\alpha_2$={_aL:g})", _lasso, orange, poly_wmax_reg),
    ]

    _degres = np.arange(1, poly_max_degree + 1)
    _fig2, _axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for _ax2, (_nom, _m, _col, _wm) in zip(_axes, _panneaux):
        _c = _m[-1].coef_
        _ax2.axhline(0, color="#cbd5e0", lw=1, zorder=1)
        _ax2.bar(_degres, _c, width=0.62, color=_col, zorder=3, edgecolor="white", linewidth=0.6)
        for _d, _tv in zip(_degres, poly_true_std):
            _ax2.plot([_d - 0.34, _d + 0.34], [_tv, _tv], color="black", lw=2.4, zorder=6,
                      solid_capstyle="round")
        _ax2.set_title(f"{_nom}\nRMSE test {_rmse(_m, poly_Xte, poly_yte):.2f}", fontsize=10.5, pad=8)
        _ax2.set_xticks(_degres); _ax2.set_xlabel("degré du terme", fontsize=10)
        _ax2.set_xlim(0.4, poly_max_degree + 0.6); _ax2.set_ylim(-_wm, _wm)
        _ax2.grid(axis="y", color="#edf2f7", lw=0.8, zorder=0)
        for _s in ("top", "right"): _ax2.spines[_s].set_visible(False)
    _axes[0].set_ylabel("poids signé (base standardisée)", fontsize=10)
    _axes[0].plot([], [], color="black", lw=2.4, label="vrai poids")
    _axes[0].legend(fontsize=8, loc="upper left")
    _fig2.tight_layout(); plt.close(_fig2)

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

    C'est une des raisons pour lesquelles on préfère souvent Ridge pour les bases polynomiales.

    ### Faut-il cesser de plafonner le choix du degré ?

    Non. Même lorsqu'on utilise Ridge ou Lasso, on limite généralement le degré maximal pour des raisons qu'on connaît bien :

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
    """)
    return


@app.cell(hide_code=True)
def _(ElasticNet, courbe_regularisation, np, violet):
    courbe_regularisation(lambda _a: ElasticNet(alpha=_a, l1_ratio=0.5, max_iter=300_000),
                          np.logspace(-3, 0.5, 60), violet, "Elastic-Net")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## E. Apprentissage ensembliste
    """)
    return


if __name__ == "__main__":
    app.run()
