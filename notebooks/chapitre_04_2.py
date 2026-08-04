import marimo

__generated_with = "0.23.5"
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
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RidgeCV, LassoCV, ElasticNetCV
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score, RepeatedKFold
    from copy import deepcopy

    def make_poly_pipe(model, degree):
        return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), model)

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
            _cv.append(-cross_val_score(_pipe, poly_X, poly_y, cv=_cv_split,
                                        scoring="neg_root_mean_squared_error").mean())
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
        _ax.set_title(f"RMSE test {_test_best:.3f} | "
                      f"$\\Delta_{{\\text{{gén}}}}$ sans régularisation {_gap0:.2f} | $\\Delta_{{\\text{{gén}}}}$ Ridge {_gap:.2f}", fontsize=11.5)
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
                model = make_poly_pipe(model, 10)
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

    Pourquoi est-ce que réduire la norme 2 des paramètres empêcherait _nécessairement_ le modèle de trop coller au données ?

    En notant $\mathbf{w}^\star$ le vrai vecteur des paramètres (sans le biais) issu de la fonction de génération, on peut montrer que :

    $$\;\mathbb{E}\big[\|\hat{\mathbf{w}}\|_2^{\,2}\big] \;=\; \|\mathbf{w}^\star\|_2^{\,2} \;+\operatorname{tr}\left(\operatorname{Cov}(\hat{\mathbf{w}})\right)  \quad ; \quad \operatorname{tr}\left(\operatorname{Cov}(\hat{\mathbf{w}})\right)>0$$

    En moyenne sur le bruit, la norme du modèle d'estimation dépasse toujours celle de la vraie fonction, et l'excès vaut exactement la **variance des paramètres estimés**. Et cette variance des paramètres intervient justement dans l'expression de la variance des prédictions, celle-là même de la décomposition biais-variance de la première partie. Et on sait déjà que variance élevée $\Longrightarrow$ overfitting.

    > Ce résultat est vrai en espérance sur le tirage du bruit, ce n'est donc pas une régle absolue. On pourrait trouver une distribution bruitée pour laquelle un modèle qui surapprend a une norme inférieure à la fonction de génération.

    ### Standardisation

    Cette régression étant particulièrement sensible aux poids du modèle, il devient d'autant plus important de standardiser au préalable les features (via `StandardScaler` par exemple) de sorte à ce que les poids des modèles soient seulement proportionnels à leur importance, et non plus à leur ordre de grandeur.

    C'est d'ailleurs vrai pour **tous les modèles de régularisation linéaires** : Ridge, Lasso et Elastic-Net.

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
    Dans ce cas particulier, les estimations coïncident presque parfaitement. On a bon espoir que dans le cas général, la prédiction de Scikit soit **meilleure** et **plus rapide**.

    On peut aussi utiliser la **descente de gradient stochastique** pour calculer $\hat{\boldsymbol{\theta}}$ ; les avantages / inconvénients sont ceux qu'on a abordé dans la première partie. L'implémentation se fait de nouveau via `SGDRegressor` :
    """)
    return


@app.cell
def _(SGDRegressor, X, m, y):
    sgd_reg = SGDRegressor(penalty="l2", alpha=2*0.1 / m, tol=None, max_iter=1000, eta0=0.01, random_state=42)
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
            gradients = (core * 2 / len(X) * X.T @ (X @ theta - y)
                         + l1 * np.sign(theta) + l2 * theta)
            theta = theta - eta * gradients
            path.append(theta)
        return np.array(path)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 7))

    for i, N, l1, l2, title in ((0, N1, 2.0, 0, "Lasso"), (1, N2, 0, 2.0, "Ridge")):
        JR = J + l1 * N1 + l2 * 0.5 * N2 ** 2

        tr_min_idx = np.unravel_index(JR.argmin(), JR.shape)
        t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

        levels = np.exp(np.linspace(0, 1, 20)) - 1
        levelsJ = levels * (J.max() - J.min()) + J.min()
        levelsJR = levels * (JR.max() - JR.min()) + JR.min()
        levelsN = np.linspace(0, N.max(), 10)

        path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
        path_JR = bgd_path(t_init, Xr, yr, l1, l2)
        path_N = bgd_path(theta=np.array([[2.0], [0.5]]), X=Xr, y=yr,
                          l1=np.sign(l1) / 3, l2=np.sign(l2), core=0)
        ax = axes[i, 0]
        ax.grid()
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.contourf(t1, t2, N / 2.0, levels=levelsN)
        ax.plot(path_N[:, 0], path_N[:, 1], "y--")
        ax.plot(0, 0, "ys")
        ax.plot(t1_min, t2_min, "ys")
        ax.set_title(fr"$\ell_{i + 1}$ penalty")
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
    courbe_regularisation(lambda _a: Lasso(alpha=_a, max_iter=300_000), np.logspace(-3, 0.5, 60), orange, "Lasso")
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
    sgd_lasso_reg = SGDRegressor(penalty="l1", alpha=0.1 / m, tol=None, max_iter=1000, eta0=0.01, random_state=42)
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
    make_poly_pipe,
    np,
    rmse,
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

    def fit_poly(model, degree=poly_max_degree):
        return make_poly_pipe(model, degree).fit(poly_X, poly_y)

    poly_model_ols = fit_poly(LinearRegression())
    poly_coefs_ols = poly_model_ols[-1].coef_
    poly_rmse_ols = (rmse(poly_model_ols, poly_X, poly_y), rmse(poly_model_ols, poly_Xte, poly_yte))
    poly_rmse_oracle = rmse(fit_poly(LinearRegression(), poly_ref_degree), poly_Xte, poly_yte)

    _sigma = PolynomialFeatures(poly_max_degree, include_bias=False).fit_transform(poly_X).std(axis=0)
    poly_true_std = np.array([poly_true_coefs[1], poly_true_coefs[2], poly_true_coefs[3],
                              0.0, 0.0, 0.0, 0.0]) * _sigma

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
        (montre_vrai.value,  None,            "#1a202c", "vrai polynôme (deg 3)", "--"),
        (montre_ols.value,   poly_model_ols,  rouge,    "sans pénalité",          "-"),
        (montre_ridge.value, poly_ridge,      vert,      f"Ridge ($\\alpha_1$={aR:g})", "-"),
        (montre_lasso.value, poly_lasso,      orange,    f"Lasso ($\\alpha_2$={aL:g})", "-"),
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
        (f"Ridge ($\\alpha_1$={aR:g})", poly_ridge, vert,   poly_wmax_reg),
        (f"Lasso ($\\alpha_2$={aL:g})", poly_lasso, orange, poly_wmax_reg),
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
        _ax2.set_title(f"{_nom}\nRMSE test {rmse(_m, poly_Xte, poly_yte):.2f}", fontsize=10.5, pad=8)
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
    courbe_regularisation(lambda _a: ElasticNet(alpha=_a, l1_ratio=0.5, max_iter=300_000),
                          np.logspace(-3, 0.5, 60), violet, "Elastic-Net")
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
    elasticnetcv_reg = ElasticNetCV(l1_ratio=np.linspace(0.1, 0.9, 9), alphas=np.logspace(-3, 3, 100))
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

    ## D. Early stopping

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
    plt.annotate("Meilleur modèle",
                 xy=(_best_epoch, _best_valid_rmse),
                 xytext=(_best_epoch, _best_valid_rmse + 0.5),
                 ha="center",
                 arrowprops=dict(facecolor="black", shrink=0.05))
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
    """)
    return


@app.cell(hide_code=True)
def _(np):
    _rng = np.random.default_rng(seed=42)
    _m = 200
    _X = 6 * _rng.random((_m, 1)) - 3
    _y = 0.5 * _X ** 2 + _X + 2 + _rng.standard_normal((_m, 1))
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
    _preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False), StandardScaler())
    _X_train_prep = _preprocessing.fit_transform(X_train)
    _X_valid_prep = _preprocessing.transform(X_valid)
    _sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
    _n_epochs = 500
    _best_valid_rmse = float("inf")

    train_errors, val_errors = [], []
    best_model = deepcopy(_sgd_reg)

    for _epoch in range(_n_epochs):
        _sgd_reg.partial_fit(_X_train_prep, y_train)
        _val_error = rmse(_sgd_reg, _X_valid_prep, y_valid)
        if _val_error < _best_valid_rmse:
            _best_valid_rmse = _val_error
            best_model = deepcopy(_sgd_reg)
        _train_error = rmse(_sgd_reg, _X_train_prep, y_train)
        val_errors.append(_val_error)
        train_errors.append(_train_error)
    return train_errors, val_errors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Le lien avec Ridge

    [insérer éventuellement une section mathématique ici]
    """)
    return


if __name__ == "__main__":
    app.run()
