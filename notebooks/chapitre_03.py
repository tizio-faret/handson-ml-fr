import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Chapitre 3 - Classification**
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ce notebook est inspiré du chapitre 3 de Hands-On Machine Learning (3e éd.) d'Aurélien Géron. Le code est adapté de [ageron/handson-ml3](https://github.com/ageron/handson-ml3), sous licence Apache 2.0.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # I. Le dataset **MNIST**

    ## A. Récupération et découverte des données
    Ce dataset agrège un ensemble de 70 000 images (format 28x28) de chiffres manucrits. Chaque image est encodée en niveau de gris et étiquetée avec le numéro du chiffre qu'elle représente. C'est un dataset très classique en ML, popularisé par Yann LeCun (🇫🇷 !).

    Scikit-Learn via `sklearn.datasets` permet d'importer facilement de tels datasets. Ici, on va utiliser la fonction `fetch_openml` qui sollicite OpenML, une plateforme open source qui fournit différentes ressources pour le machine learning.
    > On peut de la même façon importer Fashion-MNIST, CIFAR, breast-cancer, titanic, etc..

    Les dataset importés via `fetch_openml` possèdent généralement trois attributs :
    1. `.DESCR` : une description du dataset.
    2. `.data` : les prédicteurs (tableau NumPy 2D)
    3. `.target` : les labels (tableau NumPy 1D)

    Avec 70 000 images de 28 par 28 pixels, on s'attend à obtenir un tableau de taille (70000, 28*28=784).
    """)
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    # 28*28 = 784
    mnist = fetch_openml('mnist_784', as_frame=False)

    mnist.data.shape
    return (mnist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut visualiser les images en niveau de gris en utilisant une _colormap_ : option `cmap="binary"` de `plt.imshow()`.

    On décide de consulter la première instance (l'instance d'indice 0) du dataset.
    """)
    return


@app.cell
def _(mnist):
    import matplotlib.pyplot as plt

    def plot_digit(image_data, x_dim, y_dim):
        # On convertit le vecteur d'entrée en tableau 2D 28x28
        image = image_data.reshape(28, 28)
        plt.figure(figsize=(x_dim, y_dim))
        # Colormap
        plt.imshow(image, cmap="binary")
        plt.axis("off")

    some_digit = mnist.data[0]
    plot_digit(some_digit, 2, 2)
    plt.show()

    print(f"Label de l'instance : {mnist.target[0]}")
    return plt, some_digit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ça ressemble à un 5, ce que confirme le label.

    ---
    ## B. Préparation des données

    Avant de s'aventurer plus loin, on décide de fixer le set d'entraînement et le set de test. Par défaut dans ce dataset, les 60 000 premières instances sont dédiées à l'entraînement et les 10 000 restantes au test.
    """)
    return


@app.cell
def _(mnist):
    X_train, X_test, y_train, y_test = mnist.data[:60000], mnist.data[60000:], mnist.target[:60000], mnist.target[60000:]
    return X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On trace un histogramme pour s'assurer que cette répartition suit un échantillonage stratifié.

    > Avec l'option `density=True`, on affiche les **proportions** des occurences (plutôt que les effectifs bruts).
    """)
    return


@app.cell
def _(plt, y_test, y_train):
    import numpy as np

    plt.figure(figsize=(6, 3)) 
    plt.hist([y_train, y_test], color = ['steelblue', 'gold'], label = ['train set', 'test set'], 
             histtype = 'bar', density=True) 
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Les proportions ont l'air semblables d'un set à l'autre. On laisse donc les variables `X_train, X_test, y_train, y_test` telles quelles.

    # II. Entraînement d'un classifieur binaire

    Un modèle de **classification binaire** est un d'algorithme de ML qui prédit si une entrée appartient ou non à une classe. Pour résoudre notre tâche de classification, on va d'abord s'attacher à entraîner et à évaluer un tel modèle.

    On décide d'implémenter un premier modèle qui détermine si une image est un cinq ou non. La feature cible de chaque instance sera donc '5' (prédiction positive) ou 'non 5' (prédiction négative).
    """)
    return


@app.cell
def _(y_test, y_train):
    y_train_5 = (y_train == '5')
    y_test_5 = (y_test == '5')
    return (y_train_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. SGDClassifier

    Le premier classifieur binaire auquel on s'intéresse est `SGDClassifier` (Stochastic Gradient Descent Classifier). On va s'abord essayer de comprendre son fonctionnement dans les grandes lignes avant de l'implémenter.

    ---

    ### Principe de fonctionnement

    Un modèle de classification binaire typique sur des datasets de grande taille repose sur une fonction de décision, dont les poids sont fixés par une descente de gradient stochastique. Le principe est simple, et c'est celui d'un réseau de neurones à une seule couche.
    1. On attribue à chacune des 784 features $f_j$ un poids $w_j$, ce qui nous donne un vecteur de poids $\mathbf{w}$.
    2. On définit avec ces poids une fonction de décision, qui attribue un score à chaque instance :
    $$\text{decision}\_\text{function}(\mathbf{x}^{(i)}) = \mathbf{w}^\top \mathbf{x}^{(i)}+b=\sum_{j=1}^{784} w_j * \mathbf{x_j}^{(i)}+ b$$
    3. On définit un seuil (threshold), et une nouvelle instance $\mathbf{x}^{(i)}$ appartient à la classe étudiée si et seulement si
    $$\text{decision}\_\text{function}(\mathbf{x}^{(i)})\geq\text{threshold}$$

    L'ajustement des poids $w_j$ se fait par descente de gradient et constitue l'étape d'entraînement du modèle.

    > Par défaut et pendant l'entraînement le threshold vaut 0 ; on peut choisir de l'augmenter pour améliorer la **précision** du modèle - ou de le baisser, pour augmenter le **recall** (voir section "precision vs recall" plus bas).

    ---

    ### La descente de gradient

    Pour mettre en place une descente de gradient, il faut avoir une **fonction de coût** à minimiser. Elle est de la forme $J(\theta)$, où $\theta$ est notre inconnue : elle regroupe les paramètres à apprendre (ici le vecteur de poids $\mathbf{w}$ et le biais $b$). Le principe de la descente est simple : on définit la suite $\theta_k$, on calcule ses termes et on s'arrête lorsqu'une condition bien choisie est vérifiée.

    1. On part d'un $\theta_0$ quelconque
    2. Puisque le gradient $\nabla J(\theta)$ donne la direction de la plus forte pente, on se déplace dans la direction opposée :

    $$\theta_{k+1} = \theta_k - t_k\,\nabla J(\theta_k)$$
    où $t_k \geq 0$ est le **pas** (en ML, on l'appelle le *taux d'apprentissage* ou *learning rate* ).

    3. On répète jusqu'à ce que le gradient soit suffisamment petit : $\|\nabla J(\theta_k)\| \leq \varepsilon$.

    ---

    ### La fonction de coût

    Pour un classifieur, le coût $J$ est une moyenne sur les instances d'entraînement d'une fonction de perte $L$ (qu'on ne détaille pas) :
    $$J(\theta) = \frac{1}{m}\sum_{i=1}^{m} L\big(\theta\,;\,\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\big)$$
    $$\nabla J(\theta) = \frac{1}{m}\sum_{i=1}^{m} \nabla L\big(\theta\,;\,\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\big).$$

    La descente de gradient classique (dite **batch**) utilise toutes les instances à chaque itération. Avec la version stochastique, on va essayer de s'émanciper de ces $k \times 60 000$ évaluations du gradient.

    ---

    ### Sélection du pas

    Trouver un bon $t_k$ revient à minimiser la fonction d'une variable $g(t) = J(\theta_k - t\,\nabla J(\theta_k))$. Plusieurs stratégies existent :

    | Stratégie | Principe | En pratique |
    |---|---|---|
    | **Pas constant** | $t_k = t$ fixé| $t$ trop petit ⇒ convergence lente ; $t$ trop grand ⇒ divergence. Sous de bonnes hypothèses, le choix optimal est $t = 1/L$, où $L$ est la constante de Lipschitz de $\nabla J$. |
    | **Recherche linéaire exacte** | $t_k = \arg\min_{t\geq 0} g(t)$ | Idéal mais rarement calculable explicitement. |
    | **Backtracking** | Partir d'un pas $t_0$ et le réduire tant que la décroissance n'est pas suffisante | Bon compromis : c'est celle qu'on utilise en pratique. |

    > **Backtracking** : une implémentation littérale requerrait de réévaluer $J$ (et donc de re-balayer les données) plusieurs fois **par pas** : c'est coûteux sur de gros datasets. En pratique on fixe plutôt un taux d'apprentissage que l'on fait décroître au fil des itérations.

    ---

    ### La descente *stochastique*

    Le défaut de la version batch, c'est donc clairement le coût des balayages du set d'entraînement à chaque itération.

    Pour y remédier, la descente de gradient stochastique qu'implémente `SGDClassifier` propose l'heuristique suivante : à chaque itération, on estime le gradient à partir d'**une seule instance tirée au hasard**.

    $$\theta_{k+1} = \theta_k - t_k\,\nabla L\big(\theta_k\,;\,\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\big), \qquad i \text{ tiré aléatoirement.}$$

    Une autre solution consiste aussi à estimer le gradient en faisant une moyenne sur un nombre réduit d'instances (mini-batch).

    > Comme chaque pas ne consulte qu'une seule instance, il faut $m$ pas (ici 60 000) pour avoir vu tout le training set une fois. On dénombre donc plus volontiers les itérations en nombre d'**époques**, c'est-à-dire en équivalent training set.

    ---

    ### Implémentation

    On implémente donc notre classifieur en l'entraînant sur tout le training set.
    """)
    return


@app.cell
def _(X_train, y_train_5):
    from sklearn.linear_model import SGDClassifier

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    return SGDClassifier, sgd_clf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # III. Mesure des performances d'un classifieur binaire

    La mesure des performances d'un modèle de classification n'est pas un problème simple. Il existe plusieurs métriques (exactitude, précision, recall, matrice de confusion, score F1, ..) qui ont toutes leur intérêt.

    ## A. Mesurer l'**accuracy** par validation croisée

    On s'intéresse à la métrique suivante :

    $$ \text{accuracy} = \frac{\text{nombre de prédictions correctes}}{\text{nombre total d'instances}} $$

    Pour évaluer l'**exactitude** (accuracy) de notre modèle, une bonne pratique consiste à procéder par validation croisée.

    > Rappel chapitre 2 : La fonction `cross_val_score` du package Scikit-Learn en est l'implémentation la plus idiomatique. Elle partitionne le train set en k subsets (ou **folds**), ce qui donne par passage au complémentaire k splits entraînement/validation. On entraîne-évalue k fois et on définit l'erreur totale comme la moyenne de ces k erreurs.
    """)
    return


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import cross_val_score
    import pandas as pd

    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On a plus de 95% d'exactitude sur chaque fold : la quasi-totalité des instances classées par notre modèle (prédictions positives et négatives confondues) étaient correctes. C'est assez prometteur, mais cette métrique devient peu informative lorsque les classes sont **fortement déséquilibrées** (ce qui est clairement le cas ici).
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(plt, y_train_5):
    plt.figure(figsize=(3, 2)) 
    plt.bar(['5', 'non 5'], [sum(y_train_5),len(y_train_5) - sum(y_train_5)], color=['steelblue', 'gold'])
    plt.show()

    print(f"Proportion de 'non 5' : {round((1 - sum(y_train_5)/len(y_train_5))*100)} %")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À retenir

    En effet, un modèle qui tendrait (même légèrement) à classer arbitrairement les images dans la classe la plus fréquente aura mécaniquement une accuracy élevée et cette métrique passera sous silence son **biais**. Si, en production, le modèle est confronté à des données où les classes sont présentes dans des proportions différentes, **l'accuracy seule ne permettra pas d'anticiper une éventuelle dégradation de ses performances**.

    Si on force le trait, un modèle de classification naïf qui attribuerait à chaque instance la classe la plus fréquente dans le set d'entraînement (i.e. la classe 'non 5') aurait ici une accuracy de l'ordre de 90%.

    ---

    ## B. Matrices de confusion

    Une matrice de confusion synthétise, pour chaque couple de classes (A,B), **le nombre de fois où une instance de classe A a été classée par le modèle comme appartenant à la classe B**. Dans notre cas, le classifieur est binaire et les deux classes sont '5' et 'non 5'. Un exemple sera plus parlant.

    NB : Pour calculer cette matrice, il faut analyser (et donc calculer) des prédictions. Pour éviter l'overfitting sans _snooper_ dans le test set, on voudrait procéder par validation croisée. C'est tout à fait possible : `cross_val_predict` fait la même chose que `cross_val_score`, mais renvoie le **vecteur des prédictions réalisées**. On a donc d'ailleurs pas besoin de préciser à cette fonction une métrique d'erreur.
    """)
    return


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    return cross_val_predict, y_train_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Une matrice de confusion se lit comme suit :
    - Chaque ligne de la matrice correspond aux **vraies** classes (première ligne 'non 5')
    - Chaque colonne de la classe correspond aux classes **prédites** (première colonne 'non 5')

    Le code qui suit est assez explicite, Scikit fournit une fonction pour calculer notre matrice de confusion.
    """)
    return


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_train_5, y_train_pred)
    cm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 53 892 images ont correctement été classées comme 'non 5' <---> $\text{(TN, True Negative)}$.
    - 3 530 images ont correctement été classées comme '5' <---> $\text{(TP, True Positive)}$.
    - 1 891 images ont été classées à tort comme 'non 5' <---> $\text{(FN, False Negative)}$.
    - 687 images ont été classées à tort comme '5' <---> $\text{(FP, False Positive)}$.

    > Un classifieur parfait aurait donc une matrice de confusion diagonale.

    ---
    ## C. D'autres métriques importantes

    ### Précision

    La lecture de la matrice de confusion est édifiante, mais on peut lui préférer une mesure plus concise.

    À ce titre, on s'intérèsse souvent à la **précision** d'un classifieur. Elle quantifie la proportion de prédictions positives correctes.
    $$ \text{précision} = \frac{\text{TP}}{\text{TP + FP}}$$

    Malheureusement cette métrique seule ne suffit pas. En effet, pour avoir une grande proportion de prédictions **positives** correctes, il suffit de ne classer positivement que les quelques instances associées à une très haute probabilité d'appartenir à la classe. De cette façon, on fait mécaniquement grimper la précision du modèle, mais **beaucoup d'instances positives passent sous le radar**.

    ### Recall

    La précision doit donc être mise en perspective avec la proportion d'instances positives qui _passent sous le radar_. En pratique on calcule le **recall**, mais c'est directement lié ; le recall quantifie la proportion d'instances positives qui sont classées comme telles par le modèle.

    $$ \text{recall} = \frac{\text{TP}}{\text{TP + FN}}$$

    À nouveau, Scikit fournit des fonctions prêtes à l'emploi pour calculer nos deux métriques.
    """)
    return


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import precision_score, recall_score

    _precision = precision_score(y_train_5, y_train_pred)
    _recall = recall_score(y_train_5, y_train_pred)

    print(f"Précision : {round(_precision,2)*100} % \nRecall : {round(_recall,2)*100} %")
    return precision_score, recall_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    C'est moins flatteur que les 95% d'accuracy.

    ### F1-score

    Pour combiner la précision et le recall, on définit le **F1-score** comme la **moyenne harmonique** des deux métriques.
    $$F_1=\frac{2}{\frac{1}{\text{précision}}+\frac{1}{\text{recall}}}=2\times\frac{\text{précision}\times \text{recall}}{\text{précision}+\text{recall}}=\frac{\text{TP}}{\text{TP}+\frac{\text{FN}+\text{FP}}{2}} \in \left[ 0, 1 \right]$$
    L’avantage de la moyenne _harmonique_, c’est qu’elle pénalise fortement les **faibles valeurs**. Un modèle n'obtient donc un bon F1-score que si les deux métriques sont simultanément élevées (mais ce n'est pas toujours ce que l'on veut !).

    Là aussi, Scikit facilite l'implémentation.
    """)
    return


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import f1_score

    f1_score(y_train_5, y_train_pred)
    return (f1_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## D. Le compromis précision / recall

    Maximiser **simultanément** la précision et le recall d'un modèle de classification est un problème difficile. Pour s'en convaincre et tenter de trouver un compromis, on décide de s'intéresser de nouveau au `SGDClassifier`.

    On l'a vu tout à l'heure, le `SGDClassifier` classe une instance $\mathbf{x}^{(i)}$ positivement à condition que
    $$\text{decision}\_\text{function}(\mathbf{x}^{(i)})\geq\text{threshold}$$

    Par défaut ce seuil (threshold) vaut 0. Si l'on décide de l'augmenter, on va mécaniquement réaliser moins de prédictions positives, mais puisque ces prédictions seront associées à un score plus élevé, il est plus probables qu'elles soient justes : on augmente la **précision**.

    Ce faisant, certaines instances réellement positives risquent d'échapper au modèle. La proportion d'instances positives classées comme telles par le modèle va diminuer : on dégrade le **recall**.

    Le threshold n'est pas un paramètre natif du modèle `SGDClassifier`, mais il dispose de la méthode `decision_function()`.
    """)
    return


@app.cell
def _(sgd_clf, some_digit):
    # `some_digit` c'est l'instance numéro 0 ; c'est le 5 qu'on a visualisé avec la colormap
    decision_score = sgd_clf.decision_function([some_digit])
    decision_score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On a $2164\geq0\Longleftrightarrow\text{decision}\_\text{function}(\mathbf{x}^{(0)})\geq\text{threshold}$, c'est pourquoi le modèle reconnait effectivement l'instance comme un 5.
    """)
    return


@app.cell
def _(sgd_clf, some_digit):
    # Prédiction du modèle, avec un threshold interne nul
    sgd_clf.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pour déterminer la valeur optimale du threshold, on voudrait disposer des scores de la decision_function et mesurer l'évolution de la précision/du recall lorsque l'on change le seuil. Et si possible, il faudrait que ces scores soient issus d'une validation croisée.

    Scikit intègre directement cette fonctionnalité. Dans `cross_val_predict()`, il suffit de préciser `method="decision_function"` pour que la fonction ne renvoie plus les prédictions mais les scores de décision.
    """)
    return


@app.cell
def _(X_train, cross_val_predict, sgd_clf, y_train_5):
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    return (y_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ensuite, on appelle `precision_recall_curve()` pour calculer la précision et le recall pour toutes les valeurs possibles du threshold.

    On visualise le tout avec Matplotlib.
    """)
    return


@app.cell
def _(y_scores, y_train_5):
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    return precision_recall_curve, precisions, recalls, thresholds


@app.cell(hide_code=True)
def _(plt, precisions, recalls, thresholds):
    threshold = 3000

    plt.figure(figsize=(8, 4))  
    plt.plot(thresholds, precisions[:-1], "--", label="Precision", color="steelblue", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "-", label="Recall", color="gold", linewidth=2)
    plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
    idx = (thresholds >= threshold).argmax() 
    plt.plot(thresholds[idx], precisions[idx], "o", color="steelblue")
    plt.plot(thresholds[idx], recalls[idx], "o", color="gold")
    plt.axis([-50000, 50000, 0, 1])
    plt.grid()
    plt.xlabel("Threshold")
    plt.legend(loc="center right")
    plt.show()
    return (idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut aussi tracer directement l'évolution de la précision en fonction du recall.
    """)
    return


@app.cell(hide_code=True)
def _(idx, plt, precisions, recalls):
    import matplotlib.patches as patches 

    plt.figure(figsize=(6, 5)) 

    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

    plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
    plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
    plt.plot([recalls[idx]], [precisions[idx]], "ko",
             label="Point at threshold 3,000")
    plt.gca().add_patch(patches.FancyArrowPatch(
        (0.79, 0.60), (0.61, 0.78),
        connectionstyle="arc3,rad=.2",
        arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
        color="#444444"))
    plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")
    plt.title("Courbe Precision-Recall (PR)")

    plt.show()
    return (patches,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ces diagrammes nous donnent un bon aperçu de la difficulté à trouver un compromis précision / recall.

    >En pratique, on impose donc souvent une condition sur une seule des deux métriques.

    Ainsi, pour déterminer la plus petite valeur du threshold qui satisfasse à une précision donnée (disons 90%), il vaut mieux utiliser la méthode `argmax()` du package Numpy.
    """)
    return


@app.cell
def _(precisions, thresholds):
    idx_for_90_precision = (precisions >= 0.90).argmax()
    threshold_for_90_precision = thresholds[idx_for_90_precision]
    threshold_for_90_precision
    return (threshold_for_90_precision,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Une fois le seuil fixé, on construit le vecteur des prédictions à la main.
    """)
    return


@app.cell
def _(
    precision_score,
    recall_score,
    threshold_for_90_precision,
    y_scores,
    y_train_5,
):
    y_train_pred_90 = (y_scores >= threshold_for_90_precision)

    _precision = precision_score(y_train_5, y_train_pred_90)
    _recall = recall_score(y_train_5, y_train_pred_90)

    print(f"Précision : {round(_precision,2)*100} % \nRecall : {round(_recall,2)*100} %")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## E. Automatiser le choix du seuil

    Cette section est absente de la dernière édition du livre. Elle présente deux alternatives à `SGDClassifier` qui intègrent nativement la gestion du threshold.

    > Ces estimateurs ont été introduits dans la version 1.5 de Scikit (mai 2024).

    ### La chaîne de traitement

    On veut pouvoir mettre en place efficacement la chaîne de traitement suivante :
    1. On entraîne par validation croisée un `SGDclassifieur`
    2. On extrait les valeurs de la fonction de décision calculées sur les folds de tests
    3. On calcule l'évolution des métriques pour des classifieurs associés à différents seuils
    4. On choisit une condition sur une métrique (par exemple, 80% de précision)
    5. On sélectionne le plus petit threshold satisfaisant à cette condition
    6. On implémente un `SGDclassifieur` avec ce seuil custom

    ### L'estimateur `FixedThresholdClassifier`

    Cet estimateur permet d'automatiser l'étape 6. Ça a l'avantage de nous laisser un contrôle explicite sur toute la partie amont du traitement mais surtout, ça nous permet d'**intégrer notre modèle dans une pipeline**.
    """)
    return


@app.cell
def _(SGDClassifier, X_train, precisions, some_digit, thresholds, y_train_5):
    from sklearn.model_selection import FixedThresholdClassifier

    idx_for_80_precision = (precisions >= 0.80).argmax()
    threshold_for_80_precision = thresholds[idx_for_80_precision]


    sgd_clf_90 = FixedThresholdClassifier(SGDClassifier(random_state=42), threshold=threshold_for_80_precision,
                                          response_method="decision_function")

    sgd_clf_90.fit(X_train, y_train_5)
    sgd_clf_90.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La condition de précision à 80% est assez faible et impose un threshold de -1328. Sachant que l'instance `some_digit` a un score de décision de 2164, la condition decision_function > threshold est largement vérifiée. En imposant 90% de précision, le threshold monte à 3370 et la condition n'est plus vérifiée : le modèle classe alors l'instance négativement.

    ### L'estimateur `TunedThresholdClassifierCV`

    Cet estimateur automatise quasiment toute la chaîne, des étapes 1 à 5.

    Le seuil n'est plus un nombre que l'on calcule à la main, mais un **paramètre** que l'estimateur règle lui-même pendant le `fit` pour respecter une condition sur une métrique donnée.

    Contrairement à la chaîne manuelle, `TunedThresholdClassifierCV` ne sélectionne pas le seuil sous une contrainte du type « précision ≥ 90% » : il **maximise** spécifiquement une métrique. On a choisi ici le F1-score.

    > On peut toutefois parvenir à implémenter cette logique de contrainte « précision ≥ 90% », en mettant en place un scorer (`make_scorer`). On choisit de ne pas le détailler pas ici, mais c'est assez simple à implémenter.
    """)
    return


@app.cell
def _(SGDClassifier, X_train, some_digit, y_train_5):
    from sklearn.model_selection import TunedThresholdClassifierCV

    tuned_clf = TunedThresholdClassifierCV(SGDClassifier(random_state=42), scoring="f1", response_method="decision_function", cv=3, random_state=42,)

    tuned_clf.fit(X_train, y_train_5)
    tuned_clf.predict([some_digit])
    return (tuned_clf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Une fois entraîné, le threshold retenu et le F1-score sont accessibles via les attributs `best_threshold_` et `best_score_`.
    """)
    return


@app.cell
def _(tuned_clf):
    tuned_clf.best_threshold_, tuned_clf.best_score_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## F. La courbe ROC

    La **courbe ROC** est un autre outil très utilisé pour évaluer les classifieurs binaires.

    Au lieu de tracer la précision en fonction du recall (courbe PR), on trace l'évolution du **TPR** (True Positive Rate, c'est le recall) en fonction du **FPR** (False Positive Rate).

    - TPR (recall) : proportion d'instances positives classées comme telles par le modèle
    - FPR : proportion d'instances négatives classées positivement par le modèle

    On définit aussi la specifity ou le TNR (True Negative Rate), de sorte que :
    $$ \text{FPR} = 1 - \text{TNR} = 1 – \text{specificity} $$

    > Courbe PR et ROC sont liées : une courbe qui en domine une autre dans l'espace ROC la domine aussi dans l'espace PR (Davis et Goadrich, 2006). On préfère généralement la courbe PR lorsque la classe positive est rare, ou lorsqu'on accorde plus d’importance aux faux positifs qu’aux faux négatifs.

    L'implémentation Python est identique au tracé de la courbe precision-recall.
    """)
    return


@app.cell
def _(y_scores, y_train_5):
    from sklearn.metrics import roc_curve

    fpr, tpr, roc_thresholds = roc_curve(y_train_5, y_scores)
    return fpr, roc_thresholds, tpr


@app.cell(hide_code=True)
def _(fpr, patches, plt, roc_thresholds, threshold_for_90_precision, tpr):
    idx_for_threshold_at_90 = (roc_thresholds <= threshold_for_90_precision).argmax()
    tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2,color="steelblue", label="ROC curve")
    plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
    plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

    plt.gca().add_patch(patches.FancyArrowPatch(
        (0.20, 0.89), (0.07, 0.70),
        connectionstyle="arc3,rad=.4",
        arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
        color="#444444"))
    plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid()
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right", fontsize=13)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    À nouveau, il n'est pas possible de maximiser **simultanément** FPR et TPR.

    > Sans surprise, la courbe ROC d'un bon classifieur doit se rapprocher de l'angle haut-gauche et l'aire sous la courbe (AUC) doit tendre vers 1.

    À ce propos, l'**AUC-ROC** est une métrique simple et fréquemment utilisée pour comparer deux modèles. On l'implémente typiquement avec la fonction `roc_auc_score` de Scikit.
    """)
    return


@app.cell
def _(y_scores, y_train_5):
    from sklearn.metrics import roc_auc_score

    roc_auc_score(y_train_5, y_scores)
    return (roc_auc_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## G. Comparatif des modèles

    On entraîne un autre classifieur, une forêt aléatoire, dans le but de mettre en perspectives nos métriques.

    Pour tracer la courbe PR, on doit réaliser des prédictions associées à différentes valeurs du seuil. On s'appuyait jusque là sur la fonction de décision ajustée par le modèle pendant l'entraînement, mais `RandomForestClassifier` ne dispose pas de fonction de score de décision interne. Il fournit toutefois une probabilité d'appartenance à la classe : on s'en servira comme d'un score.
    """)
    return


@app.cell
def _(X_train, cross_val_predict, y_train_5):
    from sklearn.ensemble import RandomForestClassifier

    forest_clf = RandomForestClassifier(random_state=42)

    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_probas_forest.shape
    return (y_probas_forest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Avec l'option `method="predict_proba"`, on se retrouve avec un tableau (une colonne par classe). Chaque colonne représente la probabilité d'appartenance à la classe : c'est une information utile, sauf pour le problème de classification binaire où les deux probabilités somment à 1.

    On fournit donc la deuxième colonne (classe positive) du tableau à la fonction `precision_recall_curve()` pour tracer la courbe PR.
    """)
    return


@app.cell
def _(precision_recall_curve, y_probas_forest, y_train_5):
    y_scores_forest = y_probas_forest[:, 1]

    precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
    return precisions_forest, recalls_forest, y_scores_forest


@app.cell(hide_code=True)
def _(plt, precisions, precisions_forest, recalls, recalls_forest):
    plt.figure(figsize=(6, 5))  
    plt.plot(recalls_forest, precisions_forest, "-", color="steelblue",
             linewidth=2, label="Random Forest")
    plt.plot(recalls, precisions, "--", color ="gold", linewidth=2, label="SGD")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")
    plt.title("Courbe Precision-Recall (PR)")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La forêt aléatoire fait clairement mieux. Comparons les autres métriques.
    """)
    return


@app.cell
def _(
    X_train,
    cross_val_predict,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    tuned_clf,
    y_probas_forest,
    y_scores,
    y_scores_forest,
    y_train_5,
):
    # On se dispense de relancer un cross_val_predict() sans l'option method="predict_proba"
    y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%

    # Forêt aléatoire 
    _f1_forest        = f1_score(y_train_5, y_train_pred_forest)
    _precision_forest = precision_score(y_train_5, y_train_pred_forest)
    _recall_forest    = recall_score(y_train_5, y_train_pred_forest)
    _roc_auc_forest   = roc_auc_score(y_train_5, y_scores_forest)   # score continu

    # SGD 
    y_tuned_pred = cross_val_predict(tuned_clf, X_train, y_train_5, cv=3)
    _f1_sgd        = f1_score(y_train_5, y_tuned_pred)
    _precision_sgd = precision_score(y_train_5, y_tuned_pred)
    _recall_sgd    = recall_score(y_train_5, y_tuned_pred)
    _roc_auc_sgd   = roc_auc_score(y_train_5, y_scores)

    print(f"--- SGDClassifier - maximisation F1-score ---\nF1-score  : {_f1_sgd*100:.1f} %\nAUC-ROC   : {_roc_auc_sgd*100:.1f} %\nPrécision : {_precision_sgd*100:.1f} %\nRecall    : {_recall_sgd*100:.1f} %\n")

    print(f"--- Forêt aléatoire ---\nF1-score  : {_f1_forest*100:.1f} %\nAUC-ROC   : {_roc_auc_forest*100:.1f} %\nPrécision : {_precision_forest*100:.1f} %\nRecall    : {_recall_forest*100:.1f} %")
    return


if __name__ == "__main__":
    app.run()
