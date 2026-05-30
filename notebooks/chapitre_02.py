import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ce notebook est inspiré du chapitre 2 de Hands-On Machine Learning (3e éd.) d'Aurélien Géron. Le code est adapté de [ageron/handson-ml3](https://github.com/ageron/handson-ml3), sous licence Apache 2.0.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    assert sys.version_info >= (3,7)

    from packaging import version
    import sklearn

    # On vérifie que la version de Scikit-Learn est >= à 1.0.1
    assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
    return (sys,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Récupération et découverte des données

    On télécharge le jeu de données *California Housing*, on le charge dans un DataFrame, puis on l'explore brièvement.
    """)
    return


@app.cell
async def _(sys):
    import tarfile
    from pathlib import Path
    import pandas as pd

    async def load_housing_data():
        tarball_path = Path("datasets/housing.tgz")
        csv_path = Path("datasets/housing/housing.csv")
        if not csv_path.is_file():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            url = "https://raw.githubusercontent.com/ageron/data/main/housing.tgz"
            if "pyodide" in sys.modules:
                import pyodide.http
                response = await pyodide.http.pyfetch(url)
                data = await response.bytes()
                with open(tarball_path, "wb") as f:
                    f.write(data)
            else:
                import urllib.request
                urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="datasets", filter="data")

        return pd.read_csv(csv_path)

    housing = await load_housing_data()
    return housing, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Premier aperçu des distributions

    On trace l'histogramme de chaque variable numérique pour visualiser les ordres de grandeur et la forme des distributions.
    """)
    return


@app.cell
def _(housing):
    import matplotlib.pyplot as plt

    # bins <-> nombre de colonnes 
    housing.hist(bins=50, figsize=(9, 6))
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Création du jeu de test

    ## Séparation aléatoire naïve

    On veut répartir aléatoirement les données en deux datasets (training set et test set).
    On propose une première implémentation naïve de cette répartition.

    ```python
    import numpy as np

    def shuffle_and_split_data(data, test_ratio):
        # Un tableau NumPy contenant une permutation aléatoire de [0, 1, ..., len(data)-1]
        shuffled_indices = np.random.permutation(len(data))

        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        return data.iloc[train_indices] , data.iloc[test_indices]

    # On a généralement un ratio de 80/20 %
    _train_set, _test_set = shuffle_and_split_data(housing, 0.2)
    ```

    Une implémentation plus idiomatique et concise mobilise la fonction `train_test_split()` de Scikit-Learn.
    ```python
    from sklearn.model_selection import train_test_split

    # On peut rendre cette division reproductible avec le paramètre `random_state=42`
    _train_set, _test_set = train_test_split(housing, test_size=0.2)
    ```

    ---

    ## Séparation reproductible par hachage

    Cette simple séparation aléatoire pose problème : **à chaque exécution, le jeu de test change, et le modèle finit par "voir" l'ensemble des données**.

    On pourrait sauvegarder les sets d'entraînement et de test une fois générés, ou fixer une graine aléatoire pour garantir la reproductibilité de la génération, mais ces deux approches échouent dès que le dataset est mis à jour.

    Une meilleure solution consiste à attribuer à chaque instance un  **identifiant unique et immuable** et à le **hasher**.

    On procède ainsi :
    1. On calcule un hash de l'identifiant.
    2. On place l'instance dans le jeu de test si `hash ≤ 20 % de la valeur max du hash`.

    Cette répartition est stable d'une exécution à l'autre et reste cohérente même après mise à jour du dataset.

    ```python
    from zlib import crc32

    def is_id_in_test_set(identifier, test_ratio):
        # max_hash = 2**32 ; car crc32 retourne un entier non signé sur 32 bits
        return crc32(np.int64(identifier)) < test_ratio * 2**32

    def split_data_with_id_hash(data, test_ratio, id_column):
        ids = data[id_column] # vecteur des identifiants des instances
        in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_,test_ratio)) # vecteur de booléen

        # Donne toutes les lignes où la colonne du vecteur de booléen est True
        # ~ inverse le masque (équivalent du `not` pour les booléens vectorisés)
        return data.loc[~in_test_set], data.loc[in_test_set]

    housing_with_id = housing.reset_index() # ajoute une colonne `index`

    _train_set, _test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
    ```

    ### Construire un identifiant robuste

    Il faut être prudent lorsque l'on choisit le numéro de la ligne comme identifiant. En cas de mise à jour du dataset, il faut s'assurer que les identifiants existants soient inchangés. Cela requiert notamment que les lignes des nouvelles données soient ajoutées à la fin du dataset.

    Le mieux est toutefois de **créer un identifiant plus robuste à partir d'une ou plusieurs feature existantes**. On peut par exemple combiner la longitude et la latitude (<1000).

    ```python
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

    _train_set, _test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
    ```

    ---

    ## Échantillonnage stratifié

    On veut maintenant tester la représentativité du test_set vis à vis de la feature `median_income`, que l'on considère très importante pour la prédiction de prix de l'immobilier.

    Puisque le `median_income` est continu, on va essayer d'échelonner les valeurs en plusieurs catégories (income category attribute).

    Une approche possible consiste à échelonner régulièrement entre min_median_income et max_median_income.
    """)
    return


@app.cell
def _(housing, pd):
    import numpy as np

    # Crée une colonne "income_cat" dont les valeurs sont dans [1,2,3,4,5]
    housing_with_cat = housing.assign(
        income_cat=pd.cut(housing["median_income"],
                          bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                          labels=[1, 2, 3, 4, 5])
    )
    return housing_with_cat, np


@app.cell
def _(housing_with_cat, plt):
    housing_with_cat["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    NB : Le **stratified sampling**, c'est le fait de forcer la représentativité du test set et du train set vis-à-vis d'une feature particulière. On procède ainsi :
    1. On **découpe la population en strates**
    2. On **échantillonne dans chaque strate** en respectant sa proportion dans la population totale

    Les strates n'ont pas besoin d'être de tailles semblables, la distribution est prise en compte dans l'échantillonnage **proportionnel** des strates.


    ### Les implémentations possibles du **stratified sampling** :

    En pratique, on génère soit un seul split train/test (pour réserver un test set final), soit plusieurs splits pour faire de la _cross validation_ (validation croisée) sur le training set. Dans ce deuxième cas de figure, deux approches sont possibles.

    1. **StratifiedShuffleSplit** : On réalise k tirages aléatoires indépendants, chacun produisant un split train/test. On  prélève proportionnellement dans chaque strate pour générer les deux sets. **Ici les test sets se chevauchent** : une même instance peut se retrouver dans plusieurs test sets (ou dans aucun).
    2. **StratifiedKFold** : **On partitionne le dataset en k test sets**, ce qui donne par simple passage au complémentaire k splits train/test. Cette partition est faite de sorte à respecter elle aussi la distribution des strates.

    Les versions non stratifiées correspondantes (ShuffleSplit, KFold) existent mais ne sont pas détaillées ici.

    ### Tableau comparatif

    | Aspect | `StratifiedKFold(k)` | `StratifiedShuffleSplit(n_splits=k, test_size=0.2)` |
    |---|---|---|
    | Type de découpage | Partition exhaustive | Tirages aléatoires indépendants |
    | Test sets disjoints ? | Oui | Non, peuvent se chevaucher |
    | Couvre tout le dataset ? | Oui | Non garanti |
    | Taille du test set | Imposée : `n/k` | Libre : `test_size` |
    | Stratification | Oui | Oui |
    | Reproductible avec `random_state` | Oui | Oui |


    Une implémentation explicite de cet échantillonage stratifié :

    ```
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    strat_splits = []
    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])

    strat_train_set, strat_test_set = strat_splits[0]
    ```

    En pratique on utilise plutôt `train_test_split()`, qui fait la même chose de façon plus concise.
    """)
    return


@app.cell
def _(housing_with_cat):
    from sklearn.model_selection import train_test_split

    # On précise le dataset initial, le ratio (80/20) et la feature à stratifier
    strat_train_set_full, strat_test_set_full = train_test_split(
        housing_with_cat, test_size=0.2,
        stratify=housing_with_cat["income_cat"], random_state=42)
    return strat_test_set_full, strat_train_set_full


@app.cell
def _(plt, strat_test_set_full):
    # On trace le même histogramme que tout à l'heure mais sur test_set. Même allure !

    strat_test_set_full["income_cat"].value_counts().sort_index().plot.bar(rot=0,
    grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()
    return


@app.cell
def _(strat_test_set_full, strat_train_set_full):
    # Une fois le split effectué, cette feature n'est plus utile (car redondante vis-à-vis de `median_income `)
    strat_train_set = strat_train_set_full.drop("income_cat", axis=1)
    strat_test_set  = strat_test_set_full.drop("income_cat", axis=1)
    return strat_test_set, strat_train_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploration et visualisation des données

    Avant de préparer les données pour l'apprentissage, on les explore sur une copie du jeu d'entraînement, ce qui permet de manipuler les données sans risque.

    ## Visualisation géographique des données
    """)
    return


@app.cell
def _(strat_train_set):
    # Phase d'analyse : on crée une copie du set d'entraînement pour manipuler sans crainte les données
    housing_exp_set = strat_train_set.copy()
    return (housing_exp_set,)


@app.cell
def _(housing_exp_set, plt):
    housing_exp_set.plot(
        kind="scatter", 
        x="longitude", y="latitude",
        grid=True,
        s=housing_exp_set["population"] / 100, 
        label="population",
        c="median_house_value", 
        cmap="jet", 
        colorbar=True,
        legend=True, 
        sharex=False, 
        figsize=(10, 7))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Recherche de corrélations

    ### Le coefficient de corrélation de Pearson

    Pour deux variables aléatoires $X$ et $Y$ avec variances finies non nulles :

    $$\rho(X, Y) = \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)} \sqrt{\operatorname{Var}(Y)}} = \frac{\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]}{\sqrt{\operatorname{Var}(X)} \sqrt{\operatorname{Var}(Y)}}$$

    Estimateur empirique sur un échantillon de taille $n$ :

    $$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2} \sqrt{\sum_i (y_i - \bar{y})^2}}$$

    C'est ce que `housing.corr()` calcule pour chaque paire de colonnes.

    ### L'interprétation géométrique

    Le coefficient de corrélation linéaire mesure la dépendance affine de $X$ et $Y$. Ainsi, si $|\rho(X, Y)| = 1$, il existe des réels $a$ et $b$ tels que $Y = aX + b$. À l'autre bout de l'échelle, si $X$ et $Y$ sont indépendantes, $\rho(X, Y) = 0$, la réciproque étant fausse.

    ### Lien avec la régression linéaire

    En cherchant la droite des moindres carrés $y = \hat{a}x + \hat{b}$ minimisant $\sum_i (y_i - ax_i - b)^2$, on obtient :

    $$\hat{a} = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}, \qquad r = \hat{a} \cdot \frac{\sigma_X}{\sigma_Y}$$

    Et surtout : $r^2$ est le **coefficient de détermination**, la fraction de la variance de $Y$ expliquée par la régression linéaire sur $X$ :

    $$r^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

    C'est pour cela qu'en ML on regarde la corrélation : $r^2$ indique, intuitivement, « quelle part du signal de $Y$ peut être capturée par une simple droite sur $X$ ».

    ### Ce que fait `housing.corr()` concrètement

    Pour un dataset avec $p$ variables numériques $X_1, \dots, X_p$, la matrice de corrélation $C \in \mathbb{R}^{p \times p}$ est définie par :

    $$C_{ij} = \rho(X_i, X_j) = \frac{\operatorname{Cov}(X_i, X_j)}{\sigma_{X_i} \, \sigma_{X_j}}$$

    Concrètement, pour 8 colonnes numériques dans `housing`, `housing.corr()` renvoie une matrice $8 \times 8$ où la case $(i, j)$ contient la corrélation entre la colonne $i$ et la colonne $j$.

    Ainsi `housing.corr()` renvoie un DataFrame qui ressemble à :

    ```
                        longitude  latitude  median_income  ...  median_house_value
    longitude            1.000    -0.925     -0.015              -0.046
    latitude            -0.925     1.000     -0.080              -0.144
    median_income       -0.015    -0.080      1.000               0.688
    ...
    median_house_value  -0.046    -0.144      0.688               1.000
    ```

    On regarde typiquement **la colonne (ou ligne) de la variable cible** : ici `median_house_value`. Les features avec **$|r|$** élevé sont de bonnes candidates pour prédire la cible.
    """)
    return


@app.cell
def _(housing_exp_set):
    corr_matrix = housing_exp_set.corr(numeric_only=True)

    corr_matrix["median_house_value"].sort_values(ascending=False)
    return


@app.cell
def _(housing_exp_set, plt):
    # On ne croise que les variables les plus prometteuses
    from pandas.plotting import scatter_matrix
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing_exp_set[attributes], figsize=(12, 8))
    plt.show()

    # Sur la diagonale on a les histogrammes plutôt que des droites y=x sans intérêt
    return


@app.cell
def _(housing_exp_set, plt):
    # On zoome sur la feature `median_income` qui semble prometteuse pour prédire `median_house_value`
    housing_exp_set.plot(
        kind="scatter", 
        x="median_income",
        y="median_house_value",
        alpha=0.1, 
        grid=True)
    plt.show()

    # On observe un plafond à 500k et des paliers (450k, 350k)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Expérimenter des combinaisons d'attributs

    Plutôt que d'utiliser certaines features brutes, on teste de nouvelles features potentiellement plus corrélées à la cible.
    """)
    return


@app.cell
def _(housing_exp_set):
    # On essaye de créer de nouvelles features pertinentes

    housing_exp_extra = housing_exp_set.assign(
        rooms_per_house = housing_exp_set["total_rooms"]    / housing_exp_set["households"],
        bedrooms_ratio  = housing_exp_set["total_bedrooms"] / housing_exp_set["total_rooms"],
        people_per_house= housing_exp_set["population"]      / housing_exp_set["households"],
    )
    return (housing_exp_extra,)


@app.cell
def _(housing_exp_extra):
    # Jetons un oeil à la corrélation linéaire de `median_house_value` avec ces nouvelles features
    corr_matrix_2 = housing_exp_extra.corr(numeric_only=True)
    corr_matrix_2["median_house_value"].sort_values(ascending=False)

    # `bedrooms_ratio` et `rooms_per_house` se démarquent
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Préparation des données

    On repart d'un jeu d'entraînement propre et on sépare les prédicteurs (`housing_train`) de la cible (`housing_labels`) avant d'enchaîner les transformations.
    """)
    return


@app.cell
def _(strat_train_set):
    # On sépare les predicteurs des labels
    housing_train = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing_labels, housing_train


@app.cell
def _(housing_train):
    housing_train.info()

    # Il y a des valeurs manquantes pour la feature `total_bedrooms` (16344 vs 16512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gérer les valeurs manquantes

    Trois possibilités pour gérer ces valeurs manquantes :
    1. `dropna()` : Supprimer les lignes qui contiennent des NaN.
    2. `drop()` : Supprimer une colonne entière (typiquement parce qu'elle contient trop de NaN pour être exploitable)
    3. `fillna()` : Remplacer les NaN par des valeurs (moyenne, médiane ou dernière valeur valide pour une série temporelle). On parle d'**imputation**.

    ### Supprimer les instances corrompues : `dropna()`
    - `housing.dropna(subset=["total_bedrooms"])` supprime les instances dont `total_bedrooms` vaut NaN
    - `housing.dropna(thresh=n)` supprime les instances ayant moins de n valeurs valables
    - `inplace=True` pour modifier le DataFrame en place (plutôt que de renvoyer une copie)

    ### Corriger les instances corrompues : `fillna()`
    - `median = housing["total_bedrooms"].median()` puis `housing["total_bedrooms"].fillna(median)`
    - `.ffill()` ou `.bfill()` pour propager respectivement la dernière/prochaine valeur valide (utile pour les séries temporelles).

    ### Corriger proprement les instances corrompues : `SimpleImputer`
    - On instancie une classe `SimpleImputer`.
    - Elle apprend les médianes sur le training set avec `fit()` (une fois pour toutes).
    - On corrige ensuite avec `transform()` les instances corrompues à chaque fois que c'est nécessaire (sur le training set mais aussi sur le test set).

    Contrairement à `fillna()`, cette séparation évite de recalculer la médiane sur le test set. Cela représenterait une petite fuite d'information (le test set ne doit jamais influencer le modèle).

    ```python
    from sklearn.impute import SimpleImputer

    # 1. On crée l'objet (Stratégies alternatives : "mean", "most_frequent", "constant")
    imputer = SimpleImputer(strategy="median")

    # Création d'une copie du dataset numerical-features-only
    housing_num = housing_train.select_dtypes(include=[np.number])

    # 2. Il apprend les médianes
    imputer.fit(housing_num)

    # 3. Il applique les médianes (et renvoie un numpy.ndarray)
    X = imputer.transform(housing_num)

    # Conversion numpy.ndarray -> DataFrame
    r = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    ```

    ### Autres _imputers_ utiles

    - `KNNImputer` replace chaque valeur manquante par la moyenne des k plus proches voisins. La distance est calculée à partir de toutes les autres features numériques disponibles : c'est très fort, mais c'est coûteux.
    - `IterativeImputer` entraine un modèle de régression linéaire pour prédire les valeurs manquantes.

    ---

    ## Variables catégorielles

    On s'intéresse maintenant à la seule variable catégorielle du jeu de données, `ocean_proximity`, qu'il faut convertir en valeurs numériques exploitables par un modèle.
    """)
    return


@app.cell
def _(housing_train):
    housing_cat = housing_train[["ocean_proximity"]]
    housing_cat.head(8)

    # Convertissons ces catégories textuelles en nombres
    return (housing_cat,)


@app.cell
def _(housing_cat):
    from sklearn.preprocessing import OrdinalEncoder

    # L'estimateur `OrdinalEncoder` transforme les variables catégorielles en entiers (une catégorie = un entier)
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

    # Aperçu des catégories (l'entier attribué est leur indice dans cette liste)
    ordinal_encoder.categories_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Le problème de l'encodage ordinal

    Le problème des variables catégorielles, c'est qu'une fois transformées en entiers, les modèles de ML interprètent deux valeurs proches comme plus semblables que deux valeurs éloignées. Pourtant, dans la colonne `ocean_proximity` les catégories 0 et 4 sont clairement plus similaires que les catégories 0 et 1.

    ### One-hot encoding

    La solution classique à adopter s'appelle le **one-hot encoding**.
    - On **attribue une nouvelle feature** à chaque catégorie.
    - Chaque feature devient **binaire** : 1 si l'instance appartient à cette catégorie, 0 sinon.

    ### Implémentation : Scikit-Learn vs Pandas
    - Scikit-Learn fournit la classe `OneHotEncoder` qui renvoie une matrice creuse, car la plupart des valeurs sont des zéros.
    - Pandas fournit une fonction `get_dummies()` qui fait sensiblement la même chose.

    ### Pourquoi `OneHotEncoder` est meilleur

    - get_dummies() est stateless. C'est une simple fonction pandas, sans mémoire.

    - OneHotEncoder est stateful. C'est un objet qui se souvient. Une fois fit(), son transform() produira toujours les mêmes colonnes, dans le même ordre, quoi qu'il arrive dans les nouvelles données.

    Supposons que notre modèle ait été entraîné sur 5 catégories d'`ocean_proximity`. En production, on reçoit un lot de données où, par hasard, seules 3 de ces catégories apparaissent :

    ```
    # Avec get_dummies : problèmes

    pd.get_dummies(df_test)
    # → ne crée que 3 colonnes au lieu de 5
    # → le modèle attend 5 features et en reçoit 3 → crash ou prédictions absurdes

    # Avec OneHotEncoder déjà fitté : robuste

    cat_encoder.transform(df_test)
    # → crée toujours les 5 colonnes, les 2 absentes sont juste à 0 partout
    # → le modèle reçoit exactement les features attendues
    ```

    Et inversement, si une catégorie jamais vue à l'entraînement apparaît, `OneHotEncoder` peut la gérer proprement (paramètre handle_unknown="ignore"), alors que get_dummies créerait sournoisement une nouvelle colonne que le modèle ne sait pas interpréter.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

    # Conversion en numpy.ndarray
    housing_cat_1hot.toarray()
    ```

    # Mise à l'échelle et transformation des variables

    Les algorithmes de ML performent moins bien si les input features ont des échelles très différentes.
    Une étape préalable de mise à l'échelle des données (dans [0,1] ou [-1,1] pour les réseaux de neurones) est donc souvent nécessaire. On présente plusieurs implémentation possibles.

    ## Méthodes de mise à l'échelle

    ### **Min-max scaling** : scaled_value = (value − min) / (max − min)

    Scikit-Learn fournit la classe-transformer `MinMaxScaler` :
    ```python
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    training_set = min_max_scaler.fit_transform(training_set)
    ```

    ### **Standardization** : scaled_value = (value − μ) / σ

    - Cette méthode ne restreint pas strictement les valeurs dans un intervalle
    - Elle est toutefois moins sensibles aux _outliers_ (valeurs aberrantes)

    Scikit-Learn fournit la classe-transformer `StandardScaler` :
    ```python
    std_scaler = StandardScaler()
    training_set = std_scaler.fit_transform(training_set)
    ```

    ---

    ## Symétriser les distributions difficiles

    ### Travail préalable : le **Heavy tail** shrinking et la symétrisation
    Lorsque la distribution des valeurs s'étale trop vers la droite (heavy tail), le scaling min-max scaling et la standardization
    tendent à **écraser la plupart des valeurs** dans un petit intervalle. Il faut donc préalablement essayer de **contracter** et de **symétriser** les distributions.
    - Une solution possible consiste à considérer le **logarithme** de la feature ou de l'**élever à une puissance comprise entre 0 et 1**.
    - L'autre solution s'appelle le _bucketizing_ (discrétisation) : on **échelonne la distribution** en intervalles réguliers et on **remplace la feature par l'indice** de l'intervalle auquel il appartient.


    ### Symétrisation des distributions **multimodales**
    Une distribution est dite multimodale si elle a deux **modes** (peaks) ou plus.

    - Le **bucketizing** est ici aussi une bonne approche pour symétriser la distribution. Il faut alors considérer les _bucket IDs_ comme des catégories (plutôt que comme des valeurs numériques) puis one-hot-encoder les appartenances.

    - Une autre approche possible consiste à **ajouter une nouvelle feature** pour chaque mode de la distribution, puis de mesurer pour chaque instance la distance à ce mode. On utilise pour la distance une _radial basis function_ (RBF) : **RBF Gaussienne**, Multiquadratique, Multiquadratique inverse..

    La fonction `rbf_kernel()` de Scikit-Learn implémente la RBF Gaussienne.

    NB : En réalité les RBF renvoient une valeur inversement proportionnelle à la distance ; c'est davantage une mesure de similarité.
    """)
    return


@app.cell
def _(housing, plt):
    housing[["housing_median_age"]].hist(bins=50, figsize=(11, 5))
    plt.show()

    # On a clairement une distribution multimodale et un mode à 35
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    from sklearn.metrics.pairwise import rbf_kernel

    # renvoie le vecteur colonne (tableau NumPy) des similarités au mode 35
    # gamma détermine la "vitesse" à laquelle la distance décroît
    age_simil_35 = rbf_kernel(housing[["housing_median_age"]],[[35]], gamma=0.1)
    ```

    ---

    ## Mettre à l'échelle la variable cible

    Plus rarement, on peut vouloir mettre à l'échelle la distribution des valeurs de la feature cible. Si l'on décide par exemple de remplacer ces valeurs par leur logarithme, le modèle apprendra à prédire le logarithme de `median_house_value`.

    Ce n'est pas gênant, à condition que ces transformations soient mathématiquement **réversibles**.
    Le cas échéant, on peut utiliser la méthode `inverse_transform` des transformers vus précedemment (`StandardScaler` par exemple).

    ```python
    from sklearn.preprocessing import StandardScaler

    # Transformer de standardization : scaled_value = (value − μ) / σ
    # Application au label d'entraînement (y) <-> la colonne median_house_value du train set
    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

    from sklearn.linear_model import LinearRegression

    # Entraînement : prédire le prix (scalé) à partir du revenu médian
    lin_reg = LinearRegression()
    lin_reg.fit(housing_train[["median_income"]], scaled_labels)

    # Prédiction sur un échantillon test fictif
    some_new_data = housing_train[["median_income"]].iloc[:5] # pretend this is new data
    scaled_predictions = lin_reg.predict(some_new_data)

    # Mise à l'échelle originale de ces prédictions
    predictions_manual = target_scaler.inverse_transform(scaled_predictions)
    ```

    La classe `TransformedTargetRegressor` permet de combiner de façon concise la mise à échelle des données, le choix du modèle de régression et la mise à l'échelle initiale (l'étape d'inversion) des données.

    ```python
    from sklearn.compose import TransformedTargetRegressor

    # On précise que le modèle de ML est `LinearRegression()` et que la mise à l'échelle de la feature à prédire se fait par Standardization (`StandardScaler()`)
    ttr_model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())

    # Entraînement : prédire le prix (scalé) à partir du revenu médian
    ttr_model.fit(housing_train[["median_income"]], housing_labels)

    # Prédiction sur un échantillon test fictif + mise à l'échelle originale de ces prédictions
    predictions_ttr = ttr_model.predict(some_new_data)
    ```


    # Transformers personnalisés

    Il est souvent utile de créer ses propres Transformers. Plusieurs exemples :

    ## Transformers à partir d'une fonction (`FunctionTransformer`)

    ```python
    from sklearn.preprocessing import FunctionTransformer

    # On crée le Transformer `log_transformer` à partir de la fonction `np.log`
    # Préciser l'inverse permet de l'utiliser dans un `TransformedTargetRegressor`
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)

    # Pas besoin de fit() avec ce Transformer
    log_pop = log_transformer.transform(housing[["population"]])
    ```

    Revenons à la fonction `rbf_kernel()` de Scikit-Learn qui implémente la RBF Gaussienne.
    - Si on lui donne deux features, elle ne calcule pas une similarité pour la latitude et une autre pour la longitude mais **fusionne les deux en une seule distance euclidienne** 2D : il en sort donc une seule colonne de similarité de forme (n, 1) et pas (n, 2).

    Étant donné les features de latitude et de longitude et un point de référence à deux coordonnées (San Francisco ≈ 37.77, −122.41), on peut créer un Transformer bien utile.

    ```python
    sf_coords = 37.7749, -122.41
    sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))

    # Version utilisant le Transformer
    sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

    # Version moins concise
    sf_simil = rbf_kernel(housing[["latitude", "longitude"]], Y=[sf_coords], gamma=0.1)
    ```

    ---

    ## Créer des Transformers _fittables_

    `FunctionTransformer()` ne suffit plus : il nous faut créer une classe disposant des méthodes `fit()` (qui doit retourner `self`), `transform()`, et `fit_transform()`. Une implémentation naïve de la méthode `fit_transform()` s'implémente en appelant consécutivement `fit()` puis `transform()`

    Si on décide de faire hériter notre classe de la classe parente `BaseEstimator` fournie par Scikit-Learn, elle gagne automatiquement les méthodes `get_params()` et `set_params()` :

    - `get_params()` : Permet de récupérer les hyperparamètres actuels de votre classe.
    - `set_params()` : Permet de modifier ces hyperparamètres après la création de l'objet.

    Ces méthodes sont indispensables pour l'**optimisation automatique des hyperparamètres** (avec des outils comme `GridSearchCV` ou `RandomizedSearchCV`). Pour tester différentes combinaisons de paramètres et trouver la meilleure, Scikit-Learn doit être capable de lire et de modifier les paramètres de votre classe automatiquement en utilisant ces deux méthodes exactes.

    Puisque la classe `BaseEstimator` a besoin de regarder le code de `__init__` pour savoir exactement comment s'appellent les paramètres :

    * **Ce qu'il faut faire :** Définir explicitement tous les noms (ex: `def __init__(self, mon_parametre=10):`).
    * **Ce qu'il ne faut pas faire :** Utiliser `args` (liste d'arguments de longueur variable) ou `kwargs` (dictionnaire d'arguments nommés de longueur variable). Sinon Scikit-Learn échoue à deviner le nom des paramètres, empêchant les méthodes `get_params()` et `set_params()` de fonctionner.

    Ci-dessous un exemple de custom Transformer qui fait semblablement la même chose que `StandardScaler`. Par soucis de clarté, on y omet la définition de `feature_names_in_`, de `get_feature_names_out()` et de `inverse_transform()`
    ```python
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_array, check_is_fitted

    class StandardScalerClone(BaseEstimator, TransformerMixin):
        def __init__(self, with_mean=True): # no *args or **kwargs!
            self.with_mean = with_mean

        def fit(self, X, y=None): # y is required even though we don't use it
            X = check_array(X) # checks that X is an array with finite float values
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
            return self # always return self!

        def transform(self, X):
            check_is_fitted(self) # looks for learned attributes (with trailing _)
            X = check_array(X)
            assert self.n_features_in_ == X.shape[1]
            if self.with_mean:
                X = X - self.mean_
            return X / self.scale_
    ```

    ---

    ## À propos des Transformers personnalisés

    Le package `sklearn.utils.validation` fournit des fonctions à intégrer dans la définition des classes des Transformers personnalisés. Elles permettent de vérifier et de formater les données avant qu'un modèle n'essaie de faire des calculs dessus. Parmi elles : `check_array`, `check_X_y`, `check_is_fitted` ou encore `check_estimator()` (voir https://scikit-learn.org/stable/developers pour plus de détails).

    Tout estimateur Scikit-Learn doit définir un attribut `n_features_in_` à l'intérieur de leur méthode `fit()`. Il mémorise le nombre exact de colonnes (features) observées lors de l'entraînement. Lors des appels ultérieurs à `transform()` ou `predict()`, le modèle s'en sert pour vérifier que les nouvelles données aient exactement les mêmes dimensions. Ce mécanisme permet d'interrompre le programme avec un message d'erreur clair et immédiat.

    ---

    ## Un Transformer composite

    On présente la construction d'un Transformer utile, qui s'appuie sur un autre estimateur (`KMeans`) pour produire des features de similarité géographique.
    """)
    return


@app.cell
def _():
    from sklearn.cluster import KMeans
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics.pairwise import rbf_kernel

    class ClusterSimilarity(BaseEstimator, TransformerMixin):
        """
        Les Custom Transformer font souvent appel à d'autres estimateurs.

        Celui-ci utilise un modèle KMeans pour identifier les clusters principaux dans les données d'entraînement.
        Il appelle ensuite `rbf_kernel()` pour mesurer la distance de chaque "sample" au cluster central.
        """
        def __init__(self, n_clusters=10, gamma=1.0,
            random_state=None):
            self.n_clusters = n_clusters
            self.gamma = gamma
            self.random_state = random_state

        def fit(self, X, y=None, sample_weight=None):
            self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
            self.kmeans_.fit(X, sample_weight=sample_weight)
            return self # always return self!

        def transform(self, X):
            return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

        def get_feature_names_out(self, names=None):
            return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

    return (ClusterSimilarity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aparté sur le **k-means clustering**

    Cet algorithme détecte des groupes (clusters) dans les données.
    - `n_clusters` détermine le **nombre de clusters** à rechercher
    - l'attribut `cluster_centers_` enregistre les centres des clusters
    - l'option `sample_weight` de la méthode `fit()` permet d'attribuer des poids relatifs aux **instances**

    L'algorithme **k-means** étant stochastique, il faut fixer le paramètre `random_state` pour garantir la reproductibilité des résultats.
    """)
    return


@app.cell
def _(ClusterSimilarity, housing_labels, housing_train):
    # On utilise le Transformer fraîchement créé
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

    # `housing_labels` correspond à la colonne `median_house_value` (prix médian) du train set
    # `housing_train[["latitude", "longitude"]]` fixe les features utilisées par le clustering et la distance RBF
    similarities = cluster_simil.fit_transform(housing_train[["latitude", "longitude"]], sample_weight=housing_labels)
    return cluster_simil, similarities


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    L'algorithme K-means regroupe les instances en 10 clusters selon leur latitude/longitude. Ce clustering est pondéré par la valeur `median_house_value` de chaque instance.

    Une fois les 10 centres calculés, on génère des vecteurs colonne (tableau NumPy) des similarités (RBF Gaussienne) par rapport aux centres. Ces 10 colonnes sont autant de nouvelles features à ajouter au train set qui quantifient la proximité de chaque instance à ces 10 zones géographiques.

    Cette étape est importante : **la latitude et la longitude brutes sont presque inexploitables** pour un modèle linéaire.
    """)
    return


@app.cell(hide_code=True)
def _(cluster_simil, housing_train, plt, similarities):
    housing_renamed = housing_train.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

    housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                         s=housing_renamed["Population"] / 100, label="Population",
                         c="Max cluster similarity",
                         cmap="jet", colorbar=True,
                         legend=True, sharex=False, figsize=(8, 6))
    plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
             cluster_simil.kmeans_.cluster_centers_[:, 0],
             linestyle="", color="black", marker="X", markersize=20,
             label="Cluster centers")
    plt.legend(loc="upper right")
    plt.savefig("district_cluster_plot.png", dpi=120, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipelines de transformation
    ## Généralités

    Le constructeur `Pipeline` prend en entrée une liste ordonnée de paires (nom, estimateur). Lorsque la pipeline est exécutée avec `Pipeline.fit()`, les méthodes `fit_transform()` des estimateurs sont appelées **consécutivement** - sauf pour le dernier estimateur dans la liste, pour lequel seule `fit()` est appelée. Il faut donc impérativement que les estimateurs soient des **Transformers**, à l'exception du dernier.

    ```python
    from sklearn.pipeline import Pipeline

    num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("standardize", StandardScaler()),])
    ```

    Dans cet exemple, la pipeline intègre un traitement simple des input features du train set :
    - Correction des instances corrompues avec `SimpleImputer` ( `NaN` -> `values.median()` )
    - Standardization (feature scaling où scaled_value = (value − μ) / σ ) avec `StandardScaler`


    Mais on peut aussi se dispenser de nommer les estimateurs en utilisant la fonction `make_pipeline()`. Les noms associés sont alors ceux des classes des Transformers.
    ```python
    from sklearn.pipeline import make_pipeline

    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    ```

    NB : La pipeline dispose des **mêmes méthodes** que le dernier estimateur.

    ```python
    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    ```

    ---

    ## Fonctionnalités du transformateur `Pipeline`
    - `pipeline[1]` retourne le deuxième estimateur de la pipeline
    - `pipeline[:-1]` retourn un objet Pipeline contenant tous les estimateurs sauf le dernier
    - `num_pipeline["simpleimputer"]` retourne l'estimateur dont le nom est "simpleimputer"
    - `if "simpleimputer" in num_pipeline` renvoie un booléen

    ---

    ## Gérer simultanément les features numériques et catégorielles

    Jusque là, on a divisé manuellement le DataFrame en deux, on a appliqué les transformations séparément puis on a concaténé les résultats ensemble. Grâce au `ColumnTransformer`, on peut se simplifier la tâche en gérant **simultanément** les deux types de variables. L'inconvénient est que cet objet est plus rigide (il **ne supporte pas les fonctionnalités** ci-dessus, notamment puisqu'il parallélise).

    ```python
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline

    # On définit la liste des features catégorielles et des features numériques
    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]

    # On définit les deux pipeline de traitement
    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    # On combine les deux avec le très utile `ColumnTransformer`
    _preprocessing = ColumnTransformer([ ("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs),])
    ```

    ---

    ## Sélection dynamique et nommage automatique des colonnes

    Énumérer les features de chaque type pose deux problèmes : c'est fastidieux dès lors qu'on a un grand nombre de variables et c'est peu robuste si on ré-entraine notre Pipeline sur un Dataset qui a un peu changé.

    **La solution :** `make_column_selector` agit comme un filtre dynamique. Au lieu de lire les noms des colonnes, il regarde leur **type de données** (`dtype` dans pandas).

    ```python
    from sklearn.compose import make_column_selector

    # On crée des sélecteurs dynamiques
    numerical_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=object) # ou bool, etc.

    # On les passe au ColumnTransformer à la place des listes de noms
    full_pipeline = ColumnTransformer([("num", num_pipeline, numerical_selector), ("cat", cat_pipeline, categorical_selector),])
    ```

    Toutefois, `ColumnTransformer` exige de donner un nom à chaque branche (`"num"` et `"cat"` dans l'exemple ci-dessus). C'est inutilement contraignant : il nous faut un outil qui automatise ce nommage, comme le fait `make_pipeline()`.

    **La solution :** La fonction `make_column_transformer()` fait tout comme `ColumnTransformer`, mais elle génère automatiquement les noms des estimateurs (généralement en utilisant le nom de la classe).


    ```python
    from sklearn.compose import make_column_transformer

    preprocessing_auto_named = make_column_transformer((num_pipeline, numerical_selector), (cat_pipeline, categorical_selector))

    # Dataset pipeliné (tableau NumPy)
    _housing_prepared = preprocessing_auto_named.fit_transform(housing_train)
    ```

    ---

    ## Bilan sur la Pipeline implémentée

    Récapitulons ce qu'elle fait :
    - Les valeurs manquantes respectivement dans les variables numériques et catégorielles sont remplacées par la médiane et la catégorie la plus fréquente.
    - Les features catégorielles sont one-hot encodées.
    - De nouvelles features sont ajoutées (`bedrooms_ratio`, `rooms_per_house`, et `people_per_house`).
    - Une feature de cluster-similarity est ajoutée, potentiellement plus utile que la longitude et la latitude.
    - Les features avec une _heavy tail_ sont remplacées par leur logarithme
    - Toutes les features numériques (cible exclue) sont standardisées
    """)
    return


@app.cell
def _(cluster_simil, np):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.pipeline import make_pipeline

    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(function_transformer, feature_names_in):
        return [f"{feature_names_in[0]}_per_{feature_names_in[1]}"]

    def ratio_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(column_ratio, feature_names_out=ratio_name),
            StandardScaler())

    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    preprocessing = ColumnTransformer([
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                   "households", "median_income"]),
            ("geo", cluster_simil, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline)  # one column remaining: housing_median_age
    return make_pipeline, preprocessing


@app.cell
def _(housing_train, preprocessing):
    housing_prepared = preprocessing.fit_transform(housing_train)
    housing_prepared.shape
    return


@app.cell
def _(preprocessing):
    preprocessing.get_feature_names_out()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Récapitulatif des variables déclarées jusqu'ici

    | Variables | Utilité | Catégorie |
    | --- | --- | --- |
    | `housing` | Dataset brut complet (20 640 lignes) chargé du CSV. | **Source** |
    | `strat_train_set`, `strat_test_set` | Split stratifié train/test, `income_cat` retirée. | **Définitive** |
    | `housing_train` | Prédicteurs d'entraînement (`X`) = `strat_train_set` sans `median_house_value`. | **Définitive** |
    | `housing_labels` | Labels d'entraînement (`y`) = la colonne `median_house_value` du train set. | **Définitive** |
    | `cluster_simil` | Transformer `ClusterSimilarity`, réutilisé comme branche `"geo"` du pipeline final. | **Définitive** |
    | `preprocessing`, `housing_prepared` | Pipeline final et jeu de données transformé, prêt pour l'entraînement. | **Définitive** |
    | `housing_with_cat` | Copie de `housing` augmentée de la colonne `income_cat` (split stratifié). | Transitoire |
    | `strat_train_set_full`, `strat_test_set_full` | Split stratifié avant retrait d'`income_cat` (graphe de vérification de la stratification). | Transitoire |
    | `income_cat` | Feature catégorielle de revenu créée dans `housing_with_cat` pour stratifier, absente des sets définitifs. | Transitoire |
    | `housing_exp_set` | Copie du train set pour exploration (corrélations, scatter plots). | Illustrative (exploration) |
    | `housing_exp_extra` | `housing_exp_set` augmentée des features `rooms_per_house`, `bedrooms_ratio`, `people_per_house`. | Illustrative (exploration) |
    | `corr_matrix`, `corr_matrix_2`, `attributes` | Analyse de corrélation et choix des features à tracer. | Illustrative (exploration) |
    | `ordinal_encoder`, `housing_cat`, `housing_cat_encoded` | Démo d'encodage avec `OrdinalEncoder` exécutée dans le notebook. | Illustrative (démo transformer) |
    | `similarities`, `housing_renamed` | Sortie de démonstration de `ClusterSimilarity` et données mises en forme pour la Figure 2-19. | Illustrative (figure) |

    Bilan : seules les lignes « Définitive » servent réellement à l'entraînement. Les variables « Transitoire » ne servent qu'à fabriquer le split stratifié ; les « Illustrative » servent à explorer les données ou à démontrer une méthode, avant que le pipeline ne reconstruise tout proprement sur `housing_train`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sélection et entraînement du modèle

    ## Évaluation et entraînement sur le train set

    On va entraîner deux modèles et évaluer leur prédictions sur le train set (**training error**). Pourquoi pas sur le test set ? Car on veut éviter que ce dernier induise un trop fort biais de sélection du modèle. En principe, on n'évalue pas sur le test set avant d'avoir définitivement sélectionné un modèle.

    ### 1. Régression linéaire multiple : l'équation locale (à l'échelle d'un district)

    Pour un district donné, le modèle fait l'hypothèse que le prix de l'immobilier est une combinaison linéaire de toutes ses caractéristiques après traitement. L'équation mathématique s'écrit :

    $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$

    Où :

    * $y$ est la variable cible (le prix médian que l'on cherche à deviner, stocké dans `housing_labels`).
    * $x_1, x_2, \dots, x_p$ sont les $p$ caractéristiques numériques générées en sortie du `preprocessing` (le logarithme du revenu médian, les ratios de chambres, les 10 variables de similarité géographique, etc..).
    * $\beta_0$ est l'ordonnée à l'origine (*intercept*). C'est la valeur de base si toutes les features valaient zéro.
    * $\beta_1, \beta_2, \dots, \beta_p$ sont les **coefficients** (ou poids) associés à chaque feature. Ils indiquent de combien le prix augmente (ou diminue) lorsque la feature augmente d'une unité.
    * $\epsilon$ représente l'erreur résiduelle

    ### 2. Le point de vue matriciel

    $$Y = X\beta + \epsilon$$

    * $Y$ est un vecteur colonne de dimensions $n \times 1$ contenant les prix de tous les districts.
    * $X$ est la **matrice de design** de dimensions $n \times (p+1)$. Sa première colonne est remplie de $1$ (pour l'intercept $\beta_0$), et les colonnes suivantes correspondent à la matrice `housing_prepared`.
    * $\beta$ est le vecteur colonne des coefficients, de dimensions $(p+1) \times 1$. C'est **l'inconnue à déterminer**.

    ### 3. Les Moindres Carrés Ordinaires (MCO)

    Au lancement de l'instruction `lin_reg.fit(housing_train, housing_labels)`, l'algorithme cherche les valeurs de $\beta$ qui minimisent la somme des erreurs au carré. La fonction de coût (MSE) s'écrit :

    $$MSE(X, \beta) = \frac{1}{n} \|Y - X\beta\|^2$$

    Comme cette fonction est convexe, elle admet un minimum global unique. En annulant sa dérivée par rapport à $\beta$,on obtient une formule mathématique exacte appelée l'**Équation Normale** :

    $$\hat{\beta} = (X^T X)^{-1} X^T Y$$

    C'est la formule théorique. En pratique, inverser explicitement $X^T X$ est coûteux et numériquement instable dès que deux features sont fortement corrélées. À la place, Scikit-Learn trouve le vecteur $\hat{\beta}$ idéal en faisant appel à des bibliothèques d'algèbre linéaire hautement optimisées (LAPACK) qui mobilisent une **décomposition en valeurs singulières (SVD)** de $X$.

    Cette approche calcule le **pseudo-inverse de Moore-Penrose** $X^{+}$, et la solution s'écrit :

    $$\hat{\beta} = X^{+} Y$$

    Le résultat est mathématiquement équivalent à l'équation normale quand $X^T X$ est inversible, mais la SVD reste stable même dans le cas contraire (elle renvoie alors la solution de norme minimale). On retrouvera tous ces détails au chapitre 4.

    On propose ci-dessous une implémentation idiomatique de cette régression.
    """)
    return


@app.cell
def _(housing_labels, housing_train, make_pipeline, preprocessing):
    from sklearn.linear_model import LinearRegression

    # Combine le bloc de préparation des données (preprocessing) et l'algo de régression linéaire dans un objet Pipeline
    lin_reg = make_pipeline(preprocessing, LinearRegression())

    # Lance la préparation des données puis entraîne le modèle de ML 
    lin_reg.fit(housing_train, housing_labels)
    return (lin_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Il faut bien comprendre que `preprocessing` (qui est un `ColumnTransformer`) ne lance aucun traitement tant que sa méthode `fit()` n'est pas appelée. À l'éxécution de la cellule précédente, Scikit-Learn va donc :
    1. Passer `housing_train` dans le preprocessing.
    2. Calculer toutes les transformations et sortir une matrice numérique propre (l'équivalent de `housing_prepared`, plus haut).
    3. Donner cette matrice à la régression linéaire avec les étiquettes de prix (`housing_labels`) pour ajuster ses coefficients mathématiques.

    Voyons alors quelles prédictions notre modèle réalise.
    """)
    return


@app.cell
def _(housing_labels, housing_train, lin_reg):
    # On fait les prédiction à partir du train set
    housing_predictions_lin_reg = lin_reg.predict(housing_train)

    # Prédictions des prix médians pour les 5 premières instances (districts)
    _predictions = housing_predictions_lin_reg [:5].round(-2)

    # Vraies valeurs de ces 5 instances
    _real_values = housing_labels.iloc[:5].values

    # Affichage des ratios d'erreur
    _error_ratios = _predictions / _real_values - 1
    print(", ".join([f"{100 * ratio:.1f}%" for ratio in _error_ratios]))

    # C'est pas horrible, mais la première prédiction est clairement mauvaise
    return (housing_predictions_lin_reg,)


@app.cell
def _(housing_labels, housing_predictions_lin_reg):
    from sklearn.metrics import root_mean_squared_error

    # Calcul et affichage de la mesure d'erreur du modèle : racine de l'erreur quadratique moyenne (RMSE)
    _lin_rmse = root_mean_squared_error(housing_labels, housing_predictions_lin_reg)
    _lin_rmse
    return (root_mean_squared_error,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    C'est assez mauvais puisque le prix médian d'un district est d'environ 180 000, se tromper de ~69 000 en moyenne représente une erreur conséquente. On a donc clairement un phénomène d'**underfitting** des données d'entraînement. On a vu au chapitre précédent des méthodes pour résoudre cela :
    - Choisir un modèle plus puissant (régression polynomiale, arbre de décision, forêt aléatoire, modèle à vecteurs de support, ..)
    - Choisir de meilleures features
    - Réduire les contraintes sur le modèle (inutile ici)

    On décide d'essayer avec un modèle plus puissant.

    ### Arbre de décision

    Essayons avec un modèle d'arbre de décision `DecisionTreeRegressor`.
    """)
    return


@app.cell
def _(housing_labels, housing_train, make_pipeline, preprocessing):
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(housing_train, housing_labels)
    return (tree_reg,)


@app.cell
def _(housing_labels, housing_train, root_mean_squared_error, tree_reg):
    # On fait les prédiction à partir du train set
    housing_predictions_tree_reg  = tree_reg.predict(housing_train)

    # Calcul et affichage de la RMSE
    tree_rmse = root_mean_squared_error(housing_labels, housing_predictions_tree_reg)
    tree_rmse
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On pourrait se dire qu'une training error nulle c'est très bien, mais puisque justement on évalue les prédictions sur le même set qui a entraîné le modèle, il y a une très forte suspicion d'**overfitting**. Une façon de s'en assurer sans évaluer sur le test set est d'utiliser un procédé de **validation croisée**.

    ---
    ## Mieux évaluer avec la validation croisée

    La fonction `cross_val_score` du package Scikit-Learn en est l'implémentation la plus idiomatique. Elle partitionne le train set en k subsets (ou **folds**), ce qui donne par passage au complémentaire k splits train/validation. On entraîne-évalue k fois et on définit la **validation error** comme la moyenne de ces k erreurs.

    ### Préprocessing dans la cross-validation
    Pour chaque split train/validation, on repasse tout le sous-train set dans le preprocessing. Pourquoi ne pas preprocesser une fois pour toutes et découper ensuite ?

    Parce que `StandardScaler` calcule la moyenne et l'écart-type des features, `SimpleImputer` calcule la médiane des colonne pour remplir les valeurs manquantes, etc.. Ce sont toutes des statistiques **globales** du dataset. Si on appliquait le preprocessing à l'ensemble du train set avant de le découper, ces statistiques seraient calculées en incluant déjà les districts qui serviront plus tard de fold de validation.

    En refaisant tourner le preprocessing à l'intérieur de chaque fold, on garantit au contraire que le fold de validation est transformé avec des statistiques apprises sans lui. C'est l'unique manière d'obtenir une mesure honnête de la capacité de **généralisation** du modèle.
    """)
    return


@app.cell
def _(housing_labels, housing_train, pd, tree_reg):
    from sklearn.model_selection import cross_val_score

    # On précise le modèle, les features sources, la feature cible, une mesure d'erreur et un nombre k de splits (cv)
    tree_rmses = -cross_val_score(tree_reg, housing_train, housing_labels, scoring = "neg_root_mean_squared_error", cv=10)

    # Conversion en objet Series (Pandas)
    pd.Series(tree_rmses).describe()
    return (cross_val_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ça nous donne une RMSE moyenne de 67000, avec un écart-type sur l'erreur de ~1500. On avait ~69000 avec le modèle de régression linéaire ; à peine mieux donc. Surtout, une haute validation error pour une faible training error est un signal fort d'overfitting.

    ### Forêt aléatoire

    On ré-essaye avec un autre type de modèle : une forêt aléatoire `RandomForestRegressor`.
    """)
    return


@app.cell
def _(
    cross_val_score,
    housing_labels,
    housing_train,
    make_pipeline,
    pd,
    preprocessing,
):
    from sklearn.ensemble import RandomForestRegressor

    # n_jobs = -1 active le multithreading
    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42, n_jobs=-1))
    forest_rmses = -cross_val_score(forest_reg, housing_train, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

    pd.Series(forest_rmses).describe()
    return RandomForestRegressor, forest_reg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On a maintenant une RMSE moyenne de ~47000, avec un écart-type de ~1100. C'est clairement mieux. Comparons-la à la training error pour avoir une idée de l'overfitting de ce modèle.
    """)
    return


@app.cell
def _(forest_reg, housing_labels, housing_train, root_mean_squared_error):
    # On entraîne le modèle
    forest_reg.fit(housing_train, housing_labels)

    # On fait les prédiction à partir du train set
    housing_predictions_forest_reg = forest_reg.predict(housing_train)

    # Calcul et affichage de la RMSE
    forest_rmse = root_mean_squared_error(housing_labels, housing_predictions_forest_reg)
    forest_rmse

    # 47000 >> 17519 ; on a encore de l'overfitting
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Régression Lasso

    La régression Lasso est une version régularisée de la régression linéaire. Elle s'implémente typiquement avec l'estimateur `Lasso` de Sckikit-Learn.

    NB : `DecisionTreeRegressor`, `RandomForestRegressor` et `Lasso` sont des **estimateurs** puisqu'ils disposent d'une méthode `fit()` (ils apprennent quelque chose sur un ensemble de données, pour _estimer_ autre chose). Mais ce sont plus spécifiquement des **prédicteurs** (puisqu'ils disposent d'une méthode `predict()` !).
    """)
    return


@app.cell
def _(
    cross_val_score,
    housing_labels,
    housing_train,
    make_pipeline,
    pd,
    preprocessing,
    root_mean_squared_error,
):
    from sklearn.linear_model import Lasso

    # Les underscores imposent à Marimo de considérer les variables comme locales à la cellule
    _lasso_reg = make_pipeline(preprocessing, Lasso(alpha=100, max_iter=100000))
    _lasso_reg.fit(housing_train, housing_labels)
    _housing_predictions_lasso_reg = _lasso_reg.predict(housing_train)
    _training_error_lasso_reg = root_mean_squared_error(housing_labels, _housing_predictions_lasso_reg)

    _lasso_rmses = -cross_val_score(_lasso_reg, housing_train, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    _validation_error_lasso_reg = pd.Series(_lasso_rmses).mean()

    print(f"Erreur d'entraînement (RMSE) : {_training_error_lasso_reg:.2f} | Erreur de validation (RMSE moyen) : {_validation_error_lasso_reg:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### K plus proches voisins (KNN)

    Pour prédire le prix d'une maison, on peut aussi faire la moyenne des k maisons les plus "proches" vis à des vis des features sources. C'est ce que fait l'algorithme des k plus proches voisins.
    """)
    return


@app.cell
def _(
    cross_val_score,
    housing_labels,
    housing_train,
    make_pipeline,
    pd,
    preprocessing,
    root_mean_squared_error,
):
    from sklearn.neighbors import KNeighborsRegressor

    # Les underscores imposent à Marimo de considérer les variables comme locales à la cellule
    _knn_reg = make_pipeline(preprocessing, KNeighborsRegressor(n_neighbors=5))
    _knn_reg.fit(housing_train, housing_labels)

    _housing_predictions_knn_reg = _knn_reg.predict(housing_train) 
    _training_error_knn_reg = root_mean_squared_error(housing_labels, _housing_predictions_knn_reg)

    _knn_rmses = -cross_val_score(_knn_reg, housing_train, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    _validation_error_knn_reg = pd.Series(_knn_rmses).mean()

    print(f"Erreur d'entraînement (RMSE) : {_training_error_knn_reg:.2f} | Erreur de validation (RMSE moyen) : {_validation_error_knn_reg:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gradient boosting

    Dans son fonctionnement, le Gradient Boosting (`HistGradientBoostingRegressor` avec Scikit) s'inscrit dans la lignée des arbres et forêts. On essaye pas d'en comprendre davantage pour l'instant (rendez-vous au chapitre 7 du livre pour ça).
    """)
    return


@app.cell
def _(
    cross_val_score,
    housing_labels,
    housing_train,
    make_pipeline,
    pd,
    preprocessing,
    root_mean_squared_error,
):
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Les underscores imposent à Marimo de considérer les variables comme locales à la cellule
    _gb_reg = make_pipeline(preprocessing, HistGradientBoostingRegressor())
    _gb_reg.fit(housing_train, housing_labels)

    _housing_predictions_gb_reg = _gb_reg.predict(housing_train)
    _training_error_gb_reg = root_mean_squared_error(housing_labels, _housing_predictions_gb_reg)

    _gb_rmses = -cross_val_score(_gb_reg, housing_train, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    _validation_error_gb_reg = pd.Series(_gb_rmses).mean()

    print(f"Erreur d'entraînement (RMSE) : {_training_error_gb_reg:.2f} | Erreur de validation (RMSE moyen) : {_validation_error_gb_reg:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tableau récapitulatif

    | Type du modèle | Erreur d'entraînement (RMSE) | Erreur de validation (RMSE moyen) |
    | --- | --- | --- |
    | Régression linéaire multiple | 68 973 | 70 003 |
    | Arbre de décision | 0.0 | 67 013 |
    | Forêt aléatoire | 17 520 | 47 125 |
    | Régression Lasso | 69 157 | 69 941 |
    | K plus proches voisins (KNN) | 48 441 | 60 328 |
    | Gradient boosting | 38 358 | 45 786 |

    Le gradient boosting s'en sort particulièrement bien (RMSE moyen minimal ici, peu d'overfitting). À noter que ces scores sont obtenus avec des valeurs arbitraires des hyperparamètres.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Réglage des hyperparamètres du modèle (fine-tuning)

    ## Recherche par grille (grid search)

    Pour optimiser les performances de notre modèle, une première approche consiste à définir une **plage d'hyperparamètres** et à tester notre modèle pour chaque configuration. En dimension supérieure à 1 (lorsqu'on teste plus d'un paramètre), on parle plus volontiers de grille (grid) que de plage – d'où cette appellation.

    La classe `GridSearchCV` de Scikit implémente cette approche, en évaluant chaque configuration par validation croisée.

    Toutefois, notre pipeline contient une série d'objets imbriqués, ce qui ne rend pas triviale la localisation des hyperparamètres. Pour permettre à `GridSearchCV` de modifier leur valeur, il faut respecter une syntaxe qui ressemble peu ou prou à celle des chemins de fichiers. Par exemple, lorsqu'on écrit `"preprocessing__geo__n_clusters"`, Scikit-Learn coupe la chaîne à chaque double tiret bas (`__`) et descend pas à pas :
    1. `preprocessing` → il cherche dans le pipeline l'étape nommée « preprocessing » et tombe sur `ColumnTransformer`.
    2. `geo` → il cherche *à l'intérieur* de ce `ColumnTransformer` le transformateur nommé « geo » et tombe sur le transformateur de similarité géographique (`ClusterSimilarity`).
    3. `n_clusters` → il accède enfin au réglage `n_clusters` de ce transformateur précis.
    """)
    return


@app.cell
def _(RandomForestRegressor, housing_labels, housing_train, preprocessing):
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    full_pipeline = Pipeline([("preprocessing", preprocessing), ("random_forest", RandomForestRegressor(random_state=42, n_jobs=-1)), ])

    _param_grid = [{'preprocessing__geo__n_clusters': [5, 8, 10], 'random_forest__max_features': [4, 6, 8]},
                  {'preprocessing__geo__n_clusters': [10, 15],   'random_forest__max_features': [6, 8, 10]},]

    grid_search = GridSearchCV(full_pipeline, _param_grid, cv=3, scoring='neg_root_mean_squared_error', refit=True)
    grid_search.fit(housing_train, housing_labels)
    return Pipeline, full_pipeline, grid_search


@app.cell
def _(grid_search):
    grid_search.best_params_

    # L'optimum de `n_clusters` tombe à 15, or c'est la valeur max de la plage : l'optimum est donc probablement au-delà
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Avec l'option par défaut `refit=True`, une fois que GridSearchCV a identifié la meilleure configuration des hyperparamètres, il réentraîne le modèle sur la totalité du train set avec cette configuration optimale. Ça évite d'avoir en sortie un modèle entraîné sur une fraction du train set à cause de la validation croisée. Le prédicteur final est alors accessible via `grid_search.best_estimator_`.

    On accède aux résultats du fine-tuning grid search avec `grid_search.cv_results_` : l'erreur de validation (RMSE moyen) descend à 44 079.
    """)
    return


@app.cell
def _(grid_search, pd):
    _cv_res = pd.DataFrame(grid_search.cv_results_)
    _cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
    _cv_res.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Recherche aléatoire (randomized search)

    L'inconvénient majeur de fine-tuner un modèle avec une approche grid search est clairement le coût de calcul : les quelques 12 valeurs qu'on a entré ont déjà nécessité 45 entraînements du modèle.

    La **randomized search** est une stratégie alternative : au lieu de tester exhaustivement les points d'une grille, elle en pioche aléatoirement un **nombre fixé** dans des plages de valeurs données. Ce n'est pas aussi bien qu'une descente de gradient dans l'espace des hyperparamètres (quasi-impossible à mettre en oeuvre), mais en pratique ça permet d'explorer l'espace aussi bien que l'approche grid search, pour un coût de calcul **réduit** et **contrôlé**.

    La classe `RandomizedSearchCV` implémente cette approche et permet de guider le parcours aléatoire avec une **distribution**. Ainsi, avec `randint(low=3, high=50)` on tire au hasard selon la loi uniforme des valeurs entières des hyperparamètres, comprises entre 3 et 49.
    """)
    return


@app.cell
def _(full_pipeline, housing_labels, housing_train):
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50), 
                      'random_forest__max_features': randint(low=2, high=20)}

    rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)

    rnd_search.fit(housing_train, housing_labels)

    # Affichage de l'erreur de validation (on avait 44 079 avec la grid search)
    -rnd_search.best_score_
    return RandomizedSearchCV, param_distribs, rnd_search


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Successive halving
    Les classes `HalvingRandomSearchCV` et `HalvingGridSearchCV` font sensiblement la même chose que les classes précédentes, mais introduisent des optimisations de performance.

    Jusqu'ici, chaque combinaison candidate subissait le même entraînement complet en validation croisée. Au lieu de cela, le **successive halving** propose d'allouer de façon adaptative les _ressources_ sous forme de **tournoi**.

    On procède par **manches successives** : on évalue le modèle sous les différentes configurations (issues d'une grille ou de distributions, selon la classe Halving* choisie), puis on compare les taux d'erreurs. À chaque manche, on sélectionne les meilleures configurations et on **augmente les ressources** allouées.

    La notion de « ressource » est volontairement abstraite : ça peut être la **taille de l'échantillon** d'entraînement, le **nombre d'itérations** d'un modèle, le **nombre d'arbres** d'une forêt, etc. L'intuition sous-jacente est qu'un mauvais candidat se révèle généralement mauvais même avec peu de ressources.

    ---

    ## Méthodes d'ensemble

    Un autre moyen très simple d'améliorer les prédictions consiste à **combiner des modèles**, par exemple en considérant une combinaison convexe d'un KNN et d'un modèle de gradient boosting. On trouvera plus de détail sur ce sujet au Chapitre 7.

    ---

    ## Sélection de variables (feature selection)

    Un bon moyen de comprendre la structure sous-jacente des données consiste à faire du reverse-engineering sur un modèle performant. Par exemple, la classe `RandomForestRegressor` dispose de l'attribut `feature_importances_` qui indique l'importance relative des différentes features pour réaliser des prédictions précises.
    """)
    return


@app.cell
def _(rnd_search):
    final_model = rnd_search.best_estimator_
    _feature_importances = final_model["random_forest"].feature_importances_
    sorted(zip(_feature_importances, final_model["preprocessing"].get_feature_names_out()), reverse=True)
    return (final_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut envisager de supprimer certaines des features les moins utiles (par exemple, seule une catégorie de `ocean_proximity` semble véritablement utile, on pourrait essayer de retirer les autres).

    C'est ce qu'automatise la classe `SelectFromModel` de Scikit. Elle s'insère en dernière étape du preprocessing : elle entraîne un modèle simple (pas fine-tuné), consulte son attribut `feature_importances_` et supprime les features les moins importantes. Ensuite viennent l'entraînement du vrai modèle et son fine-tuning.

    On utilise souvent une **forêt aléatoire** pour le modèle de « feature selection » parce que son attribut `feature_importances_` est plutôt fiable et qu'il a la particularité d'être un sous-produit de son entraînement.

    On décide de mesurer les performances de notre modèle en intégrant cet « élagage » des features. Deux cas de figure possibles :
    - Soit ça améliore la RMSE (auquel cas la sélection a effectivement retiré du **bruit**)
    - Soit ça la dégrade légèrement (auquel cas certaines features faiblement notées contribuaient quand même un peu)
    """)
    return


@app.cell
def _(
    Pipeline,
    RandomForestRegressor,
    RandomizedSearchCV,
    housing_labels,
    housing_train,
    param_distribs,
    preprocessing,
    rnd_search,
):
    from sklearn.feature_selection import SelectFromModel

    _full_pipeline_with_selection = Pipeline([
        ("preprocessing", preprocessing),
        ("feature_selection", SelectFromModel(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            threshold="median"
        )),
        ("random_forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    _rnd_search_2 = RandomizedSearchCV(_full_pipeline_with_selection, param_distributions=param_distribs, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)

    _rnd_search_2.fit(housing_train, housing_labels)

    print(f"Erreur de validation sans élagage des features : {-rnd_search.best_score_:,.2f}\n"
        f"Erreur de validation avec élagage des features : {-_rnd_search_2.best_score_:,.2f}\n"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Évaluation sur le test set

    On décide finalement d'évaluer notre modèle (celui sans élagage des features) sur le test set.
    """)
    return


@app.cell
def _(final_model, root_mean_squared_error, strat_test_set):
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    final_predictions = final_model.predict(X_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    print(final_rmse)
    return final_predictions, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bootstrap

    Pour déterminer un **intervalle de confiance** à 95% sur l'erreur, on peut utiliser une méthode **boostrap**. Ça a l'inconvénient d'un coût de calcul non négligeable, mais ça donne un IC assez fiable.

    Puisque la RMSE est la racine de la moyenne des erreurs de prédiction au carré, l'idée du bootstrap consiste à réaliser dans le test set n tirages **avec remise** de ces erreurs au carré, puis d'en calculer la moyenne (et d'en prendre la racine carrée). En répétant ce processus on parvient à estimer la **distribution d'échantillonnage** de la RMSE et à calculer l'intervalle de confiance associé.

    Cela s'implémente naturellement avec la méthode `boostrap()` de la classe `stats` de scipy.
    """)
    return


@app.cell
def _(final_predictions, np, y_test):
    from scipy import stats

    def rmse(squared_errors):
        return np.sqrt(np.mean(squared_errors))

    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    boot_result = stats.bootstrap([squared_errors], rmse, confidence_level=confidence, random_state=42)
    rmse_lower, rmse_upper = boot_result.confidence_interval
    rmse_lower, rmse_upper
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sauvegarder et déployer le modèle en production

    On présente un moyen simple (package `joblib`) de sauvegarder un modèle.

    ```
    import joblib

    joblib.dump(final_model, "my_california_housing_model.pkl")
    ```

    On peut ensuite déployer le modèle en production plus tard pour effectuer des prédictions.

    ```
    # import joblib
    # from sklearn.cluster import KMeans
    # from sklearn.base import BaseEstimator, TransformerMixin
    # from sklearn.metrics.pairwise import rbf_kernel

    final_model_reloaded = joblib.load("my_california_housing_model.pkl")
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
