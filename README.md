# Hands-On ML - adaptation française enrichie 

Adaptation française et enrichie des notebooks de *Hands-On Machine Learning with
Scikit-Learn and PyTorch* d'Aurélien Géron. Le code est adapté de
[ageron/handson-mlp](https://github.com/ageron/handson-mlp) (licence Apache 2.0).

J'y ajoute des sections originales, des démonstrations que le livre laisse de côté et des visualisations
interactives. **Sur un chapitre comme le 4, une bonne moitié du contenu rédigé est originale.**

Notebooks écrits avec Marimo : stockés en fichiers Python, réactifs (les cellules dépendantes se réexécutent), et très jolis.

## Chapitres

| Chapitre | Sujet | Version web | Notebook interactif |
|---|---|---|---|
| 2 | End-to-End ML Project (California Housing) | [Notebook statique](https://tizio-faret.github.io/handson-ml-fr/chapitre_02.html) | [Molab](https://molab.marimo.io/github/tizio-faret/handson-ml-fr/blob/main/notebooks/chapitre_02.py/server) |
| 3 | Classification (MNIST) | [Notebook statique](https://tizio-faret.github.io/handson-ml-fr/chapitre_03.html) | [Molab](https://molab.marimo.io/github/tizio-faret/handson-ml-fr/blob/main/notebooks/chapitre_03.py/server) |
| 4 | Training models | [Notebook statique](https://tizio-faret.github.io/handson-ml-fr/chapitre_04.html) | [Molab](https://molab.marimo.io/github/tizio-faret/handson-ml-fr/blob/main/notebooks/chapitre_04.py/server) |

Les chapitres suivants seront ajoutés au fil du temps.

## Lancer en local 

L'entraînement des modèles abordés dans ces chapitres requiert une puissance de calcul importante : il n'est pas rare que l'exécution de certaines cellules 
prenne plusieurs minutes.

```bash
git clone https://github.com/tizio-faret/handson-ml-fr.git
cd handson-ml-fr
python -m venv .venv && source .venv/bin/activate # Remplacer python par python3 en cas d'échec
pip install -r requirements.txt 
marimo edit notebooks/chapitre_04.py # À adapter selon le chapitre
```

## Licence

Code adapté de handson-mlp (Apache 2.0, voir [`LICENSE`](LICENSE)).




