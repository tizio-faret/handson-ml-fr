# Hands-On ML — portage français enrichi

Adaptation française et enrichie des notebooks de *Hands-On Machine Learning with
Scikit-Learn and PyTorch* d'Aurélien Géron. Le code est adapté de
[ageron/handson-mlp](https://github.com/ageron/handson-mlp) (licence Apache 2.0).

Notebooks écrits avec Marimo : stockés en fichiers Python, réactifs (les cellules dépendantes se réexécutent), et très jolis.

## Chapitres

| Chapitre | Sujet | Ouvrir | Molab |
|---|---|---|---|
| 2 | End-to-End ML Project (California Housing) | [Notebook en ligne](https://tizio-faret.github.io/handson-ml-fr/chapitre_02.html) | |
| 3 | Classification | [Notebook en ligne](https://tizio-faret.github.io/handson-ml-fr/chapitre_03.html) | |
| 4 | Training models | [Notebook en ligne](https://tizio-faret.github.io/handson-ml-fr/chapitre_04.html) | [Notebook interactif](https://molab.marimo.io/github/tizio-faret/handson-ml-fr/blob/main/notebooks/chapitre_04.py/wasm) |

Les chapitres suivants seront ajoutés au fil du temps.

## Lancer en local 

L'entraînement des modèles abordés dans ces chapitres requiert une puissance de calcul importante : il n'est pas rare que l'exécution de certaines cellules 
prenne plusieurs minutes.

```bash
git clone https://github.com/tizio-faret/handson-ml-fr.git
cd handson-ml-fr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt # Installe marimo et des packages Python
marimo edit notebooks/chapitre_03.py # À adapter selon le chapitre
```

## Licence

Code adapté de handson-mlp (Apache 2.0, voir [`LICENSE`](LICENSE)).




