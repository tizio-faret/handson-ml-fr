# Hands-On ML — portage français enrichi

Adaptation française et enrichie des notebooks de *Hands-On Machine Learning with
Scikit-Learn, Keras & TensorFlow* (3ᵉ éd.) d'Aurélien Géron. Le code est adapté de
[ageron/handson-ml3](https://github.com/ageron/handson-ml3) (licence Apache 2.0).

Notebooks écrits avec Marimo : réactifs, stockés en fichiers texte et exécutables dans le navigateur via molab.

## Chapitres

| Chapitre | Sujet | Ouvrir |
|---|---|---|
| 2 | End-to-End ML Project (California Housing) | [Notebook (molab)](https://molab.marimo.io/github/tizio-faret/handson-ml-fr/blob/main/notebooks/chapitre_02.py/wasm) |

Les chapitres suivants seront ajoutés au fil du temps.

## Lancer en local (recommandé)

L'entraînement des modèles abordés dans ces chapitres requiert une puissance de calcul importante. L'environnement web ne supportant pas le multi-threading, 
je recommande vivement d'exécuter ces notebooks en local sur votre propre machine.

```bash
git clone https://github.com/tizio-faret/handson-ml-fr.git
cd handson-ml-fr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt # Installe marimo et des packages Python
marimo edit notebooks/chapitre_02.py # À adapter selon le chapitre
```

## Licence

Code adapté de handson-ml3 (Apache 2.0, voir [`LICENSE`](LICENSE)).


