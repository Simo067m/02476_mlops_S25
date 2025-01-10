# mlops_grp5

January 2025 version of 02476 Machine Learning Operations.

Group 5:
Katharina Strauss Søgaard (s214634), 
Cecilie Dahl Hvilsted (s214605),
Simon Stenbæk Jensen (s214592)

# Project description
**Freshness classification of fruits and vegetables via image recognition**

### Goal of project

The goal of this project is to create a model that can classify fruits and vegetables into fresh or rotten. This helps the user determine whether their food is edible or unedible, and thus helps to reduce food waste. <br>
The classification will be done on images, so an end user can upload images of their own fruits and vegetables to be told the freshness.

### Framework

We will use Pytorch and in extension of this, Pytorch Lightning, to take advantage of their train and eval functions. <br>
We will also be using sklearn for model evaluation functions.

### Data

The data consists of image data of fresh and rotten vegetables from a Kaggle dataset:
https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset <br>
(size = 920 MB, 12000 images). <br>
The dataset holds images of 5 different fruits and 5 different vegetables, where each has a fresh and rotten image pair.

### Models

We will use one (or more) of the pre-trained models from Pytorch-image-models, and fine-tune it on our dataset. An example of this is creating sweeps at Weights and Biases. Lastly, we will focus on choosing low size, high speed models which can have a fast inference rate and can run in the cloud.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).