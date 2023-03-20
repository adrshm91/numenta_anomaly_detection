# mlops-platform-evaluation

This repo makes it easy to evaluate various mlops-platforms available in the market by using a baseline anomaly detection project.

In order to evaluate the candidates for our Unified Platform for Analytics at LDA, we need a suitable use case and a suitable dataset. Since most of our use cases are related to Predictive Maintenance using Time series data, we could use [Numenta Anomaly Benchmark](https://www.numenta.com/resources/htm/numenta-anomoly-benchmark/) for our evaluation. This can help to define Personas and clear goals/responsibilities for each of these Personas. Understanding the need of such a platform and knowing what problems we need to solve is very important for the evaluation.

## Getting started

The repo contains various branches based on the mlops platform that is used. For example, the branch [0_no_mlops](https://code.siemens.com/sidrive-iq/teams/team-gov/analytics/mlops-platform-evaluation/-/tree/0_no_mlops) is not using any mlops platform and serves as a baseline/benchmark to compare mlops usage against various other platforms.

## Objective

As a data scientist, I would like to perform two workflows

1. Load data, preprocess data, train a model and save a model
2. Try different combinations of hyperparameters, model architectures in order to generate the best model.
3. Perform retraining of the best model by changing input data and model parameters

The objective is to note down the difficulties/pain points experienced as a data scientist when performing the above workloads. And to check if these pain points are addressed with mlops tools.

### Pain points during model creation

1. **Experiment Management**

    - Every change of input data feature, change of hyperparameters, change of code corresponds to an experiment. Without mlops, the experiments are tracked by creating/maintaining folder_names/file_names which can easily be unmanagable

2. **Reproducibility**

    - Without proper tracking, it is difficult to reproduce an experiment that was created using a specific dataset, with specific code, with specific environment, with specific hyperparameters and that generated specific results/artifacts

3. **Orchestration**

    - Often times, we have different scripts that are used for preprocess, training, evaluation, etc. In order to orchestrate these steps we need ml-pipelines.

4. **Data Versioning and Model Registry**

    - We don't want to store data and model artifacts in the git repository which is meant for code versioning. Instead we would like to version the data and the model in a separate versioning system

5. **Transfer execution to a powerful machine**

    - We don't want to train locally in our laptop when we have a huge dataset. Without mlops it is difficult to replicate the environment in a new machine and run training there.


## Installation

Python version: **Python 3.10.0**

You can use pyenv and venv to create a virtual environment of this python version. For instructions refer this [link](https://sidriveiq.opscenter.siemens.cloud/wiki/pages/viewpage.action?pageId=161349727)

### Create a virtual environment

```
python -m venv .venv
.venv/Scripts/activate
```

### Install the requirements

```
pip install -r requirements.txt
```

## Generate the model

```
python src/01_preprocess_data.py
python src/02_isolation_forest.py
```
