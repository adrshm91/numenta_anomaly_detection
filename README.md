# mlops-platform-evaluation

This repo makes it easy to evaluate various mlops-platforms available in the market by using a baseline anomaly detection project.

In order to evaluate the candidates for our Unified Platform for Analytics at LDA, we need a suitable use case and a suitable dataset. Since most of our use cases are related to Predictive Maintenance using Time series data, we could use [Numenta Anomaly Benchmark](https://www.numenta.com/resources/htm/numenta-anomoly-benchmark/) for our evaluation. This can help to define Personas and clear goals/responsibilities for each of these Personas. Understanding the need of such a platform and knowing what problems we need to solve is very important for the evaluation.

## Getting started

The repo contains various branches based on the mlops platform that is used. For example, the branch [0_no_mlops](https://code.siemens.com/sidrive-iq/teams/team-gov/analytics/mlops-platform-evaluation/-/tree/0_no_mlops) is not using any mlops platform and serves as a baseline/benchmark to compare mlops usage against various other platforms.

## Objective

As a data scientist, I would like to perform two workflows

1. Load data, preprocess data, train a model and save a model
2. Perform retraining of the model by changing model parameters

The objective is to note down the difficulties/pain points experienced as a data scientist when performing the above workloads. And to check if these pain points are addressed with mlops tools.

## Installation

Python version: **Python 3.10.0**

You can use pyenv and venv to create a virtual environment of this python version. For instructions refer this [link](https://sidriveiq.opscenter.siemens.cloud/wiki/pages/viewpage.action?pageId=161349727)

### Create a virtual environment

```
python -m venv .venv
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
