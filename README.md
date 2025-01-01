# FIPE: Functionally Identical Pruning of Ensembles

[![PyPI](https://img.shields.io/pypi/v/fipepy
)](https://pypi.org/project/fipepy/)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/fipepy.svg)](https://pypi.org/project/fipepy/)
![test](https://github.com/eminyous/fipe/actions/workflows/main.yml/badge.svg)

This repository provides methods for Functionally-Identical Pruning of Tree Ensembles (FIPE). Given a trained scikit-learn model, FIPE provides a pruned model that is certified to be equivalent to the original model on the entire feature space. The algorithm is described in detail in the paper: <https://arxiv.org/abs/2408.16167> .

## Installation

This project requires the gurobi solver. Free academic licenses are available. Please consult:

- [Gurobi academic program and licenses](https://www.gurobi.com/academia/academic-program-and-licenses/)
- [Gurobi academic license agreement](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)

Run the following commands from the project root to install the requirements. You may have to install python and venv before.

```shell
virtualenv -p python3.10 env
pip install fipepy
```

The installation can be checked by running the test suite:

```shell
pip install pytest
pytest
```

The integration tests require a working Gurobi license. If a license is not available, the tests will pass and print a warning.

### Getting started

A minimal working example to prune an AdaBoost ensemble is presented below.

```python
import gurobipy as gp
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from fipe import FIPE, FeatureEncoder

# Load data encode features
data = load_iris(as_frame=True)
X = pd.DataFrame(data.data)
y = data.target

encoder = FeatureEncoder(X)
X = encoder.X.to_numpy()

# Train tree ensemble
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
base = AdaBoostClassifier(n_estimators=100, random_state=42)
base.fit(X, y)

# Read and normalize weights
w = base.estimator_weights_
w = (w / w.max()) * 1e5

# Prune using FIPE
norm = 1
print(f"Pruning model by minimizing l_{norm} norm.")
env = gp.Env()
env.setParam("OutputFlag", 0)
pruner = FIPE(
    base=base,
    encoder=encoder,
    weights=w,
    norm=norm,
    env=env,
    eps=1e-6,
    tol=1e-4,
)
print("Building pruner...")
pruner.build()
pruner.add_samples(X_train)
print("Pruning...")
pruner.prune()
print("Finished pruning.")

# Read pruned model
n_active_estimators = pruner.n_active_estimators
print(
    f"The pruned ensemble has {n_active_estimators}"
    f"/{base.n_estimators} active estimators."
)

# Verify functionally-identical on test data
y_pred = base.predict(X_test)
y_pruned = pruner.predict(X_test)
fidelity = np.mean(y_pred == y_pruned)
print(f"Fidelity to initial ensemble is {fidelity * 100:.2f}%.")
```
