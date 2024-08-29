# FIPE: Functionally Identical Pruning of Ensembles

![Tests badge](https://github.com/eminyous/fipe/actions/workflows/main.yml/badge.svg?branch=main)

This repository provides methods for Functionally-Identical Pruning of Tree Ensembles (FIPE). Given a trained scikit-learn model, FIPE provides a pruned model that is certified to be equivalent to the original model on the entire feature space.

## Installation

This project requires the gurobi solver. Free academic licenses are available. Please consult:

- [Gurobi academic program and licenses](https://www.gurobi.com/academia/academic-program-and-licenses/)
- [Gurobi academic license agreement](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)

Run the following commands from the project root to install the requirements. You may have to install python and venv before.

```shell
    virtualenv -p python3.10 env
    pip install -e .
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
    from fipe import FIPE, FeatureEncoder
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier


    # Load data encode features
    data = load_iris()
    X = pd.DataFrame(data.data)
    y = data.target

    encoder = FeatureEncoder(X)
    X = encoder.X.values

    # Train tree ensemble
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    base = AdaBoostClassifier(algorithm="SAMME", n_estimators=100)
    base.fit(X, y)

    # Read and normalize weights
    w = base.estimator_weights_
    w = (w / w.max()) * 1e5

    # Prune using FIPE
    norm = 1
    print(f'Pruning model by minimizing l_{norm} norm.')
    pruner = FIPE(base=base, weights=w, encoder=encoder, eps=1e-6)
    pruner.build()
    pruner.set_norm(norm)
    pruner.add_samples(X_train)
    pruner.oracle.setParam('LogToConsole', 0)
    pruner.prune()
    print('\n Finished pruning.')

    # Read pruned model
    n_activated = pruner.n_activated
    print('The pruned ensemble has ', n_activated, ' estimators.')

    # Verify functionally-identical on test data
    y_pred = base.predict(X_test)
    y_pruned = pruner.predict(X_test)
    fidelity = np.mean(y_pred == y_pruned)
    print('Fidelity to initial ensemble is ', fidelity, '%.')
```
