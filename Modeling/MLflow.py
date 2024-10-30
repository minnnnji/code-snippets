import mlflow
import os
import pandas as pd
import numpy as np
import re

from sklearn.tree import plot_tree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import *
import statsmodels as sm

import matplotlib.pyplot as plt
import seaborn as sns

def get_next_run_name(experiment_name, base_name='model'):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    try: 
        max_number = 0
        for run_name in runs['tags.mlflow.runName']:
            match = re.match(rf'{base_name}_(\d+)', run_name)
            if match:
                max_number = max(max_number, int(match.group(1)))
        return f"{base_name}_{max_number + 1}"
    except:
        return f"{base_name}_1"

