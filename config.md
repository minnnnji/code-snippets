## Basic setting or Module Import

```py

# 기본 
import os
import time
import pythoncom
import pandas as pd
import numpy as np
from glob import glob
import win32com.client
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 그래프
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib

import plotly.express as px
import cufflinks as cf

sns.set_style('whitegrid')
sns.set_palette('pastel')
plt.style.use('default')
plt.rc('font', family='Malgun Gothic')

matplotlib.rcParams["axes.unicode_minus"] = False
cf.go_offline()

# 모델링
from sklearn.tree import plot_tree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import statsmodels.api as sm
```