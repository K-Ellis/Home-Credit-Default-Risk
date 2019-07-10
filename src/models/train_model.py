import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Imputer, OneHotEncoder, PolynomialFeatures, OrdinalEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from time import time
from sys import argv

# get location of training data
training_data_loc = argv[1]

# import data
app_train = pd.read_csv(training_data_loc)
assert len(app_train) > 1, "training data empty"

