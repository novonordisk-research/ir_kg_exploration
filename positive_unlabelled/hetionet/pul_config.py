import pandas as pd
import numpy as np
from pulearn.elkanoto import ElkanotoPuClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

class PULConfig:
    @classmethod
    def build_pipeline(cls):
        pass
    @classmethod
    def build_param_grid(cls):
        pass

class ElkanotoConfig(PULConfig):
    @classmethod
    def build_pipeline(cls):
        estimator = SVC(kernel='rbf', gamma=0.2, probability=True)
        pipe = Pipeline([
            ('pca', PCA()),
            ('elkanoto', ElkanotoPuClassifier(estimator=estimator))
        ])
        return pipe

    @classmethod
    def build_param_grid(cls):
        param_grid = {
            'pca__n_components': [1,2,3,4,5,6,7,8,9,10],
            'elkanoto__hold_out_ratio': [0.2, 0.3],
            'elkanoto__estimator': [
                SVC(kernel='rbf', gamma=0.1, probability=True),
                SVC(kernel='rbf', gamma=0.2, probability=True),
                SVC(kernel='rbf', gamma=0.3, probability=True),
                SVC(kernel='rbf', gamma=0.4, probability=True),
                SVC(kernel='rbf', gamma=0.5, probability=True),
            ]
        }
        return param_grid

class ElkanotoXGBoostConfig(PULConfig):
    @classmethod
    def build_pipeline(cls):
        estimator = XGBClassifier()
        pipe = Pipeline([
            ('pca', PCA()),
            ('elkanoto', ElkanotoPuClassifier(estimator=estimator))
        ])
        return pipe

    @classmethod
    def build_param_grid(cls):
        param_grid = {
            'pca__n_components': [1,2,3,4,5,6,7,8,9,10],
            'elkanoto__hold_out_ratio': [0.2, 0.3],
            'elkanoto__estimator': [
                XGBClassifier(n_estimators=2, max_leaves=0),
                XGBClassifier(n_estimators=5, max_leaves=0),
                XGBClassifier(n_estimators=10, max_leaves=0),
                XGBClassifier(n_estimators=20, max_leaves=0),
                XGBClassifier(n_estimators=2, max_depth=1, max_leaves=0),
                XGBClassifier(n_estimators=5, max_depth=1, max_leaves=0),
                XGBClassifier(n_estimators=10, max_depth=1, max_leaves=0),
                XGBClassifier(n_estimators=20, max_depth=1, max_leaves=0),
                XGBClassifier(n_estimators=2, max_leaves=10),
                XGBClassifier(n_estimators=5, max_leaves=10),
                XGBClassifier(n_estimators=10, max_leaves=10),
                XGBClassifier(n_estimators=20, max_leaves=10),
                XGBClassifier(n_estimators=2, max_depth=1, max_leaves=10),
                XGBClassifier(n_estimators=5, max_depth=1, max_leaves=10),
                XGBClassifier(n_estimators=10, max_depth=1, max_leaves=10),
                XGBClassifier(n_estimators=20, max_depth=1, max_leaves=10),
            ]
        }
        return param_grid
    
class LocalOutlierFactorConfig(PULConfig):
    @classmethod
    def build_pipeline(cls):
        pipe = ImblearnPipeline([
            ('pca', PCA()),
            ('random_over_sampler', RandomOverSampler()),
            ('local_outlier_factor', LocalOutlierFactor(novelty=True))
        ])
        return pipe

    @classmethod
    def build_param_grid(cls):
        param_grid = {
            'pca__n_components': [1,2,3,4,5,6,7,8,9,10],
            'random_over_sampler__sampling_strategy': [1.0],
            'local_outlier_factor__n_neighbors': [2, 5, 10, 20, 40, 80],
            'local_outlier_factor__p': [1.0, 1.5, 2.0],
            'local_outlier_factor__contamination': [0.5]
        }
        return param_grid

class IsolationForestConfig(PULConfig):
    @classmethod
    def build_pipeline(cls):
        pipe = ImblearnPipeline([
            ('pca', PCA()),
            ('random_over_sampler', RandomOverSampler()),
            ('isolation_forest', IsolationForest())
        ])
        return pipe

    @classmethod
    def build_param_grid(cls):
        param_grid = {
            'pca__n_components': [1,2,3,4,5,6,7,8,9,10],
            'random_over_sampler__sampling_strategy': [1.0],
            'isolation_forest__n_estimators': [100, 250, 500],
            'isolation_forest__max_samples': ['auto', 0.2, 0.5, 0.75],
            'isolation_forest__max_features': [1.0],
            'isolation_forest__contamination': [0.5]
        }
        return param_grid

class OneClassSVMConfig(PULConfig):
    @classmethod
    def build_pipeline(cls):
        pipe = ImblearnPipeline([
            ('pca', PCA()),
            ('random_over_sampler', RandomOverSampler()),
            ('one_class_svm', OneClassSVM())
        ])
        return pipe

    @classmethod
    def build_param_grid(cls):
        param_grid = {
            'pca__n_components': [1,2,3,4,5,6,7,8,9,10],
            'random_over_sampler__sampling_strategy': [1.0],
            'one_class_svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #'one_class_svm__degree': [3,4,5],
            #'one_class_svm__gamma': ['scale',0.1,0.3,0.5],
        }
        return param_grid

class SVCConfig(PULConfig):
    @classmethod
    def build_pipeline(cls):
        pipe = ImblearnPipeline([
            ('pca', PCA()),
            ('random_over_sampler', RandomOverSampler()),
            ('svc', SVC(kernel='rbf', gamma=0.2, probability=True))
        ])
        return pipe

    @classmethod
    def build_param_grid(cls):
        param_grid = {
            #'pca__n_components': [1,2,3,4,5,6,7,8,9,10],
            'pca__n_components': [5,],
            'random_over_sampler__sampling_strategy': [1.0],
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #'svc__gamma': ['scale',0.1,0.3,0.5],
        }
        return param_grid
