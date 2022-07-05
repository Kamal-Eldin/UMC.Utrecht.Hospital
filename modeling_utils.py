import matplotlib.pyplot as plt
import pyximport
import pandas as pd
import numpy as np
import gzip
import pickle

import xgboost as xgb
import sklearn
from numpy.random import Generator, PCG64

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV, RepeatedStratifiedKFold, train_test_split, RepeatedKFold, StratifiedKFold, cross_validate, ShuffleSplit, KFold
from scipy import stats


class dev_xgb:
    """Instantiates the modeling sequence.
    
    """
    def __init__(self, init_params = None,*, inner_cv= None, outer_cv= None, search_space= None, seed= 606, n_jobs= 8):
        """Intializes the modeling object with a set of initial parameters, inner and outer cross-validation sklearn objects, a random search space and a seed for defining all random states.
        
        The function uses `init_params` to instantiate a base XGBClassifier object (self.clf) which would undergo the cross-val procedures.
        The function instantiates a random search space with scipy distributions for a selected subset of xgb parameters.
        
        Parameters
        ----------
        init_params : dict
            Dictionary of initial parameters to define a basic xgboost classifier (default None).
        inner_cv : sklearn cross-validation splitter object
            sklearn CV splitter object defining the CV inner loop for model hyper-parameter search in a nested cross-val procedure (default None).
        outer_cv : sklearn cross-validation splitter object
            sklearn CV splitter object defining the CV outer loop for model error estimation in a nested cross-val procedure (default None).
        search_space : dict
            Dictionary of hyper parameters defining the search space for the random search (default None; no search).
        seed : int
            Define the Random State for sklearn CV procedures (default 606).
        n_jobs : int
            Defines the number of cpu workers to use in model training and cross validation (default 8).
            
        """
        self.outer_cv= outer_cv
        self.inner_cv= inner_cv
        self.seed= seed
        self.randomizer = Generator(PCG64(seed))
        self.n_jobs= n_jobs
        
        if not init_params:
            self.init_params= {'objective': 'binary:logistic', 'learning_rate': 0.3, 'gamma': 1, 'max_depth': 6,
                                  'subsample': 0.5, 'colsample': 0.5, 'reg_lambda': 1, 'early_stopping': 5,
                                  'n_estimators': 10, 'eval_metric':['auc','aucpr']}
        else:
            self.init_params= init_params

        
        if not search_space:
            norm= stats.norm (loc= 0.32, scale= .1)
            regularizer= stats.expon(loc= 3, scale= 5)
            tree_depth= stats.nbinom(8, .6)
            cover= stats.nbinom(10, .6)
            self.search_space= {'learning_rate': stats.loguniform(.0001, .5),
                                 'max_depth': tree_depth,
                                 'subsample': norm,
                                 'colsample_bytree': norm,  
                                 'colsample_bysplit': norm,
                                 'gamma': regularizer,
                                 'reg_lambda': regularizer,
                                 'reg_alpha': regularizer, 
                                 # 'early_stopping': regularizer,
                                 'min_child_weight': cover}

        else:
            self.search_space= search_space

        self.clf= xgb.XGBClassifier(**self.init_params, verbosity= 1, n_jobs= self.n_jobs, random_state = self.seed)
    
    def cross_val(self, X, y, n_candids= 10, halving= True):
        """Defines and executes the cross-val procedure. Returns DataFrame with errors/scores per cv iteration and its estimator.
        
        Cross-val metrics are defined as the area under ROC curve (AUCROC) and the area under precision-recall curve AUCPR (ie., average precision).
        
        Parameters
        ----------
        X : array
            Input feature vector array.
        y : array
            Input labels array.
        n_candids : int
            Defines the number of candidate models to include in the Random Search.
        halving : bool
            If True (default), an sklearn successive halving random search object is used. If False, a RandomizedSearchCV object is defined and used.
            
        Returns
        -------
        DataFrame
            Df with cross-val scores per CV iteration and its estimators.
        """
        np.random.seed(seed= self.seed)
        if halving:
            rand_search= HalvingRandomSearchCV(self.clf, self.search_space, 
                                                random_state= self.seed, 
                                                cv= self.inner_cv,
                                                resource= 'n_samples',
                                                factor= 2,
                                                n_candidates= n_candids,
                                                min_resources= 'exhaust',
                                                scoring= 'average_precision',
                                                n_jobs= self.n_jobs,
                                                verbose= 1) 
        else:
            rand_search= RandomizedSearchCV(self.clf, self.search_space, 
                                            random_state= self.seed,
                                            n_iter= n_candids,
                                            cv= self.inner_cv,    
                                            scoring= 'average_precision',
                                            n_jobs= self.n_jobs,
                                            return_train_score= True)
            
            
        error_estimation= cross_validate(rand_search, X, y, cv= self.outer_cv,
                                          scoring= ['roc_auc', 'average_precision'], 
                                          return_estimator= True, 
                                          return_train_score= True,
                                          n_jobs= self.n_jobs,
                                          verbose= 2)
        
        cv_results= pd.DataFrame(error_estimation).sort_values(by= 'test_average_precision', ascending= False)
        return cv_results
    
    