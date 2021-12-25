import os
import copy
import math
import random
import pickle
import numpy as np
import sys
import lightgbm as lgb
import optuna.integration.lightgbm as test_lgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from chainer import serializers
import xgboost as xgb
import pandas as pd

import sekitoba_library as lib
import sekitoba_data_manage as dm

check_year = 2020

def lg_main( data, params = None ):
    max_pos = np.max( np.array( data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )


    if params == None:
        lgbm_params =  {
            'task': 'train',                # 学習、トレーニング ⇔　予測predict
            'boosting_type': 'gbdt',        # 勾配ブースティング
            'objective': 'regression',      # 目的関数：回帰
            'metric': 'rmse',               # 回帰分析モデルの性能を測る指標
            #'learning_rate': 0.1,
            'early_stopping_rounds': 30
        }
    else:
        lgbm_params = params

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )
    
    #print( df_importance.head( len( x_list ) ) )
    dm.pickle_upload( "last_horce_body_lightbgm_model.pickle", bst )
    lib.log.write_lightbgm( bst )
    return bst

def lgb_test( data ):
    max_pos = np.max( np.array( data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )

    lgbm_params =  {
        'task': 'train',                # 学習、トレーニング ⇔　予測predict
        'boosting_type': 'gbdt',        # 勾配ブースティング
        'objective': 'regression',      # 目的関数：回帰
        'metric': 'rmse',               # 回帰分析モデルの性能を測る指標
        #'learning_rate': 0.1,
        'early_stopping_rounds': 30
        }

    bst = test_lgb.train( params = lgbm_params,
                          train_set = lgb_train,
                          valid_sets = [lgb_train, lgb_vaild ],
                          verbose_eval = 10,
                          num_boost_round = 5000 )

    lib.log.write( "best_iteration:{}".format( str( bst.best_iteration ) ) )
    lib.log.write( "best_score:{}".format( str( bst.best_score ) ) )
    lib.log.write( "best_params:{}".format( str( bst.params ) ) )
    
    return bst.params

def test( data, model ):
    predict_answer = model.predict( np.array( data["test_teacher"] ) )
    diff = 0
    count = 0

    for i in range( 0, len( predict_answer ) ):
        diff += abs( predict_answer[i] - data["test_answer"][i] )
        count += 1

    print( diff / count )

def data_check( data ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []

    for i in range( 0, len( data["teacher"] ) ):
        year = data["year"][i]        
        current_data = data["teacher"][i]
        answer_horce_body = data["answer"][i]

        if year == "2020":
            result["test_teacher"].append( current_data )
            result["test_answer"].append( float( answer_horce_body ) )
        else:
            result["teacher"].append( current_data )
            result["answer"].append( float( answer_horce_body ) )

    return result

def main( data ):
    learn_data = data_check( data )
    #params = lgb_test( learn_data )
    model = lg_main( learn_data )
    test( learn_data, model )
