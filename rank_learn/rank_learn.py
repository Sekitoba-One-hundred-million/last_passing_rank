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
from rank_learn import rank_simulation

check_year = 2020

def lg_main( data ):
    max_pos = np.max( np.array( data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ), group = np.array( data["query"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ), group = np.array( data["test_query"] ) )

    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',   # for lambdarank
        'ndcg_eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( data["answer"], dtype = np.int32 ) ) + 1)),
        'max_position': int( max_pos ),  # for lambdarank
        'early_stopping_rounds': 30,
        'learning_rate': 0.05,
        'num_iteration': 300,
        'min_data_in_bin': 1,
        'max_depth': 200,
        'num_leaves': 175,
        'min_data_in_leaf': 25,
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )
    
    f_importance = np.array( bst.feature_importance() )
    x_list = np.array( range( len( f_importance ) ) )
    f_importance = f_importance / np.sum( f_importance )
    df_importance = pd.DataFrame( { 'feature':x_list, 'importance':f_importance } )
    df_importance = df_importance.sort_values( 'importance', ascending = False )
    pd.set_option( 'display.max_rows', 150 )
    #print( df_importance.head( len( x_list ) ) )
    
    #dm.pickle_upload( "lightbgm_test_model.pickle", bst )

    return bst

def data_check( data ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    result["query"] = []
    result["test_query"] = []

    count = 0

    for i in range( 0, len( data["query"] ) ):
        q = data["query"][i]["q"]
        year = data["query"][i]["year"]
        
        if year == "2020":
            result["test_query"].append( q )
        else:
            result["query"].append( q )

        current_data = list( data["teacher"][count:count+q] )
        current_answer = list( data["answer_rank"][count:count+q] )

        for r in range( 0, len( current_data ) ):
            answer_rank = current_answer[r]

            if year == "2020":
                result["test_teacher"].append( current_data[r] )
                result["test_answer"].append( float( answer_rank ) )
            else:
                result["teacher"].append( current_data[r] )
                result["answer"].append( float( answer_rank ) )

        count += q

    return result

def main( data, simu_data, simulation = True, learn_data = False ):
    print( "rank_learn" )
    learn_data = data_check( data )
    rank_model = lg_main( learn_data )
    recovery_rate = None
    
    if simulation:
        recovery_rate, win_rate = rank_simulation.main( rank_model, simu_data )

    if learn_data:
        return rank_model, recovery_rate, learn_data
    else:
        return rank_model, recovery_rate
