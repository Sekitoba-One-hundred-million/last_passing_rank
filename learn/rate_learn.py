import random
import numpy as np
import lightgbm as lgb

import sekitoba_library as lib
import sekitoba_data_manage as dm
#from learn import simulation

def lg_main( data, rate_kind, prod = False ):
    lgb_train = lgb.Dataset( np.array( data[rate_kind+"_teacher"] ), np.array( data[rate_kind+"_answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data[rate_kind+"_test_answer"] ) )
    print( rate_kind )

    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'early_stopping_rounds': 30,
        'learning_rate': 0.01,
        'num_iteration': 10000,
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
    
    if prod:
        dm.pickle_upload( rate_kind + "_" + lib.name.model_name() + ".prod", bst )
    else:
        dm.pickle_upload( rate_kind + "_" + lib.name.model_name(), bst )
        
    return bst

def data_check( data, prod = False ):
    result = {}
    result["one_teacher"] = []
    result["two_teacher"] = []
    result["three_teacher"] = []
    result["test_teacher"] = []
    result["one_answer"] = []
    result["two_answer"] = []
    result["three_answer"] = []
    result["one_test_answer"] = []
    result["two_test_answer"] = []
    result["three_test_answer"] = []

    count = 0

    for i in range( 0, len( data["teacher"] ) ):
        year = data["query"][i]["year"]
        current_data = list( data["teacher"][i] )
        current_answer = list( data["answer"][i] )
        current_level = list( data["level"][i] )

        for r in range( 0, len( current_data ) ):
            answer_rank = current_answer[r]
            one_answer = 0
            two_answer = 0
            three_answer = 0

            if answer_rank == 1:
                one_answer = 1
                two_answer = 1
                three_answer = 1
            elif answer_rank == 2:
                two_answer = 1
                three_answer = 1
            elif answer_rank == 3:
                three_answer = 1

            if year in lib.test_years:
                result["test_teacher"].append( current_data[r] )
                result["one_test_answer"].append( one_answer )
                result["two_test_answer"].append( two_answer )
                result["three_test_answer"].append( three_answer )
            else:
                result["one_teacher"].append( current_data[r] )
                result["one_answer"].append( one_answer )
                result["two_teacher"].append( current_data[r] )
                result["two_answer"].append( two_answer )
                result["three_teacher"].append( current_data[r] )
                result["three_answer"].append( three_answer )
                    
    return result

def main( data ):
    result = {}
    learn_data = data_check( data )
    rate_list = [ "one", "two", "three" ]

    #for rt in rate_list:
    rt = "three"
    model = lg_main( learn_data, rt )
    result[rt] = model

    return result
