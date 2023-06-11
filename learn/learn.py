import math
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

import sekitoba_library as lib
import sekitoba_data_manage as dm
#from learn import simulation

def lg_main( data ):
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )
    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
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
    
    dm.pickle_upload( lib.name.model_name(), bst )
        
    return bst

def data_check( data ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    result["query"] = []
    result["test_query"] = []

    for i in range( 0, len( data["teacher"] ) ):
        year = data["year"][i]
        query = len( data["teacher"][i] )

        if year in lib.test_years:
            result["test_query"].append( query )
        else:
            result["query"].append( query )

        n = int( query / 3 )
        
        for r in range( 0, query ):
            current_data = data["teacher"][i][r]
            first_rank = int( data["answer"][i][r] )
            current_answer = first_rank

            if first_rank / n < 1:
                current_answer -= 1
            elif first_rank / n > 2:
                current_answer += 1

            if first_rank == 1 or first_rank == 2:
                current_answer -= 1
            
            if year in lib.test_years:
                result["test_teacher"].append( current_data )
                result["test_answer"].append( current_answer )
            else:
                result["teacher"].append( current_data )
                result["answer"].append( current_answer  )

    return result

def score_check( simu_data, model ):
    score1 = 0
    score2 = 0
    count = 0
    simu_predict_data = {}
    predict_use_data = []

    for race_id in simu_data.keys():
        for horce_id in simu_data[race_id].keys():
            predict_use_data.append( simu_data[race_id][horce_id]["data"] )

    c = 0
    predict_data = model.predict( np.array( predict_use_data ) )

    for race_id in tqdm( simu_data.keys() ):
        year = race_id[0:4]
        check_data = []
        stand_data_list = []
        simu_predict_data[race_id] = {}
        all_horce_num = len( simu_data[race_id] )
        
        for horce_id in simu_data[race_id].keys():
            predict_score = min( predict_data[c], all_horce_num )
            c += 1
            answer_rank = simu_data[race_id][horce_id]["answer"]["last_passing_rank"]
            check_data.append( { "horce_id": horce_id, "answer": answer_rank, "score": predict_score } )
            stand_data_list.append( predict_score )

        stand_data_list = lib.standardization( stand_data_list )
        check_data = sorted( check_data, key = lambda x: x["score"] )
        before_score = 1
        next_rank = 1
        continue_count = 1
        
        for i in range( 0, len( check_data ) ):
            predict_score = -1
            current_score = int( check_data[i]["score"] + 0.5 )

            if continue_count >= 2:
                next_rank += continue_count
                continue_count = 0

            if i == 0:
                predict_score = 1
            elif before_score == current_score:
                continue_count += 1
                predict_score = next_rank
            else:
                next_rank += continue_count
                continue_count = 1
                predict_score = next_rank

            check_answer = check_data[i]["answer"]
            before_score = current_score
            simu_predict_data[race_id][check_data[i]["horce_id"]] = {}
            simu_predict_data[race_id][check_data[i]["horce_id"]]["index"] = predict_score
            simu_predict_data[race_id][check_data[i]["horce_id"]]["score"] = check_data[i]["score"]
            simu_predict_data[race_id][check_data[i]["horce_id"]]["stand"] = stand_data_list[i]

            if year in lib.test_years:
                score1 += math.pow( predict_score - check_answer, 2 )
                score2 += math.pow( max( check_data[i]["score"], 1 ) - check_answer, 2 )
                count += 1
            
    score1 /= count
    score1 = math.sqrt( score1 )
    score2 /= count
    score2 = math.sqrt( score2 )

    print( "score1: {}".format( score1 ) )
    print( "score2: {}".format( score2 ) )
    dm.pickle_upload( "predict_last_passing_rank.pickle", simu_predict_data )    

def importance_check( model ):
    result = []
    importance_data = model.feature_importance()
    f = open( "common/rank_score_data.txt" )
    all_data = f.readlines()
    f.close()
    c = 0

    for i in range( 0, len( all_data ) ):
        str_data = all_data[i].replace( "\n", "" )

        if "False" in str_data:
            continue

        result.append( { "key": str_data, "score": importance_data[c] } )
        c += 1

    result = sorted( result, key = lambda x: x["score"], reverse= True )

    for i in range( 0, len( result ) ):
        print( "{}: {}".format( result[i]["key"], result[i]["score"] ) )

def main( data, simu_data ):
    learn_data = data_check( data )
    model = lg_main( learn_data )
    importance_check( model )
    score_check( simu_data, model )
    #score_check( learn_data["test_teacher"], learn_data["test_answer"], model )

    return model
