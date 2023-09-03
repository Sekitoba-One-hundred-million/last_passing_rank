import math
import numpy as np

import sekitoba_library as lib
import sekitoba_data_manage as dm

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

        for r in range( 0, query ):
            current_data = data["teacher"][i][r]
            last_rank = int( data["answer"][i][r] )
            current_answer = last_rank

            if year in lib.test_years:
                result["test_teacher"].append( current_data )
                result["test_answer"].append( current_answer )
            else:
                result["teacher"].append( current_data )
                result["answer"].append( current_answer  )

    return result

def score_check( simu_data, model, upload = False ):
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

    for race_id in simu_data.keys():
        year = race_id[0:4]
        check_data = []
        stand_score_list = []
        simu_predict_data[race_id] = {}
        all_horce_num = len( simu_data[race_id] )
        
        for horce_id in simu_data[race_id].keys():
            predict_score = min( predict_data[c], all_horce_num )
            c += 1
            predict_score += simu_data[race_id][horce_id]["answer"]["predict_first_passing_rank"]
            answer_rank = simu_data[race_id][horce_id]["answer"]["last_passing_rank"]
            check_data.append( { "horce_id": horce_id, "answer": answer_rank, "score": predict_score } )
            stand_score_list.append( predict_score )

        stand_score_list = lib.standardization( stand_score_list )
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
            #predict_score = int( check_data[i]["score"] + 0.5 )
            simu_predict_data[race_id][check_data[i]["horce_id"]] = {}
            simu_predict_data[race_id][check_data[i]["horce_id"]]["index"] = predict_score
            simu_predict_data[race_id][check_data[i]["horce_id"]]["score"] = check_data[i]["score"]
            simu_predict_data[race_id][check_data[i]["horce_id"]]["stand"] = stand_score_list[i]

            if year in lib.test_years:
                score1 += math.pow( predict_score - check_answer, 2 )
                score2 += math.pow( max( int( check_data[i]["score"] + 0.5 ), 1 ) - check_answer, 2 )
                count += 1            
            
    score1 /= count
    score1 = math.sqrt( score1 )
    print( "score1: {}".format( score1 ) )

    score2 /= count
    score2 = math.sqrt( score2 )
    print( "score2: {}".format( score2 ) )

    if upload:
        dm.pickle_upload( "predict_last_passing_rank.pickle", simu_predict_data )

    return score1, score2
