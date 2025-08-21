import math
import numpy as np

import SekitobaLibrary as lib
import SekitobaDataManage as dm

def data_check( data, state = "test" ):
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
        data_check = lib.test_year_check( data["year"][i], state )

        if data_check == "test":
            result["test_query"].append( query )
        elif data_check == "teacher":
            result["query"].append( query )

        for r in range( 0, query ):
            current_data = data["teacher"][i][r]
            last_rank = int( data["answer"][i][r] )
            current_answer = last_rank

            if data_check == "test":
                result["test_teacher"].append( current_data )
                result["test_answer"].append( current_answer )
            elif data_check == "teacher":
                result["teacher"].append( current_data )
                result["answer"].append( current_answer  )

    return result

def score_check( simu_data, modelList, score_years = lib.test_years, upload = False ):
    score = 0
    count = 0
    simu_predict_data = {}
    predict_use_data = [[] for _ in range(lib.max_odds_index)]

    for race_id in simu_data.keys():
        for horce_id in simu_data[race_id].keys():
            for i in range( 0, lib.max_odds_index ):
                predict_use_data[i].append( simu_data[race_id][horce_id][i]["data"] )

    c = 0
    predict_data = [[] for _ in range(lib.max_odds_index)]
    
    for i in range( 0, lib.max_odds_index ):
        for model in modelList:
            predict_data[i].append( model.predict( np.array( predict_use_data[i] ) ) )

    for race_id in simu_data.keys():
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        check_data = [[] for _ in range(lib.max_odds_index)]
        stand_score_list = [[] for _ in range(lib.max_odds_index)]
        simu_predict_data[race_id] = {}
        all_horce_num = len( simu_data[race_id] )

        for horce_id in simu_data[race_id].keys():
            simu_predict_data[race_id][horce_id] = [{} for _ in range(lib.max_odds_index)]
            # odds_index roop
            for i in range( 0, lib.max_odds_index ):
                predict_score = 0

                for r in range( 0, len( predict_data[i] ) ):
                    predict_score += predict_data[i][r][c]

                predict_score /= len( modelList )
                predict_first_passing_rank = simu_data[race_id][horce_id][i]["answer"]["predict_first_passing_rank"]
                predict_score = min( predict_score + predict_first_passing_rank, all_horce_num )
                answer_rank = simu_data[race_id][horce_id][i]["answer"]["last_passing_rank"]
                check_data[i].append( { "horce_id": horce_id, "answer": answer_rank, "score": predict_score } )
                stand_score_list[i].append( predict_score )

            c += 1

            for i in range( 0, lib.max_odds_index ):
                stand_score_list[i] = lib.standardization( stand_score_list[i] )
                check_data[i] = sorted( check_data[i], key = lambda x: x["score"] )

                for r in range( 0, len( check_data[i] ) ):
                    check_answer = check_data[i][r]["answer"]
                    horce_id = check_data[i][r]["horce_id"]
                    simu_predict_data[race_id][horce_id][i]["index"] = i + 1
                    simu_predict_data[race_id][horce_id][i]["score"] = int( min( max( check_data[i][r]["score"], 1 ), len( check_data[i] ) ) )
                    simu_predict_data[race_id][horce_id][i]["stand"] = stand_score_list[i][r]

                    if year in score_years:
                        score += abs( simu_predict_data[race_id][horce_id][i]["score"] - check_answer )
                        count += 1

    score /= count
    print( "score: {}".format( score ) )

    if upload:
        dm.pickle_upload( "predict_last_passing_rank.pickle", simu_predict_data )

    return score
