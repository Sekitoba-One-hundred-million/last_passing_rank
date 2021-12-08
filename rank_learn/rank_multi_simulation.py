import math
import torch
import random
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm

S = 250
def softmax( data ):
    sum_data = 0
    value_max = -10000000
    
    for i in range( 0, len( data ) ):
        if value_max < data[i]["score"]:
            value_max = data[i]["score"]

    for i in range( 0, len( data ) ):
        sum_data += math.exp( data[i]["score"] - value_max )

    for i in range( 0, len( data ) ):
        data[i]["score"] = math.exp( data[i]["score"] - value_max ) / sum_data

def bet_horce_get( data ):
    max_score = 0
    result = {}
    
    for i in range( 0, len( data ) ):
        if max_score < data[i]["score"]:
            result = data[i]

    return result
        

def main( models, datas, parames ):
    params_log = ""
    
    for k in parames.keys():
        params_log += "{}:{} ".format( k, str( parames[k] ) )    

    lib.log.write( "" )
    lib.log.write( params_log )
    recovery_rate = 0
    test_result = { "count": 0, "money": 0, "win": 0 }
    money = 50000
    score_list = {}
    
    for param_name in datas.keys():
        for race_id in datas[param_name].keys():
            lib.dic_append( score_list, race_id, [] )
            append_check = False
            count = 0

            if len( score_list[race_id] ) == 0:
                append_check = True
            
            for horce_id in datas[param_name][race_id].keys():
                instance = {}
                score = models[param_name].predict( np.array( [ datas[param_name][race_id][horce_id]["data"] ] ) )[0]
                instance["score"] = score * parames[param_name] * -1
                instance["rank"] = datas[param_name][race_id][horce_id]["answer"]["rank"]
                instance["odds"] = datas[param_name][race_id][horce_id]["answer"]["odds"]

                if append_check:
                    score_list[race_id].append( instance )
                else:
                    score_list[race_id][count]["score"] += instance["score"]
                    count += 1

    base_name = list( datas.keys() )[0]

    for race_id in datas[base_name]:
        current_score = copy.deepcopy( score_list[race_id] )
        #softmax( current_score )
        current_score = sorted( current_score, key=lambda x:x["score"], reverse = True )
        bet_horce = current_score[0]        
        test_result["count"] += 1        
        if bet_horce["rank"] == 1:
            #money += bet_money * bet_horce["odds"]
            test_result["win"] += 1
            test_result["money"] += bet_horce["odds"]
            #lib.log.write( "odds:" + str( bet_horce["odds"] ) + " score:" + str( max( score_list ) ) )

        """
        if not win:
            lib.log.write( "bet_money:{} money:{}".format( bet_money, money ) )
        else:
            lib.log.write( "bet_money:{} get_money:{} money:{}".format( bet_money, bet_money * bet_horce["odds"], money ) )
        """

    recovery_rate = test_result["money"] / test_result["count"]
    recovery_rate *= 100
    win_rate = test_result["win"] / test_result["count"]
    win_rate *= 100
    
    lib.log.write( "回収率{}%".format( recovery_rate ) )
    lib.log.write( "勝率{}%".format( win_rate ) )
    lib.log.write( "賭けた回数{}回".format( test_result["count"] ) )

    return recovery_rate, win_rate
