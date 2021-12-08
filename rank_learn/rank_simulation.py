import math
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm

S = 250

def softmax( data ):
    result = []
    sum_data = 0
    value_max = min( data ) * -1

    for i in range( 0, len( data ) ):
        sum_data += math.exp( ( data[i] * -1 ) - value_max )

    for i in range( 0, len( data ) ):
        result.append( math.exp( ( data[i] * -1 ) - value_max ) / sum_data )

    return result

def probability( p_data ):
    number = -1
    count = 0
    r_check = random.random()
    
    for i in range( 0, len( p_data ) ):
        count += p_data[i]
        
        if r_check <= count:
            number = i
            break

    if number == -1:
        number = len( p_data )

    return number

def main( model, data ):
    recovery_rate = 0
    test_result = { "count": 0, "money": 0, "win": 0 }
    money = 50000
    
    for race_id in tqdm( data.keys() ):
        horce_list = []
        score_list = []
        
        for horce_id in data[race_id].keys():
            p_data = model.predict( np.array( [ data[race_id][horce_id]["data"] ] ) )
            ex_value = {}
            ex_value = {}
            ex_value["score"] = p_data[0]
            ex_value["rank"] = data[race_id][horce_id]["answer"]["rank"]
            ex_value["odds"] = data[race_id][horce_id]["answer"]["odds"]
            #ex_value["rate"] = 0
            #ex_value["ex"] = 0
            #ex_value["horce_id"] = horce_id
            score_list.append( p_data[0] )
            horce_list.append( ex_value )
            
        sort_result = sorted( horce_list, key=lambda x:x["score"] )
        score_list = softmax( score_list )
        max_rate = 0
        bet_horce = sort_result[0]
        ex_value = max( score_list ) * bet_horce["odds"]
        bet_rate = ( max( score_list ) * ( bet_horce["odds"] + 1 ) - 1 ) / bet_horce["odds"]
        
        test_result["count"] += 1        
        bet_money = int( money * bet_rate / 200 ) * 100
        money -= bet_money
        win = False
        
        if bet_horce["rank"] == 1:
            win = True
            money += bet_money * bet_horce["odds"]
            recovery_rate += bet_horce["odds"]
            test_result["win"] += 1
            test_result["money"] += bet_horce["odds"]
            lib.log.write( "odds:" + str( bet_horce["odds"] ) + " score:" + str( max( score_list ) ) )

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
