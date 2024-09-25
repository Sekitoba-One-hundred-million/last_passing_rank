import os
import numpy as np
from tqdm import tqdm

import SekitobaLibrary as lib
import SekitobaDataManage as dm
import SekitobaDataCreate as dc

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )

name = "rank_score_odds"

lib.name.set_name( "rank" )

def main():
    result = {}
    data_storage = []
    rank_score_data = dm.pickle_load( lib.name.score_name() )
    
    for k in tqdm( rank_score_data.keys() ):
        race_id = k
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        for kk in rank_score_data[k].keys():
            horce_id = kk
            score = rank_score_data[k][kk]["score"] * rank_score_data[k][kk]["answer"]["odds"]
            rank = rank_score_data[k][kk]["answer"]["rank"]
            instance = {}
            instance["key"] = score
            instance["odds"] = 0
            instance["year"] = year

            if rank == 1:
                instance["odds"] = rank_score_data[k][kk]["answer"]["odds"]

            data_storage.append( instance )

    result, split_list = lib.recovery_data_split( data_storage )
    
    for year in result.keys():
        for k in result[year].keys():
            result[year][k]["recovery"] /= result[year][k]["count"]
            result[year][k]["recovery"] = round( result[year][k]["recovery"], 2 )

    lib.write_recovery_csv( result,  name + ".csv" )
    score = lib.recoveryScoreCheck( result )
    lib.recovery_data_upload( name, score, split_list )
    print( split_list )
    print( score )

if __name__ == "__main__":
    main()
        
