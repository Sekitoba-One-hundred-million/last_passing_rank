import os
import numpy as np
from tqdm import tqdm

import SekitobaLibrary as lib
import SekitobaDataManage as dm

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "race_jockey_id_data.pickle" )
dm.dl.file_set( "true_skill_data.pickle" )

name = "jockey_true_skill"
RANK = "rank"
COUNT = "count"
    
def main():
    result = {}
    data_storage = []
    race_data = dm.dl.data_get( "race_data.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    race_jockey_id_data = dm.dl.data_get( "race_jockey_id_data.pickle" )
    horce_jockey_true_skill_data = dm.dl.data_get( "true_skill_data.pickle" )
    
    for k in tqdm( race_data.keys() ):
        race_id = lib.idGet( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( race_info[race_id]["place"] )
        key_dist = str( race_info[race_id]["dist"] )
        key_kind = str( race_info[race_id]["kind"] )        
        key_baba = str( race_info[race_id]["baba"] )

        #if year in lib.test_years:
        #    continue

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            continue

        count = 0
        
        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.raceCheck( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data )

            if not cd.raceCheck():
                continue

            try:
                jockey_id = race_jockey_id_data[race_id][horce_id]
                score = horce_jockey_true_skill_data["jockey"][race_id][jockey_id]
            except:
                score = -1

            key = str( int( score ) )
            lib.dicAppend( result, year, {} )
            lib.dicAppend( result[year], key, { RANK: 0, COUNT: 0 } )
            
            result[year][key][COUNT] += 1
            result[year][key][RANK] += cd.rank()

    for year in result.keys():
        for k in result[year].keys():
            result[year][k][RANK] /= result[year][k][COUNT]
            result[year][k][RANK] = round( result[year][k][RANK], 2 )

    lib.write_rank_csv( result, name + ".csv" )
 

if __name__ == "__main__":
    main()
        
