import SekitobaLibrary as lib
import SekitobaDataManage as dm

from SekitobaDataCreate.before_race_score_get import BeforeRaceScore

import math
from tqdm import tqdm
import matplotlib.pyplot as plt

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "wrap_data.pickle" )
dm.dl.file_set( "waku_three_rate_data.pickle" )

name = "before_race_score"
RANK = "rank"
COUNT = "count"

def main():
    result = {}
    race_data = dm.dl.data_get( "race_data.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    wrap_data = dm.dl.data_get( "wrap_data.pickle" )
    waku_three_rate_data = dm.dl.data_get( "waku_three_rate_data.pickle" )
    mdcd = 0
    count = 0
    three_rate = 0
    rate_count = 0
    
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

        data_list = []

        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.raceCheck( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data )

            if not cd.raceCheck():
                continue

            before_cd = pd.beforeCd()
            limb = lib.limbSearch( pd )
            rank = cd.rank()
            score = BeforeRaceScore().score_get( before_cd, limb, horce_id )
            key = str( int( score ) )
            
            lib.dicAppend( result, year, {} )
            lib.dicAppend( result[year], key, { RANK: 0, COUNT: 0 } )
            result[year][key][COUNT] += 1
            result[year][key][RANK] += rank

            data_list.append( { "rank": rank, "score": score } )

        if len( data_list ) == 0:
            continue
        
        data_list = sorted( data_list, key = lambda x: x["score"] )
        rate_count += 1

        if data_list[0]["rank"] <= 3:
            three_rate += 1

        for i in range( 0, len( data_list ) ):
            rank = i + 1
            mdcd += math.pow( rank - data_list[i]["rank"], 2 )
            count += 1

    mdcd /= count
    three_rate /= rate_count
    three_rate *= 100
    print( "mdcd:{}".format( mdcd ) )
    print( "three_rate:{}".format( round( three_rate, 3 ) ) )
    
    for year in result.keys():
        for k in result[year].keys():
            result[year][k][RANK] /= result[year][k][COUNT]
            result[year][k][RANK] = round( result[year][k][RANK], 2 )

    lib.write_rank_csv( result, name + ".csv" )
    
if __name__ == "__main__":
    main()
        
