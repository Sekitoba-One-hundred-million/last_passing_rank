import SekitobaLibrary as lib
import SekitobaDataManage as dm

from tqdm import tqdm
import matplotlib.pyplot as plt

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "waku_three_rate_data.pickle" )

name = "waku_three_rate"
RANK = "rank"
COUNT = "count"

def score_get( data, key_list, key_data, base_key ):
    score = 0
    count = 0
    
    for i in range( 0, len( key_list ) ):
        k1 = key_list[i]
        for r in range( i + 1, len( key_list ) ):
            k2 = key_list[r]
            key_name = k1 + "_" + k2
            try:
                score += data[key_name][key_data[k1]][key_data[k2]][base_key]
                count += 1
            except:
                continue

    return score

def main():
    result = {}
    race_data = dm.dl.data_get( "race_data.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    waku_three_rate_data = dm.dl.data_get( "waku_three_rate_data.pickle" )
    key_list = [ "place", "dist", "limb", "baba", "kind" ]
    
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

        if not year in lib.test_years:
            continue

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            continue

        us_list = []

        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.raceCheck( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data )

            if not cd.raceCheck():
                continue

            waku = -1

            if cd.horceNumber() < cd.allHorceNum() / 2:
                waku = 1
            else:
                waku = 2

            base_key = str( int( waku ) )
            key_data = {}
            key_data["place"] = key_place
            key_data["dist"] = key_dist
            key_data["baba"] = key_baba
            key_data["kind"] = key_kind
            key_data["limb"] = str( int( lib.limbSearch( pd ) ) )
            score = score_get( waku_three_rate_data, key_list, key_data, base_key )
            key_rank = str( int( cd.rank() ) )
            lib.dicAppend( result, key_rank, { "score": 0, "count": 0 } )
            result[key_rank]["score"] += score
            result[key_rank]["count"] += 1

    x_data = []
    y_data = []
    
    for k in result.keys():
        score = result[k]["score"] / result[k]["count"]
        x_data.append( int( k ) )
        y_data.append( score )

    plt.bar( x_data, y_data )
    plt.savefig( "/Volumes/Gilgamesh/sekitoba-rank/" + name + ".png" )
    
if __name__ == "__main__":
    main()
        
