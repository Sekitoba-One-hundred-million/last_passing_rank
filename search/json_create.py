import json
from tqdm import tqdm

import sekitoba_library as lib
import sekitoba_data_manage as dm

dm.dl.file_set( "race_cource_info.pickle" )
dm.dl.file_set( "race_cource_wrap.pickle" )
dm.dl.file_set( "first_pace_analyze_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "first_pace_analyze_data.pickle" )
dm.dl.file_set( "passing_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "corner_horce_body.pickle" )

def json_write( file_name, value ):
    f = open( file_name, "w" )
    json.dump( value, f, ensure_ascii = False, indent = 4 )
    f.close()

def use_corner_check( s ):
    if s == 4:
        return [ "1", "3" ]
    elif s == 3:
        return [ "2", "3" ]
    
    return [ "3" ]

def learn_corner_check( rci_info ):
    result = []
    c = 0
    use_corner = use_corner_check( min( rci_info.count( "c" ), 4 ) )
    
    for i in range( 0, len( rci_info ) - 1 ):   
        check = rci_info.count( "c", i, len( rci_info ) )

        if rci_info[i] == "s" and rci_info[i+1] == "c":
            instance = {}
            instance["count"] = i
            instance["corner"] = None

            if check <= 4:
                instance["corner"] = use_corner[c]
                c += 1
            
            result.append( instance )

    return result

def main():
    target = {}
    fevalue = {}
    target["Info"] = []
    target["Value"] = []
    target["Info"].append( { "Name": "horce_body",  "Up": False } )

    fevalue["Info"] = []
    fevalue["Value"] = []
    #fevalue["Info"].append( { "Name": "others" } )
    fevalue["Info"].append( { "Name": "escape-a" } )
    fevalue["Info"].append( { "Name": "escape-b" } )
    fevalue["Info"].append( { "Name": "preced-a" } )
    fevalue["Info"].append( { "Name": "preced-b" } )
    #fevalue["Info"].append( { "Name": "insert-a" } )
    #fevalue["Info"].append( { "Name": "insert-b" } )
    #fevalue["Info"].append( { "Name": "chase" } )
    #fevalue["Info"].append( { "Name": "rear" } )
    fevalue["Info"].append( { "Name": "dist" } )
    fevalue["Info"].append( { "Name": "run_dist" } )
    fevalue["Info"].append( { "Name": "straight_dist" } )
    fevalue["Info"].append( { "Name": "predict_limb" } )
    #fevalue["Info"].append( { "Name": "before_horce_body" } )
    
    race_data = dm.dl.data_get( "race_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    race_cource_wrap = dm.dl.data_get( "race_cource_wrap.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    first_pace_analyze_data = dm.dl.data_get( "first_pace_analyze_data.pickle" )
    passing_data = dm.dl.data_get( "passing_data.pickle" )
    race_cource_info = dm.dl.data_get( "race_cource_info.pickle" )
    corner_horce_body = dm.dl.data_get( "corner_horce_body.pickle" )

    for k in tqdm( race_data.keys() ):
        race_id = lib.id_get( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( race_info[race_id]["place"] )
        key_dist = str( race_info[race_id]["dist"] )
        key_kind = str( race_info[race_id]["kind"] )        
        key_baba = str( race_info[race_id]["baba"] )
        
        info_key_dist = key_dist
        
        if race_info[race_id]["out_side"]:
            info_key_dist += "外"

        try:
            rci_dist = race_cource_info[key_place][key_kind][info_key_dist]["dist"]
            rci_info = race_cource_info[key_place][key_kind][info_key_dist]["info"]
        except:
            continue
            
        use_learn_corner = learn_corner_check( rci_info )
        race_limb = [0] * 9

        for kk in race_data[k].keys():
            horce_name = kk.replace( " ", "" )
            current_data, past_data = lib.race_check( horce_data[horce_name],
                                                          year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )
            
            if not cd.race_check():
                continue
            
            try:
                limb_math = lib.limb_search( passing_data[horce_name], pd )
            except:
                limb_math = 0

            race_limb[limb_math] += 1

        for kk in race_data[k].keys():
            horce_name = kk.replace( " ", "" )
            current_data, past_data = lib.race_check( horce_data[horce_name],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue
            
            key_horce_num = str( int( cd.horce_number() ) )
            before_horce_body = 0
            
            for i in range( 0, len( use_learn_corner ) ):                
                value_instance = []
                target_instance = []
                u = use_learn_corner[i]["count"]
                c = use_learn_corner[i]["corner"]

                if c == None:
                    continue
                
                try:
                    horce_body = corner_horce_body[race_id][c][key_horce_num]
                except:
                    horce_body = 0

                try:
                    before_horce_body = corner_horce_body[race_id][str(int(c)-1)][key_horce_num]
                except:
                    before_horce_body = 0

                #value.append( race_limb[0] )
                value_instance.append( race_limb[1] )
                value_instance.append( race_limb[2] )
                value_instance.append( race_limb[3] )
                value_instance.append( race_limb[4] )
                #value_instance.append( race_limb[5] )
                #value_instance.append( race_limb[6] )
                #value_instance.append( race_limb[7] )
                #value_instance.append( race_limb[8] )
                value_instance.append( float( key_dist ) )
                value_instance.append( sum( rci_dist[0:u] ) )
                value_instance.append( rci_dist[u] )
                value_instance.append( lib.limb_search( passing_data[horce_name], pd ) )
                #value_instance.append( before_horce_body )
                
                target_instance.append( horce_body )

                target["Value"].append( target_instance )
                fevalue["Value"].append( value_instance )

    json_write( "target.json", target )
    json_write( "fevalue.json", fevalue )

main()
