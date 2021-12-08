import sekitoba_data_manage as dm
import sekitoba_library as lib

def max_check( s ):
    try:
        return max( s )
    except:
        return -100

def main( horce_data, passing_data, parent_id, baba_index ):
    result = {}
    result["rank"] = 0
    result["two_rate"] = 0
    result["three_rate"] = 0
    result["average_speed"] = 0
    result["limb"] = 0
    result["speed_index"] = -100
    result["up_speed_index"] = -100
    result["pace_speed_index"] = -100

    try:
        parent_data = horce_data[parent_id]
    except:
        return result

    parent_pd = lib.past_data( parent_data, [] )
    speed, up_speed, pace_speed = parent_pd.speed_index( baba_index[parent_id] )  
    result["rank"] = parent_pd.rank()
    result["two_rate"] = parent_pd.two_rate()
    result["three_rate"] = parent_pd.three_rate()
    result["average_speed"] = parent_pd.average_speed()
    result["speed_index"] = max_check( speed )
    result["up_speed_index"] = max_check( up_speed )
    result["pace_speed_index"] = max_check( pace_speed )

    try:
        result["limb"] = lib.limb_search( passing_data[parent_id], parent_pd )
    except:
        return result

    return result

        
