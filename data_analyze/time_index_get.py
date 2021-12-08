import sekitoba_library as lib
import sekitoba_data_manage as dm

class time_index_get:
    def __init__( self ):
        self.time_index_data = dm.dl.data_get( "time_index_data.pickle" )

    def main( self, horce_id, day_list ):
        result = {}
        result["max"] = 0
        result["min"] = 10000
        result["average"] = 0

        try:
            time_index = self.time_index_data[horce_id]
        except:
            return result

        count = 0
        
        for i in range( 0, len( day_list ) ):
            d = day_list[i]
            
            if not time_index[d] == 0:
                count += 1
                result["max"] = max( result["max"], time_index[d] )
                result["min"] = min( result["min"], time_index[d] )
                result["average"] += time_index[d]

        if not count == 0:
            result["average"] /= count

        return result
        
        
