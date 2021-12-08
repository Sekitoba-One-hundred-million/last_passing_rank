import copy

import sekitoba_library as lib
import sekitoba_data_manage as dm

dm.dl.file_set( "jockey_anlyze_data.pickle" )
dm.dl.file_set( "jockey_id_data.pickle" )

class JockeyData:
    def __init__( self ):
        self.jockey_data = dm.dl.data_get( "jockey_anlyze_data.pickle" )
        self.jockey_id_data = dm.dl.data_get( "jockey_id_data.pickle" )

    def key_create( self, day, race_num ):
        key = ""
        ymd = day.split( "/" )
        race_num = str( int( race_num ) )
        
        try:
            if len( ymd[1] ) == 1:
                ymd[1] = "0" + ymd[1]
                
            if len( ymd[2] ) == 1:
                ymd[2] = "0" + ymd[2]

            if len( race_num ) == 1:
                new_race_num = "0" + race_num
                
            key = ymd[0] + ymd[1] + ymd[2] + new_race_num
        except:
            return key

        return key

    def data_get( self, horce_id, day, race_num ):
        result = {}
        instance = { "count": 0, "rank": 0, "one": 0, "two": 0, "three": 0, "time": 0, "up": 0 }
        instance100 = { "count": 0, "rank": 0, "one": 0, "two": 0, "three": 0, "time": 0, "up": 0 }
        result["all"] = instance
        result["100"] = instance100
        jockey_id = self.jockey_id_data[horce_id][day]
        race_key = self.key_create( day, race_num )
        
        try:
            jockey_id = self.jockey_id_data[horce_id][day]
        except:
            return result
        
        race_key = self.key_create( day, race_num )

        if len( race_key ) == 0:
            return result

        instance = copy.copy( self.jockey_data[jockey_id][race_key]["all"] )
        instance100 = copy.copy( self.jockey_data[jockey_id][race_key]["100"] )
        result["all"] = instance
        result["100"] = instance100
        
        return result
