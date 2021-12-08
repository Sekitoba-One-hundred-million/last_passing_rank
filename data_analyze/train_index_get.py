import math

import sekitoba_library as lib
import sekitoba_data_manage as dm

dm.dl.file_set( "analyze_train_data.pickle" )
dm.dl.file_set( "train_time_data.pickle" )

class train_index_get:
    def __init__( self ):
        self.train_time_data = dm.dl.data_get( "train_time_data.pickle" )
        self.train_analyze_data = dm.dl.data_get( "analyze_train_data.pickle" )

    def train_index_create( self, t_time, cource, baba, foot ):
        result = {}
        result["a"] = 0
        result["b"] = 0
        result["time"] = 0
        
        time = 0
        count = 0
        wrap_time = []

        for i in range( 0, len( t_time ) ):
            if not t_time[i] == 0:
                time = t_time[i]
                count = ( i ) % 2
                break
               
        for i in range( 0, len( t_time ) ):
            if not i % 2 == count \
              and not t_time[i] == 0:
                wrap_time.append( t_time[i] )

        try:
            a, b = lib.regression_line( wrap_time )
        except:
            a = 0
            b = 0
            
        try:
            time *= self.train_analyze_data[cource][foot]["time"]
            time *= self.train_analyze_data[cource][baba]["time"]
            time = self.train_analyze_data[cource]["time"] - time
            time /= self.train_analyze_data[cource]["time_sd"]
        except:
            time = 0

        result["time"] = time
        result["a"] = a
        result["b"] = b

        return result

    def main( self, race_data, horce_data, race_id ):
        result = {}
        fail_dic = { "time": 0, "a": 0, "b": 0 }
        t_instance = []
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]
        train_index_data = []

        for k in race_data.keys():
            horce_name = k.replace( " ", "" )
            current_data, past_data = lib.race_check( horce_data[horce_name],
                                                      year, day, num, race_place_num )#今回と過去のデータに分ける

            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not len( current_data ) == 22:
                continue
    
            key_horce_name = lib.horce_name_replace( horce_name )
            horce_name = key_horce_name
        
            if key_horce_name == "インターパテイショ":
                key_horce_name = "インターパテイション"
            elif key_horce_name == "ビューティーフラッ":
                key_horce_name = "ビューティーフラッシュ"
            elif key_horce_name == "フィフティープルー":
                key_horce_name = "フィフティープルーフ"
            elif key_horce_name == "ウルトラファンタジ":
                key_horce_name = "ウルトラファンタジー"
            elif key_horce_name == "ミッションアプルー":
                key_horce_name = "ミッションアプルーヴド"
            elif key_horce_name == "キャプテンオブヴィ":
                key_horce_name = "キャプテンオブヴィアス"
            elif key_horce_name == "アップウィズザバー":
                key_horce_name = "アップウィズザバーズ"
            elif key_horce_name == "ビューティーオンリ":
                key_horce_name = "ビューティーオンリー"
            elif key_horce_name == "ウエスタンエクスプ":
                key_horce_name = "ウエスタンエクスプレス"

            try:
                t_time = self.train_time_data[race_id][key_horce_name]["time"]        
                cource = self.train_time_data[race_id][key_horce_name]["cource"]
                baba = self.train_time_data[race_id][key_horce_name]["baba"]
                foot = self.train_time_data[race_id][key_horce_name]["foot"]
                train_index_data.append( self.train_index_create( t_time, cource, baba, foot ) )
            except:
                train_index_data.append( fail_dic )
            
        return train_index_data
