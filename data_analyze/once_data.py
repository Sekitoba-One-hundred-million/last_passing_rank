import math
from tqdm import tqdm
from mpi4py import MPI

import sekitoba_library as lib
import sekitoba_data_manage as dm
import sekitoba_data_create as dc

dm.dl.file_set( "race_cource_info.pickle" )
dm.dl.file_set( "race_cource_wrap.pickle" )
dm.dl.file_set( "first_pace_analyze_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "first_pace_analyze_data.pickle" )
dm.dl.file_set( "passing_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "corner_horce_body.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )
dm.dl.file_set( "parent_id_data.pickle" )
dm.dl.file_set( "time_index_data.pickle" )
dm.dl.file_set( "blood_closs_data.pickle" )
dm.dl.file_set( "win_rate_data.pickle" )
dm.dl.file_set( "win_rate_data.pickle" )
dm.dl.file_set( "race_limb_claster_model.pickle" )
dm.dl.file_set( "first_horce_body.pickle" )

class OnceData:
    def __init__( self ):
        self.race_data = dm.dl.data_get( "race_data.pickle" )
        self.horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
        self.race_cource_wrap = dm.dl.data_get( "race_cource_wrap.pickle" )
        self.race_info = dm.dl.data_get( "race_info_data.pickle" )
        self.first_pace_analyze_data = dm.dl.data_get( "first_pace_analyze_data.pickle" )
        self.passing_data = dm.dl.data_get( "passing_data.pickle" )
        self.race_cource_info = dm.dl.data_get( "race_cource_info.pickle" )
        self.corner_horce_body = dm.dl.data_get( "corner_horce_body.pickle" )
        self.baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
        self.parent_id_data = dm.dl.data_get( "parent_id_data.pickle" )
        self.blood_closs_data = dm.pickle_load( "blood_closs_data.pickle" )
        self.win_rate_data = dm.pickle_load( "win_rate_data.pickle" )
        self.race_limb_claster_model = dm.dl.data_get( "race_limb_claster_model.pickle" )
        self.first_horce_body_data = dm.dl.data_get( "first_horce_body.pickle" )        
        self.train_index = dc.TrainIndexGet()
        self.time_index = dc.TimeIndexGet()
        self.jockey_data = dc.JockeyData()
        self.up_score = dc.UpScore()
        self.past_horce_body = dc.PastHorceBody()        
        self.simu_data = {}        
        self.result = { "answer": [], "teacher": [], "query": [], "year": [] }

    def speed_standardization( self, data ):
        result = []
        ave = 0
        conv = 0
        count = 0

        for d in data:
            if d < 0:
                continue
        
            ave += d
            count += 1

        if count == 0:
            return [0] * len( data )

        ave /= count

        for d in data:
            if d < 0:
                continue

            conv += math.pow( d - ave, 2 )

        conv /= count
        conv = math.sqrt( conv )

        if conv == 0:
            return [0] * len( data )
    
        for d in data:
            if d < 0:
                result.append( 0 )
            else:
                result.append( ( d - ave ) / conv )

        return result

    def max_check( self, data ):
        try:
            return max( data )
        except:
            return -1

    def create( self, k ):
        race_id = lib.id_get( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( self.race_info[race_id]["place"] )
        key_dist = str( self.race_info[race_id]["dist"] )
        key_kind = str( self.race_info[race_id]["kind"] )
        key_baba = str( self.race_info[race_id]["baba"] )
        ri_list = [ key_place + ":place", key_dist + ":dist", key_kind + ":kind", key_baba + ":baba" ]
        info_key_dist = key_dist
        
        if self.race_info[race_id]["out_side"]:
            info_key_dist += "外"

        try:
            rci_dist = self.race_cource_info[key_place][key_kind][info_key_dist]["dist"]
            rci_info = self.race_cource_info[key_place][key_kind][info_key_dist]["info"]
        except:
            return
            
        rci_dist = self.race_cource_info[key_place][key_kind][info_key_dist]["dist"]
        rci_info = self.race_cource_info[key_place][key_kind][info_key_dist]["info"]
        
        race_limb = [0] * 9
        popular_limb = -1
        up3_instance = []
        data_list = {}
        data_list["speed"] = []
        data_list["up_speed"] = []
        data_list["pace_speed"] = []
        data_list["time_index"] = []
        data_list["average_speed"] = []        
        count = -1

        for kk in self.race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )
            
            if not cd.race_check():
                continue

            key_horce_num = str( int( cd.horce_number() ) )
            current_time_index = self.time_index.main( kk, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.baba_index_data[horce_id] )
            data_list["speed"].append( self.max_check( speed ) )
            data_list["up_speed"].append( self.max_check( up_speed ) )
            data_list["pace_speed"].append( self.max_check( pace_speed ) )            
            data_list["time_index"].append( current_time_index["max"] )
            data_list["average_speed"].append( pd.average_speed() )
            
            try:
                up3 = sum( self.first_up3_halon[race_id][key_horce_num] ) / len( self.first_up3_halon[race_id][key_horce_num] )
            except:
                up3 = 0

            up3_instance.append( up3 )
            limb_math = lib.limb_search( pd )

            if cd.popular() == 1:
                popular_limb = limb_math
            
            race_limb[limb_math] += 1

        data_list["stand_speed"] = self.speed_standardization( data_list["speed"] )
        data_list["stand_up_speed"] = self.speed_standardization( data_list["up_speed"] )
        data_list["stand_pace_speed"] = self.speed_standardization( data_list["pace_speed"] )
        data_list["stand_time_index"] = self.speed_standardization( data_list["time_index"] )
        data_list["stand_average_speed"] = self.speed_standardization( data_list["average_speed"] )

        try:
            ave_first_up3 = sum( up3_instance ) / len( up3_instance )
        except:
            return
        
        for kk in self.race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            count += 1
            current_jockey = self.jockey_data.data_get( horce_id, cd.birthday(), cd.race_num() )
            key_horce_num = str( int( cd.horce_number() ) )
            stand_speed = data_list["stand_speed"][count]
            stand_up_speed = data_list["stand_up_speed"][count]
            stand_pace_speed = data_list["stand_pace_speed"][count]
            stand_time_index = data_list["stand_time_index"][count]            
            stand_average_speed = data_list["stand_average_speed"][count]
            
            t_instance = []
            limb_math = lib.limb_search( pd )

            if not year == lib.test_year:
                try:
                    key = min( self.corner_horce_body[race_id] )
                    first_horce_body = self.corner_horce_body[race_id][key][key_horce_num]
                except:
                    continue
            else:
                try:
                    first_horce_body = self.first_horce_body_data[race_id][key_horce_num]
                except:
                    continue

            try:
                last_horce_body = self.corner_horce_body[race_id]["4"][key_horce_num]
            except:
                last_horce_body = -1

            father_id = self.parent_id_data[horce_id]["father"]
            mother_id = self.parent_id_data[horce_id]["mother"]
            father_data = dc.parent_data_get.main( self.horce_data, father_id, self.baba_index_data )
            mother_data = dc.parent_data_get.main( self.horce_data, mother_id, self.baba_index_data )
            current_train = self.train_index.main( race_id, key_horce_num )
            
            dm.dn.append( t_instance, race_limb[0], "その他の馬の数" )
            dm.dn.append( t_instance, race_limb[1], "逃げaの馬の数" )
            dm.dn.append( t_instance, race_limb[2], "逃げbの馬の数" )
            dm.dn.append( t_instance, race_limb[3], "先行aの馬の数" )
            dm.dn.append( t_instance, race_limb[4], "先行bの馬の数" )
            dm.dn.append( t_instance, race_limb[5], "差しaの馬の数" )
            dm.dn.append( t_instance, race_limb[6], "差しbの馬の数" )
            dm.dn.append( t_instance, race_limb[7], "追いの馬の数" )
            dm.dn.append( t_instance, race_limb[8], "後方の馬の数" )
            dm.dn.append( t_instance, popular_limb, "一番人気の馬の脚質" )
            dm.dn.append( t_instance, float( key_place ), "場所" )
            dm.dn.append( t_instance, float( key_dist ), "距離" )
            dm.dn.append( t_instance, float( key_kind ), "芝かダート" )
            dm.dn.append( t_instance, float( key_baba ), "馬場" )
            dm.dn.append( t_instance, cd.id_weight(), "馬体重の増減" )
            dm.dn.append( t_instance, cd.burden_weight(), "斤量" )
            dm.dn.append( t_instance, cd.horce_number(), "馬番" )
            dm.dn.append( t_instance, cd.flame_number(), "枠番" )
            dm.dn.append( t_instance, cd.all_horce_num(), "馬の頭数" )
            dm.dn.append( t_instance, float( key_dist ) - rci_dist[-1], "今まで走った距離" )
            dm.dn.append( t_instance, rci_dist[-1], "直線の距離" )
            dm.dn.append( t_instance, limb_math, "過去データからの予想脚質" )
            
            dm.dn.append( t_instance, data_list["speed"][count], "スピード指数" )
            dm.dn.append( t_instance, stand_speed, "standスピード指数" )
            dm.dn.append( t_instance, data_list["pace_speed"][count], "ペース指数" )
            dm.dn.append( t_instance, stand_pace_speed , "standペース指数" )            
            dm.dn.append( t_instance, data_list["time_index"][count], "タイム指数" )
            dm.dn.append( t_instance, stand_time_index, "standタイム指数" )
            
            dm.dn.append( t_instance, pd.three_average(), "過去3レースの平均順位" )
            dm.dn.append( t_instance, pd.dist_rank_average(), "過去同じ距離の種類での平均順位" )
            dm.dn.append( t_instance, pd.racekind_rank_average(), "過去同じレース状況での平均順位" )
            dm.dn.append( t_instance, pd.baba_rank_average(), "過去同じ馬場状態での平均順位" )
            dm.dn.append( t_instance, pd.jockey_rank_average(), "過去同じ騎手での平均順位" )
            dm.dn.append( t_instance, pd.three_average(), "複勝率" )
            dm.dn.append( t_instance, pd.two_rate(), "連対率" )
            dm.dn.append( t_instance, pd.get_money(), "獲得賞金" )
            dm.dn.append( t_instance, pd.best_weight(), "ベスト体重と現在の体重の差" )
            dm.dn.append( t_instance, pd.race_interval(), "中週" )
            dm.dn.append( t_instance, pd.average_speed(), "平均速度" )
            dm.dn.append( t_instance, pd.pace_up_check(), "ペースと上りの関係" )
            dm.dn.append( t_instance, pd.passing_regression(), "passing regression" )            
            dm.dn.append( t_instance, current_train["score"] ,"調教score" )
            dm.dn.append( t_instance, current_train["a"], "調教ペースの傾き" )            
            dm.dn.append( t_instance, current_train["b"], "調教ペースの切片" )            
            dm.dn.append( t_instance, father_data["rank"], "父親の平均順位" )
            dm.dn.append( t_instance, father_data["two_rate"], "父親の連対率" )
            dm.dn.append( t_instance, father_data["three_rate"], "父親の副賞率" )
            dm.dn.append( t_instance, father_data["average_speed"], "父親の平均速度" )
            dm.dn.append( t_instance, father_data["speed_index"], "父親の最大のスピード指数" )
            dm.dn.append( t_instance, father_data["up_speed_index"], "父親の最大の上りスピード指数" )
            dm.dn.append( t_instance, father_data["pace_speed_index"], "父親の最大のペース指数" )
            dm.dn.append( t_instance, father_data["limb"], "父親の脚質" )
            dm.dn.append( t_instance, mother_data["rank"], "母親の平均順位" )
            dm.dn.append( t_instance, mother_data["two_rate"], "母親の連対率" )
            dm.dn.append( t_instance, mother_data["three_rate"], "母親の副賞率" )
            dm.dn.append( t_instance, mother_data["average_speed"], "母親の平均速度" )
            dm.dn.append( t_instance, mother_data["speed_index"], "母親の最大のスピード指数" )
            dm.dn.append( t_instance, mother_data["up_speed_index"], "母親の最大の上りスピード指数" )
            dm.dn.append( t_instance, mother_data["pace_speed_index"], "母親の最大のペース指数" )
            dm.dn.append( t_instance, mother_data["limb"], "母親の脚質" )

            dm.dn.append( t_instance, current_jockey["all"]["rank"], "騎手の過去の平均順位" )
            dm.dn.append( t_instance, current_jockey["all"]["one"], "騎手の過去のone" )
            dm.dn.append( t_instance, current_jockey["all"]["two"], "騎手の過去のtwo" )
            dm.dn.append( t_instance, current_jockey["all"]["three"], "騎手の過去のthree" )
            dm.dn.append( t_instance, current_jockey["all"]["time"], "騎手の過去のタイム" )
            dm.dn.append( t_instance, current_jockey["all"]["up"], "騎手の過去の上り" )
            dm.dn.append( t_instance, current_jockey["100"]["rank"], "騎手の過去の100の平均順位" )
            dm.dn.append( t_instance, current_jockey["100"]["one"], "騎手の過去の100のone" )
            dm.dn.append( t_instance, current_jockey["100"]["two"], "騎手の過去の100のtwo" )
            dm.dn.append( t_instance, current_jockey["100"]["three"], "騎手の過去の100のthree" )
            dm.dn.append( t_instance, current_jockey["100"]["time"], "騎手の過去の100のタイム" )
            dm.dn.append( t_instance, current_jockey["100"]["up"], "騎手の過去の100の上り" )
            dm.dn.append( t_instance, self.up_score.score_get( pd ), "up_score" )
            dm.dn.append( t_instance, ( up3 - ave_first_up3 ) * rci_dist[0], "first_up_score" )            
            dm.dn.append( t_instance, first_horce_body, "最初の馬身" )

            if year == lib.test_year:
                lib.dic_append( self.simu_data, race_id, {} )
                self.simu_data[race_id][key_horce_num] = {}
                self.simu_data[race_id][key_horce_num]["answer"] = last_horce_body
                self.simu_data[race_id][key_horce_num]["data"] = t_instance

            self.result["answer"].append( last_horce_body )
            self.result["teacher"].append( t_instance )
            self.result["year"].append( year )

        if not count + 1 == 0:
            self.result["query"].append( { "q": count + 1, "year": year } )
