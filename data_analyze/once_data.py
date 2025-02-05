import math
import copy
from tqdm import tqdm
from mpi4py import MPI

import SekitobaLibrary as lib
import SekitobaDataManage as dm
import SekitobaPsql as ps

from SekitobaDataCreate.stride_ablity import StrideAblity
from SekitobaDataCreate.high_level_data_get import RaceHighLevel
from SekitobaDataCreate.time_index_get import TimeIndexGet
from SekitobaDataCreate.trainer_data_get import TrainerAnalyze
from SekitobaDataCreate.jockey_data_get import JockeyAnalyze
from SekitobaDataCreate.get_horce_data import GetHorceData

from common.name import Name

data_name = Name()

dm.dl.file_set( "predict_first_passing_rank.pickle" )
dm.dl.file_set( 'predict_pace_data.pickle' )

class OnceData:
    def __init__( self ):        
        self.predict_first_passing_rank = dm.dl.data_get( "predict_first_passing_rank.pickle" )
        self.predict_pace_data = dm.dl.data_get( "predict_pace_data.pickle" )

        self.race_data = ps.RaceData()
        self.race_horce_data = ps.RaceHorceData()
        self.horce_data = ps.HorceData()
        self.trainer_data = ps.TrainerData()
        self.jockey_data = ps.JockeyData()

        self.stride_ablity = StrideAblity( self.race_data )
        self.race_high_level = RaceHighLevel()
        self.time_index = TimeIndexGet( self.horce_data )
        self.trainer_analyze = TrainerAnalyze( self.race_data, self.race_horce_data, self.trainer_data )
        self.jockey_analyze = JockeyAnalyze( self.race_data, self.race_horce_data, self.jockey_data )

        self.data_name_list = []
        self.write_data_list = []
        self.simu_data = {}
        self.result = { "answer": [], "teacher": [], "query": [], "year": [], "level": [], "diff": [], "horce_body": [] }
        self.data_name_read()

    def data_name_read( self ):
        f = open( "common/list.txt", "r" )
        str_data_list = f.readlines()

        for str_data in str_data_list:
            self.data_name_list.append( str_data.replace( "\n", "" ) )

    def score_write( self ):
        f = open( "common/rank_score_data.txt", "w" )

        for data_name in self.write_data_list:
            f.write( data_name + "\n" )

        f.close()

    def data_list_create( self, data_dict ):
        result = []
        write_instance = []
        
        for data_name in self.data_name_list:
            try:
                result.append( data_dict[data_name] )
                write_instance.append( data_name )
            except:
                continue

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( write_instance )

        return result

    def division( self, score, d ):
        if score < 0:
            score *= -1
            score /= d
            score *= -1
        else:
            score /= d

        return int( score )

    def clear( self ):
        dm.dl.data_clear()
    
    def create( self, race_id ):
        self.race_data.get_all_data( race_id )
        self.race_horce_data.get_all_data( race_id )

        if len( self.race_horce_data.horce_id_list ) == 0:
            return

        self.horce_data.get_multi_data( self.race_horce_data.horce_id_list )
        self.trainer_data.get_multi_data( self.race_horce_data.trainer_id_list )
        self.jockey_data.get_multi_data( self.race_horce_data.jockey_id_list )

        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_kind = str( self.race_data.data["kind"] )      
        ymd = { "year": self.race_data.data["year"], \
               "month": self.race_data.data["month"], \
               "day": self.race_data.data["day"] }

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return

        predict_pace = {}

        if race_id in self.predict_pace_data:
            predict_pace = self.predict_pace_data[race_id]

        predict_netkeiba_pace = lib.netkeiba_pace( self.race_data.data["predict_netkeiba_pace"] )
        current_horce_body = self.race_data.data["corner_horce_body"]
        max_corner_key = lib.max_check( self.race_data.data["corner_horce_body"] )
        key_race_money_class = str( int( lib.money_class_get( self.race_data.data["money"] ) ) )

        teacher_data = []
        answer_data = []
        answer_horce_body = []
        diff_data = []
        horce_id_list = []
        race_limb = {}
        getHorceDataDict: dict[ str, GetHorceData ] = {}
        escape_limb1_count = 0
        escape_limb2_count = 0
        one_popular_limb = lib.escapeValue
        two_popular_limb = lib.escapeValue
        one_popular_odds = lib.escapeValue
        two_popular_odds = lib.escapeValue
        current_race_data = {}
        current_race_data[data_name.my_limb_count] = { "-1": lib.escapeValue }

        for name in self.data_name_list:
            if name in current_race_data:
                continue

            current_race_data[name] = []

        for horce_id in self.race_horce_data.horce_id_list:
            current_data, past_data = lib.race_check( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.race_check():
                continue

            getHorceData = GetHorceData( cd, pd )
            getHorceDataDict[horce_id] = getHorceData

            if not getHorceData.limb_math == -1:
                lib.dic_append( current_race_data[data_name.my_limb_count], getHorceData.key_limb, 0 )
                current_race_data[data_name.my_limb_count][getHorceData.key_limb] += 1
                
            race_limb[horce_id] = getHorceData.limb_math

            if getHorceData.limb_math == 1:
                escape_limb1_count += 1

            if getHorceData.limb_math == 2:
                escape_limb2_count += 1

            escape_within_rank = -1
            
            if getHorceData.limb_math == 1 or getHorceData.limb_math == 2:
                escape_within_rank = cd.horce_number()

            jockey_id = self.race_horce_data.data[horce_id]["jockey_id"]
            trainer_id = self.race_horce_data.data[horce_id]["trainer_id"]

            horce_true_skill = self.race_horce_data.data[horce_id]["horce_true_skill"]
            jockey_true_skill = self.race_horce_data.data[horce_id]["jockey_true_skill"]
            trainer_true_skill = self.race_horce_data.data[horce_id]["trainer_true_skill"]

            horce_last_passing_true_skill  =self.race_horce_data.data[horce_id]["horce_last_passing_true_skill"]
            jockey_last_passing_true_skill = self.race_horce_data.data[horce_id]["jockey_last_passing_true_skill"]
            trainer_last_passing_true_skill = self.race_horce_data.data[horce_id]["trainer_last_passing_true_skill"]

            past_min_last_horce_body, past_max_last_horce_body, past_ave_last_horce_body, past_std_last_horce_body = \
              getHorceData.getLastHorceBody()
                    
            popular = cd.popular()
            odds = cd.odds()

            if popular == 1:
                one_popular_limb = getHorceData.limb_math
                one_popular_odds = odds
            elif popular == 2:
                two_popular_limb = getHorceData.limb_math
                two_popular_odds = odds

            stride_ablity_data = self.stride_ablity.ablity_create( cd, pd )
            
            for stride_data_key in stride_ablity_data.keys():
                current_race_data[stride_data_key].append( stride_ablity_data[stride_data_key] )

            current_year = cd.year()
            horce_birth_day = int( horce_id[0:4] )
            age = current_year - horce_birth_day
            current_time_index = self.time_index.main( horce_id, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.horce_data.data[horce_id]["baba_index"] )
            corner_diff_rank_ave = pd.corner_diff_rank()
            current_race_data[data_name.horce_true_skill].append( horce_true_skill )
            current_race_data[data_name.jockey_true_skill].append( jockey_true_skill )
            current_race_data[data_name.trainer_true_skill].append( trainer_true_skill )
            current_race_data[data_name.horce_last_passing_true_skill].append( horce_last_passing_true_skill )
            current_race_data[data_name.jockey_last_passing_true_skill].append( jockey_last_passing_true_skill )
            current_race_data[data_name.trainer_last_passing_true_skill].append( trainer_last_passing_true_skill )
            current_race_data[data_name.corner_diff_rank_ave].append( corner_diff_rank_ave )
            current_race_data[data_name.speed_index].append( lib.max_check( speed ) + current_time_index["max"] )
            current_race_data[data_name.up_rate].append( pd.up_rate( key_race_money_class, self.race_data.data["up_kind_ave"]  ) )
            current_race_data[data_name.burden_weight].append( cd.burden_weight() )
            current_race_data[data_name.age].append( age )
            current_race_data[data_name.level_score].append( pd.level_score( self.race_data.data["money_class_true_skill"] ) )
            current_race_data[data_name.past_min_last_horce_body].append( past_min_last_horce_body )
            current_race_data[data_name.past_max_last_horce_body].append( past_max_last_horce_body )
            current_race_data[data_name.past_ave_last_horce_body].append( past_ave_last_horce_body )
            current_race_data[data_name.past_std_last_horce_body].append( past_std_last_horce_body )
            current_race_data[data_name.stamina].append( pd.stamina_create( getHorceData.key_limb ) )
            horce_id_list.append( horce_id )

        if len( horce_id_list ) < 2:
            return

        current_key_list = []

        for data_key in current_race_data.keys():
            if not type( current_race_data[data_key] ) is list or \
              len( current_race_data[data_key] ) == 0:
                continue

            current_key_list.append( data_key )

        for data_key in current_key_list:
            current_race_data[data_key+"_index"] = sorted( current_race_data[data_key], reverse = True )
            current_race_data[data_key+"_stand"] = lib.standardization( current_race_data[data_key] )
            current_race_data[data_key+"_devi"] = lib.deviation_value( current_race_data[data_key] )

        ave_burden_weight = lib.average( current_race_data[data_name.burden_weight] )
        
        N = len( horce_id_list )

        std_race_horce_true_skill = lib.stdev( current_race_data[data_name.horce_true_skill] )
        std_race_jockey_true_skill = lib.stdev( current_race_data[data_name.jockey_true_skill] )
        std_race_trainer_true_skill = lib.stdev( current_race_data[data_name.trainer_true_skill] )
        std_speed_index = lib.stdev( current_race_data[data_name.speed_index] )
        std_up_rate = lib.stdev( current_race_data[data_name.up_rate] )
        std_past_ave_last_horce_body = lib.stdev( current_race_data[data_name.past_ave_last_horce_body] )
        std_past_max_last_horce_body = lib.stdev( current_race_data[data_name.past_max_last_horce_body] )
        std_past_min_last_horce_body = lib.stdev( current_race_data[data_name.past_min_last_horce_body] )

        min_race_horce_true_skill = lib.minimum( current_race_data[data_name.horce_true_skill] )
        min_race_jockey_true_skill = lib.minimum( current_race_data[data_name.jockey_true_skill] )
        min_race_trainer_true_skill = lib.minimum( current_race_data[data_name.trainer_true_skill] )        

        min_speed_index = lib.minimum( current_race_data[data_name.speed_index] )
        min_up_rate = lib.minimum( current_race_data[data_name.up_rate] )
        min_past_ave_last_horce_body = lib.minimum( current_race_data[data_name.past_ave_last_horce_body] )
        min_past_max_last_horce_body = lib.minimum( current_race_data[data_name.past_max_last_horce_body] )
        min_past_min_last_horce_body = lib.minimum( current_race_data[data_name.past_min_last_horce_body] )

        max_race_horce_true_skill = max( current_race_data[data_name.horce_true_skill] )
        max_race_jockey_true_skill = max( current_race_data[data_name.jockey_true_skill] )
        max_race_trainer_true_skill = max( current_race_data[data_name.trainer_true_skill] )

        max_speed_index = max( current_race_data[data_name.speed_index] )
        max_up_rate = max( current_race_data[data_name.up_rate] )
        max_past_ave_last_horce_body = max( current_race_data[data_name.past_ave_last_horce_body] )
        max_past_max_last_horce_body = max( current_race_data[data_name.past_max_last_horce_body] )
        max_past_min_last_horce_body = max( current_race_data[data_name.past_min_last_horce_body] )

        ave_race_horce_true_skill = lib.average( current_race_data[data_name.horce_true_skill] )
        ave_race_jockey_true_skill = lib.average( current_race_data[data_name.jockey_true_skill] )
        ave_race_trainer_true_skill = lib.average( current_race_data[data_name.trainer_true_skill] )

        ave_speed_index = lib.average( current_race_data[data_name.speed_index] )
        ave_up_rate = lib.average( current_race_data[data_name.up_rate] )
        ave_past_ave_last_horce_body = lib.average( current_race_data[data_name.past_ave_last_horce_body] )
        ave_past_max_last_horce_body = lib.average( current_race_data[data_name.past_max_last_horce_body] )
        ave_past_min_last_horce_body = lib.average( current_race_data[data_name.past_min_last_horce_body] )

        for count, horce_id in enumerate( horce_id_list ):
            current_data, past_data = lib.race_check( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.race_check():
                continue

            getHorceData = getHorceDataDict[horce_id]
            first_passing_rank, last_passing_rank = getHorceData.getCurrentPassingRank()

            place_num = int( race_place_num )
            horce_num = int( cd.horce_number() )
            key_horce_num = str( int( horce_num ) )

            if not max_corner_key in current_horce_body or \
              not key_horce_num in current_horce_body[max_corner_key]:
                continue

            answer_corner_horce_body = current_horce_body[max_corner_key][key_horce_num]
            before_speed_score = getHorceData.getBeforeSpeed()
            before_diff_score = getHorceData.getBeforeDiff()
            before_id_weight_score = getHorceData.getBeforeIdWeight()
            before_popular = getHorceData.getBeforePopular()
            before_rank = getHorceData.getBeforeRank()
            up3_standard_value = getHorceData.getUp3StandardValue()
            diff_load_weight = getHorceData.getDiffLoadWeight()
            before_first_last_diff = getHorceData.getBeforeFirstLastDiff()
            popular_rank = getHorceData.getPopularRank()
            before_first_passing_rank, before_last_passing_rank = getHorceData.getBeforePassingRank()
            predict_first_passing_rank = lib.escapeValue
            predict_first_passing_rank_stand = lib.escapeValue
            predict_first_passing_rank_index = lib.escapeValue
            
            if race_id in self.predict_first_passing_rank and horce_id in self.predict_first_passing_rank[race_id]:
                predict_first_passing_rank = self.predict_first_passing_rank[race_id][horce_id]["score"]
                predict_first_passing_rank_index = self.predict_first_passing_rank[race_id][horce_id]["index"]
                predict_first_passing_rank_stand = self.predict_first_passing_rank[race_id][horce_id]["stand"]

            high_level_score = self.race_high_level.data_get( cd, pd, ymd )
            weight_score = getHorceData.getWeightScore()
            trainer_rank_score = self.trainer_analyze.rank( race_id, horce_id )
            jockey_year_rank_score = self.jockey_analyze.year_rank( horce_id, getHorceData.key_before_year )
            jockey_rank_score = self.jockey_analyze.rank( race_id, horce_id )
            my_limb_count_score = current_race_data[data_name.my_limb_count][getHorceData.key_limb]
            corner_diff_rank_ave = current_race_data[data_name.corner_diff_rank_ave][count]
            up_rate = current_race_data[data_name.up_rate][count]
            age = current_race_data[data_name.age][count]
            ave_burden_weight_diff = ave_burden_weight - cd.burden_weight()
            ave_first_last_diff = pd.ave_first_last_diff()
            money_score = pd.get_money()            
            burden_weight_score = cd.burden_weight()
            horce_sex = self.horce_data.data[horce_id]["sex"]
            dist_kind_count = pd.dist_kind_count()
            ave_first_passing_rank = pd.first_passing_rank()
            three_average = pd.three_average()
            three_difference = pd.three_difference()
            three_rate = pd.three_rate()
            match_rank = pd.match_rank()
            best_weight = pd.best_weight()
            passing_regression = pd.passing_regression()
            average_speed = pd.average_speed()
            best_first_passing_rank = pd.best_first_passing_rank()
            best_second_passing_rank = pd.best_second_passing_rank()
            before_continue_not_three_rank = pd.before_continue_not_three_rank()
            diff_pace_time = pd.diff_pace_time()
            diff_pace_first_passing = pd.diff_pace_first_passing()
            pace_up = pd.pace_up_check( self.race_data.data["up_pace_regressin"] )            
            predict_netkeiba_deployment = lib.escapeValue

            for t in range( 0, len( self.race_data.data["predict_netkeiba_deployment"] ) ):
                if int( horce_num ) in self.race_data.data["predict_netkeiba_deployment"][t]:
                    predict_netkeiba_deployment = t
                    break

            t_instance = {}
            t_instance[data_name.predict_first_passing_rank] = predict_first_passing_rank
            t_instance[data_name.predict_first_passing_rank_index] = predict_first_passing_rank_index
            t_instance[data_name.predict_first_passing_rank_stand] = predict_first_passing_rank_stand
            t_instance[data_name.all_horce_num] = cd.all_horce_num()
            t_instance[data_name.ave_burden_weight_diff] = ave_burden_weight_diff
            t_instance[data_name.ave_first_passing_rank] = ave_first_passing_rank
            t_instance[data_name.ave_first_last_diff] = ave_first_last_diff
            t_instance[data_name.baba] = cd.baba_status()
            t_instance[data_name.before_diff] = before_diff_score
            t_instance[data_name.before_first_passing_rank] = before_first_passing_rank
            t_instance[data_name.before_first_last_diff] = before_first_last_diff
            t_instance[data_name.before_id_weight] = before_id_weight_score
            t_instance[data_name.before_last_passing_rank] = before_last_passing_rank
            t_instance[data_name.before_rank] = before_rank
            t_instance[data_name.burden_weight] = burden_weight_score
            t_instance[data_name.dist_kind] = cd.dist_kind()
            t_instance[data_name.dist_kind_count] = dist_kind_count
            t_instance[data_name.escape_limb1_count] = escape_limb1_count
            t_instance[data_name.escape_limb2_count] = escape_limb2_count
            t_instance[data_name.horce_num] = cd.horce_number()
            t_instance[data_name.horce_sex] = horce_sex
            t_instance[data_name.limb] = getHorceData.limb_math
            t_instance[data_name.my_limb_count] = my_limb_count_score
            t_instance[data_name.odds] = cd.odds()
            t_instance[data_name.one_popular_limb] = one_popular_limb
            t_instance[data_name.one_popular_odds] = one_popular_odds
            t_instance[data_name.place] = place_num
            t_instance[data_name.two_popular_limb] = two_popular_limb
            t_instance[data_name.two_popular_odds] = two_popular_odds
            t_instance[data_name.up3_standard_value] = up3_standard_value
            t_instance[data_name.weight] = weight_score
            t_instance[data_name.weather] = cd.weather()
            t_instance[data_name.diff_load_weight] = diff_load_weight
            t_instance[data_name.popular] = cd.popular()
            t_instance[data_name.ave_horce_true_skill] = \
              lib.minus( ave_race_horce_true_skill, current_race_data[data_name.horce_true_skill][count] )
            t_instance[data_name.ave_jockey_true_skill] = \
              lib.minus( ave_race_jockey_true_skill, current_race_data[data_name.jockey_true_skill][count] )
            t_instance[data_name.ave_trainer_true_skill] = \
              lib.minus( ave_race_trainer_true_skill, current_race_data[data_name.trainer_true_skill][count] )
            t_instance[data_name.ave_speed_index] = \
              lib.minus( ave_speed_index, current_race_data[data_name.speed_index][count] )
            t_instance[data_name.ave_up_rate] = lib.minus( ave_up_rate, up_rate )
            t_instance[data_name.ave_past_ave_last_horce_body] = \
              lib.minus( ave_past_ave_last_horce_body, current_race_data[data_name.past_ave_last_horce_body][count] )
            t_instance[data_name.ave_past_max_last_horce_body] = \
              lib.minus( ave_past_max_last_horce_body, current_race_data[data_name.past_max_last_horce_body][count] )
            t_instance[data_name.ave_past_min_last_horce_body] = \
              lib.minus( ave_past_min_last_horce_body, current_race_data[data_name.past_min_last_horce_body][count] )
            t_instance[data_name.max_horce_true_skill] = \
              lib.minus( max_race_horce_true_skill, current_race_data[data_name.horce_true_skill][count] )
            t_instance[data_name.max_jockey_true_skill] = \
              lib.minus( max_race_jockey_true_skill, current_race_data[data_name.jockey_true_skill][count] )
            t_instance[data_name.max_trainer_true_skill] = \
              lib.minus( max_race_trainer_true_skill, current_race_data[data_name.trainer_true_skill][count] )
            t_instance[data_name.max_up_rate] = \
              lib.minus( max_up_rate, current_race_data[data_name.up_rate][count] )
            t_instance[data_name.max_past_ave_last_horce_body] = \
              lib.minus( max_past_ave_last_horce_body, current_race_data[data_name.past_ave_last_horce_body][count] )
            t_instance[data_name.max_past_max_last_horce_body] = \
              lib.minus( max_past_max_last_horce_body, current_race_data[data_name.past_max_last_horce_body][count] )
            t_instance[data_name.max_past_min_last_horce_body] = \
              lib.minus( max_past_min_last_horce_body, current_race_data[data_name.past_min_last_horce_body][count] )
            t_instance[data_name.min_horce_true_skill] = \
              lib.minus( min_race_horce_true_skill, current_race_data[data_name.horce_true_skill][count] )
            t_instance[data_name.min_jockey_true_skill] = \
              lib.minus( min_race_jockey_true_skill, current_race_data[data_name.jockey_true_skill][count] )
            t_instance[data_name.min_trainer_true_skill] = \
              lib.minus( min_race_trainer_true_skill, current_race_data[data_name.trainer_true_skill][count] )
            t_instance[data_name.min_speed_index] = \
              lib.minus( min_speed_index, current_race_data[data_name.speed_index][count] )
            t_instance[data_name.min_up_rate] = \
              lib.minus( min_up_rate, current_race_data[data_name.up_rate][count] )
            t_instance[data_name.min_past_ave_last_horce_body] = \
              lib.minus( min_past_ave_last_horce_body, current_race_data[data_name.past_ave_last_horce_body][count] )
            t_instance[data_name.min_past_max_last_horce_body] = \
              lib.minus( min_past_max_last_horce_body, current_race_data[data_name.past_max_last_horce_body][count] )
            t_instance[data_name.min_past_min_last_horce_body] = \
              lib.minus( min_past_min_last_horce_body, current_race_data[data_name.past_min_last_horce_body][count] )
            t_instance[data_name.std_horce_true_skill] = std_race_horce_true_skill
            t_instance[data_name.std_jockey_true_skill] = std_race_jockey_true_skill
            t_instance[data_name.std_trainer_true_skill] = std_race_trainer_true_skill
            t_instance[data_name.std_speed_index] = std_speed_index
            t_instance[data_name.std_up_rate] = std_up_rate
            t_instance[data_name.std_past_ave_last_horce_body] = std_past_ave_last_horce_body
            t_instance[data_name.std_past_max_last_horce_body] = std_past_max_last_horce_body
            t_instance[data_name.std_past_min_last_horce_body] = std_past_min_last_horce_body
            t_instance[data_name.three_average] = three_average
            t_instance[data_name.three_difference] = three_difference
            t_instance[data_name.one_rate] = pd.one_rate()
            t_instance[data_name.two_rate] = pd.two_rate()
            t_instance[data_name.three_rate] = three_rate
            t_instance[data_name.match_rank] = match_rank
            t_instance[data_name.passing_regression] = passing_regression
            t_instance[data_name.average_speed] = average_speed
            t_instance[data_name.best_first_passing_rank] = best_first_passing_rank
            t_instance[data_name.best_second_passing_rank] = best_second_passing_rank
            t_instance[data_name.best_weight] = best_weight
            t_instance[data_name.level_score] = min( current_race_data[data_name.level_score][count], 3 )
            t_instance[data_name.before_continue_not_three_rank] = before_continue_not_three_rank
            t_instance[data_name.diff_pace_time] = diff_pace_time
            t_instance[data_name.diff_pace_first_passing] = diff_pace_first_passing
            t_instance[data_name.pace_up] = pace_up
            t_instance[data_name.high_level_score] = high_level_score
            t_instance[data_name.jockey_rank] = jockey_rank_score
            t_instance[data_name.jockey_year_rank] = jockey_year_rank_score
            t_instance[data_name.trainer_rank] = trainer_rank_score
            t_instance[data_name.money] = money_score
            t_instance[data_name.popular_rank] = popular_rank
            t_instance[data_name.before_speed] = before_speed_score
            t_instance[data_name.before_popular] = before_popular
            t_instance[data_name.speed_index] = current_race_data[data_name.speed_index][count]
            t_instance[data_name.speed_index_index] = \
              current_race_data[data_name.speed_index_index].index( current_race_data[data_name.speed_index][count] )
            t_instance[data_name.speed_index_stand] = current_race_data[data_name.speed_index_stand][count]
            t_instance[data_name.predict_netkeiba_pace] = predict_netkeiba_pace
            t_instance[data_name.predict_netkeiba_deployment] = predict_netkeiba_deployment
            t_instance.update( getHorceData.getPredictPace( predict_pace ) )
            t_instance.update( lib.horce_teacher_analyze( current_race_data, t_instance, count ) )

            t_list = self.data_list_create( t_instance )
            lib.dic_append( self.simu_data, race_id, {} )
            self.simu_data[race_id][horce_id] = {}
            self.simu_data[race_id][horce_id]["data"] = t_list
            self.simu_data[race_id][horce_id]["answer"] = { "last_passing_rank": last_passing_rank,
                                                           "first_passing_rank": first_passing_rank,
                                                           "predict_first_passing_rank": predict_first_passing_rank,
                                                           "odds": cd.odds(),
                                                           "popular": cd.popular(),
                                                           "horce_num": cd.horce_number() }

            answer_horce_body.append( answer_corner_horce_body )
            answer_data.append( last_passing_rank - first_passing_rank )
            teacher_data.append( t_list )

        if not len( answer_data ) == 0:
            self.result["answer"].append( answer_data )
            self.result["teacher"].append( teacher_data )
            self.result["year"].append( year )
            self.result["horce_body"].append( answer_horce_body )
            self.result["query"].append( { "q": len( answer_data ), "year": year } )
