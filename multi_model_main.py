from argparse import ArgumentParser

import sekitoba_data_manage as dm
import sekitoba_library as lib
from data_analyze import data_create
from data_analyze import odds_data_create
from rank_learn import rank_learn
from rank_learn import rank_multi_simulation

def main():
    #lib.log.set_name( "nn_simulation_3.log" )
    lib.log.set_name( "multi_rank_learn_1.log" )
    
    parser = ArgumentParser()
    parser.add_argument( "-g", type=bool, default = False, help = "optional" )
    parser.add_argument( "-u", type=bool, default = False, help = "optional" )
    parser.add_argument( "-s", type=bool, default = False, help = "optional" )
    parser.add_argument( "-r", type=bool, default = False, help = "optional" )

    g_check = parser.parse_args().g
    u_check = parser.parse_args().u
    s_check = parser.parse_args().s
    r_check = parser.parse_args().r
    lib.log.write( "g_check:" + str( g_check ) )
    lib.log.write( "u_check:" + str( u_check ) )
    lib.log.write( "s_check:" + str( s_check ) )
    lib.log.write( "r_check:" + str( r_check ) )

    lib.log.write( "rank learn" )
    
    data_storage = {}
    simu_data_storage = {}    
    data_storage["base_data"], simu_data_storage["base_data"] = data_create.main( update = u_check )
    data_storage["odds_data"], simu_data_storage["odds_data"] = odds_data_create.main( update = u_check )

    models = {}
    params = {}
    test_datas = {}
    
    for k in data_storage.keys():
        models[k], _, _ = rank_learn.main( data_storage[k], simu_data_storage[k],
                                          simulation = False, learn_data = True )

    for i in range( 0, 101 ):
        print( i )
        rate = i / 100
        params["base_data"] = rate
        params["odds_data"] = 1 - rate 
        rank_multi_simulation.main( models, simu_data_storage, params )
    
if __name__ == "__main__":
    main()
