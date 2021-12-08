from argparse import ArgumentParser

import sekitoba_data_manage as dm
import sekitoba_library as lib
from data_analyze import data_create
from rank_learn import rank_learn

def main():
    #lib.log.set_name( "nn_simulation_3.log" )
    lib.log.set_name( "rank_learn_9.log" )
    
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
    
    data, simu_data = data_create.main( update = u_check )
    
    lib.log.write( "rank learn" )
    rank_model = rank_learn.main( data, simu_data )
    
if __name__ == "__main__":
    main()
