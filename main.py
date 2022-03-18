import torch
import numpy as np
from argparse import ArgumentParser
from mpi4py import MPI

import sekitoba_data_manage as dm
import sekitoba_library as lib

from data_analyze import data_create
#from machine_learn_torch import learn
#from machine_learn_torch.nn import LastStrightNN
from tree_learn import learn

def tree_model_data_create():
    result = {}
    model = dm.pickle_load( "last_horce_body_lightbgm_model.pickle" )
    simu_data = dm.pickle_load( "last_horce_body_simu_data.pickle" )
    
    for k in simu_data.keys():
        data = []
        lib.dic_append( result, k, {} )
        
        for kk in simu_data[k].keys():
            instance = {}
            pah = max( model.predict( np.array( [ simu_data[k][kk]["data"] ] ) )[0], 0 )
            pah = int( pah * 2 ) / 2
            
            instance["predict"] = pah
            instance["answer"] = simu_data[k][kk]["answer"]
            instance["kk"] = kk
            data.append( instance )

        sort_list = sorted( data, key=lambda x:x["predict"] )        
        
        for i in range( 0, len( sort_list ) ):
            pah = sort_list[i]["predict"] - sort_list[0]["predict"]
            kk = sort_list[i]["kk"]
            result[k][kk] = max( pah, 0 )

    dm.pickle_upload( "last_horce_body.pickle", result )

def main():
    #lib.log.set_name( "nn_simulation_3.log" )
    #lib.log.set_name( "rank_learn_9.log" )
    
    parser = ArgumentParser()
    parser.add_argument( "-g", type=bool, default = False, help = "optional" )
    parser.add_argument( "-u", type=bool, default = False, help = "optional" )
    parser.add_argument( "-s", type=bool, default = False, help = "optional" )
    parser.add_argument( "-r", type=bool, default = False, help = "optional" )

    g_check = parser.parse_args().g
    u_check = parser.parse_args().u
    s_check = parser.parse_args().s
    r_check = parser.parse_args().r

    if s_check:
        if rank == 0:
            tree_model_data_create()            
        return

    data = data_create.main( update = u_check )

    if not data == None:        
        learn.main( data["data"], data["simu"] )
        tree_model_data_create()

    MPI.Finalize()
    
if __name__ == "__main__":
    main()
