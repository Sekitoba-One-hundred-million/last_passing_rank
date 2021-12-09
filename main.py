import torch
import numpy as np
from argparse import ArgumentParser

import sekitoba_data_manage as dm
import sekitoba_library as lib
from data_analyze import data_create
from machine_learn_torch import learn
from machine_learn_torch.nn import LastStrightNN

def model_data_create():
    result = {}
    units = dm.pickle_load( "last_horce_body_units.pickle" )
    model = LastStrightNN( units["n"] )
    model = dm.model_load( "last_horce_body_model.pth", model )
    model.eval()
    simu_data = dm.pickle_load( "last_horce_body_simu_data.pickle" )
    
    for k in simu_data.keys():
        lib.dic_append( result, k, {} )
        for kk in simu_data[k].keys():
            data = simu_data[k][kk]["data"]
            predict_answer = model.forward( torch.tensor( np.array( [ data ], dtype = np.float32 ) ) ).detach().numpy()
            result[k][kk] = predict_answer[0][0]

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
        model_data_create()
        return
    
    data, simu_data = data_create.main( update = u_check )
    learn.main( data, GPU = g_check )
    #lib.log.write( "rank learn" )
    #rank_model = rank_learn.main( data, simu_data )
    
if __name__ == "__main__":
    main()
