import random

import sekitoba_library as lib
import sekitoba_data_manage as dm
from machine_learn_torch import nn

def data_check( data ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    ma = -1
    mi = 100
    
    for i in range( 0, len( data["answer"] ) ):
        ma = max( data["answer"][i], ma )
        mi = min( data["answer"][i], mi )
        
        if data["year"][i] == "2020":
            result["test_teacher"].append( data["teacher"][i] )
            result["test_answer"].append( data["answer"][i] )
        else:
            result["teacher"].append( data["teacher"][i] )
            result["answer"].append( [ data["answer"][i] ] )

    return result

def main( data, GPU = False ):
    units = {}
    learn_data = data_check( data )    
    #learn_data, a_units = batch_data_check( data )
    n_units = len( data["teacher"][0] )
    print( n_units )
    units["n"] = n_units
    #units["a"] = a_units

    dm.pickle_upload( "last_horce_body_units.pickle", units )
    model = nn.LastStrightNN( n_units )
    model = nn.main( learn_data, model, GPU )
    dm.model_upload( "last_horce_body_model.pth", model )
    
    return model, units
