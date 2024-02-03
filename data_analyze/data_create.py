import math
from tqdm import tqdm
from mpi4py import MPI

from data_analyze.once_data import OnceData
import sekitoba_library as lib
import sekitoba_data_manage as dm
import sekitoba_data_create as dc

def key_list_search( rank, size, key_list ):
    n = int( len( key_list ) / ( size - 1 ) )
    s1 = int( ( rank - 1 ) * n )

    if not rank + 1 == size:
        s2 = s1 + n
    else:
        s2 = len( key_list ) + 1

    return key_list[s1:s2]

def main( update = False ):
    result = None

    comm = MPI.COMM_WORLD   #COMM_WORLDは全体
    size = comm.Get_size()  #サイズ（指定されたプロセス（全体）数）
    rank = comm.Get_rank()  #ランク（何番目のプロセスか。プロセスID）
    name = MPI.Get_processor_name() #プロセスが動いているノードのホスト名
    
    if not update:
        if rank == 0:
            result = dm.pickle_load( lib.name.data_name() )
            simu_data = dm.pickle_load( lib.name.simu_name() )
            update_check = False
            
            if result == None:
                update_check =  True

            for i in range( 1, size ):
                comm.send( update_check, dest = i, tag = 1 )

            if not update_check:
                return { "data": result, "simu": simu_data }
                
        else:
            update_check = comm.recv( source = 0, tag = 1 )

            if not update_check:
                return None

    if rank == 0:
        result = {}
        dm.dl.local_keep()
        dm.dl.data_clear()
        
        for i in range( 1, size ):
            comm.send( True, dest = i, tag = 1 )

        result["simu"] = {}
        result["data"] = { "answer": [], "teacher": [], "query": [], "year": [], "level": [], "diff": [], "horce_body": [] }
        
        for i in range( 1, size ):
            file_name = comm.recv( source = i, tag = 2 )
            instance = dm.pickle_load( file_name )
            dm.pickle_delete( file_name )
            result["simu"].update( instance["simu"] )

            for k in instance["data"].keys():
                result["data"][k].extend( instance["data"][k] )

        dm.pickle_upload( lib.name.data_name(), result["data"] )
        dm.pickle_upload( lib.name.simu_name(), result["simu"] )
    else:
        ok = comm.recv( source = 0, tag = 1 )
        od = OnceData()
        print( "start rank:{}".format( rank ) )
        key_list = key_list_search( rank, size, list( od.race_data.keys() ) )

        if rank == 1:
            for k in tqdm( key_list ):
                od.create( k )
        else:
            for k in key_list:
                od.create( k )

        file_name = str( rank ) + "-instance.pickle"
        dm.pickle_upload( file_name, { "data": od.result, "simu": od.simu_data } )
        comm.send( file_name, dest = 0, tag = 2 )
        result = None

        if rank == 1:
            od.score_write()

    dm.dl.data_clear()
    return result    
