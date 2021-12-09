import numpy as np
import math
import random
from torch import nn, optim, cuda
import torch.nn.functional as F
import torch

from tqdm import tqdm

import sekitoba_data_manage as dm
import sekitoba_library as lib

class LastStrightNN( nn.Module ):

    def __init__( self, n ):
        super( LastStrightNN, self ).__init__()
        self.l1 = nn.Linear( n, n )
        self.l2 = nn.Linear( n, n )
        self.l3 = nn.Linear( n, n )
        self.l4 = nn.Linear( n, n )
        self.l5 = nn.Linear( n, n )
        self.l6 = nn.Linear( n, n )
        self.l7 = nn.Linear( n, n )
        self.l8 = nn.Linear( n, n )
        self.l9 = nn.Linear( n, n )
        self.l10 = nn.Linear( n, n )
        self.l11 = nn.Linear( n, n )
        self.l12 = nn.Linear( n, n )
        self.l13 = nn.Linear( n, n )
        self.l14 = nn.Linear( n, n )
        self.l15 = nn.Linear( n, n )
        self.l16 = nn.Linear( n, n )
        self.l17 = nn.Linear( n, n )
        self.l18 = nn.Linear( n, n )
        self.l30 = nn.Linear( n, 1 )
        
        self.b1 = nn.BatchNorm1d( n )
        self.b2 = nn.BatchNorm1d( n )
        self.b3 = nn.BatchNorm1d( n )
        self.b4 = nn.BatchNorm1d( n )
        self.b5 = nn.BatchNorm1d( n )
        self.b6 = nn.BatchNorm1d( n )
        self.b7 = nn.BatchNorm1d( n )
        self.b8 = nn.BatchNorm1d( n )
        self.b9 = nn.BatchNorm1d( n )
        self.b10 = nn.BatchNorm1d( n )
        self.b11 = nn.BatchNorm1d( n )
        self.b12 = nn.BatchNorm1d( n )
        self.b13 = nn.BatchNorm1d( n )
        self.b14 = nn.BatchNorm1d( n )
        self.b15 = nn.BatchNorm1d( n )
        self.b16 = nn.BatchNorm1d( n )
        self.b17 = nn.BatchNorm1d( n )
            
    def forward( self, x ):
        h1 = F.relu( self.l1( x ) )
        h2 = F.relu( self.b1( self.l2( h1 ) ) )
        h3 = F.relu( self.b2( self.l3( h2 ) ) )
        h4 = F.relu( self.b3( self.l4( h3 + h2 ) ) )
        h5 = F.relu( self.b4( self.l5( h4 ) ) )
        h6 = F.relu( self.b5( self.l6( h5 + h4 ) ) )
        h7 = F.relu( self.b6( self.l7( h6 ) ) )
        h8 = F.relu( self.b7( self.l8( h7 + h6 ) ) )
        h9 = F.relu( self.b8( self.l9( h8 ) ) )
        h10 = F.relu( self.b9( self.l10( h9 + h8 ) ) )
        h11 = F.relu( self.b10( self.l11( h10 ) ) )
        h12 = F.relu( self.b11( self.l12( h11 + h10 ) ) )
        h13 = F.relu( self.b12( self.l13( h12 ) ) )
        h14 = F.relu( self.b13( self.l14( h13 + h12 ) ) )
        h15 = F.relu( self.b14( self.l15( h14 ) ) )
        h16 = F.relu( self.b15( self.l16( h15 + h14 ) ) )
        h17 = F.relu( self.b16( self.l17( h16 ) ) )
        h18 = F.relu( self.b17( self.l18( h17 + h16 ) ) )
        h30 = self.l30( h18 )
        return h30

def test( test_teacher, test_answer, model ):
    model.eval()
    predict_answer = model.forward( torch.tensor( np.array( test_teacher, dtype = np.float32 ) ) ).detach().numpy()
    model.train()    
    count = 0
    diff_check = 0

    for i in range( 0, len( test_answer ) ):
        diff_check += abs( predict_answer[i][0] - test_answer[i] )
        
    diff_check /= len( test_teacher )
    return diff_check
    
    
def main( data, model, GPU = False ):
    teacher_data = data["teacher"]
    answer_data = data["answer"]
    test_teacher = data["test_teacher"]
    test_answer = data["test_answer"]
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    device = "cuda" if cuda.is_available() else "cpu"

    if GPU:
        if cuda.is_available():
            print( "GPU使用" )
        else:
            print( "GPU使用しようとしましたが失敗しました. 代わりにCPUを使用します." )
    else:
        print( "CPU使用" )
    xp = np

    N = len( teacher_data )
    epoch = 300
    batch_size = 2048
    teacher_data = torch.from_numpy( xp.array( teacher_data, dtype = xp.float32 ) ).to( device )
    answer_data = torch.from_numpy( xp.array( answer_data, dtype = xp.float32 ) ).to( device )
    print( "learn start" )
    #lib.log.write( "nn learn" )
    
    for e in range( 0, epoch ):
        model = model.to( device )
        all_loss = 0
        data_list = list( range( 0, N ) )
        random.shuffle( data_list )
        
        for i in range( 0, int( N / batch_size ) ):
            b = i * batch_size
            optimizer.zero_grad()
            y = model.forward( teacher_data[data_list[b:b+batch_size]] )
            loss = F.mse_loss( y, answer_data[data_list[b:b+batch_size]] )
            #loss = F.l1_loss( y, answer_data[data_list[b:b+batch_size]] )
            #loss = F.cross_entropy( y, answer_data[data_list[b:b+batch_size]].type( torch.long ) )
            all_loss += loss.detach().numpy()
            loss.backward()
            optimizer.step()

        diff = test( test_teacher, test_answer, model )
        log_text =  "学習:{}回 誤差馬身:{}秒 loss:{}".format( e + 1, diff, all_loss / int( N / batch_size ) )
        print( log_text )
        #lib.log.write( log_text )

    return model
