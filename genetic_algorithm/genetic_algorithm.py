import math
import random

class GA:
    def __init__( self, element, population ):
        self.element = element
        self.population = population
        self.parent = []
        self.scores = []
        self.best_population = None
        self.best_score = -1
        self.mutation_rate = 0.01

        for i in range( 0, self.population ):
            t = [0] * self.element
            
            for r in range( 0, self.element ):
                if random.random() < 0.5:
                    t[r] = 1

            self.parent.append( t )

    def get_best( self ):
        return self.best_population, self.best_score

    def get_parent( self ):
        return self.parent

    def scores_set( self, scores ):
        self.scores = scores

    def softmax( self ):
        result = []
        sum_value = 0
        max_value = max( self.scores )

        for i in range( 0, self.population ):
            sum_value += math.exp( self.scores[i] - max_value )


        for i in range( 0, self.population ):
            result.append( math.exp( self.scores[i] - max_value ) / sum_value )

        return result

    def roulette( self, softmax_scores ):
        result = []
        before = -1        
        
        while 1:
            if len( result ) == 2:
                break
            
            get_num = -1
            ru = random.random()
            sum_value = 0

            for i in range( 0, len( softmax_scores ) ):
                sum_value += softmax_scores[i]

                if ru < sum_value:
                    get_num = i
                    break

            if get_num == -1:
                get_num = len( softmax_scores ) - 1

            if not before == get_num:
                result.append( get_num )
                before = get_num                

        return result[0], result[1]

    def two_crossing( self, parent1, parent2 ):
        two_point = []
        before = -1

        while 1:
            if len( two_point ) == 2:
                break
            
            p = random.randint( 1, self.element - 2 )

            if not before == p:
                before = p
                two_point.append( p )

        min_point = min( two_point )
        max_point = max( two_point )
        child = parent1[0:min_point] + parent2[min_point:max_point] + parent2[max_point:self.element]

        return child

    def mutation( self, data ):
        for i in range( 0, self.element ):
            if random.random() < self.mutation_rate:
                if data[i] == 0:
                    data[i] = 1
                else:
                    data[i] = 0

        return data        
        
    def next_genetic( self ):
        result = []
        softmax_scores = self.softmax()

        for i in range( 0, self.population ):
            if self.best_score < self.scores[i]:
                self.best_score = self.scores[i]
                self.best_population = self.parent[i]

        for i in range( 0, self.population ):
            point1, point2 = self.roulette( softmax_scores )
            child = self.two_crossing( self.parent[point1], self.parent[point2] )
            child = self.mutation( child )
            result.append( child )

        self.parent = result            
        
    
            
