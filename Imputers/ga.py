from dataclasses import dataclass
import numpy as np
import random
from typing import *
from svr_impute import SVRImputer
from fcm_impute import FCMImputer

@dataclass
class GeneticAlgorithm:
    fcm: None
    svr: None
    pop_size: int = 50
    generation: int = 50
    crossover_prob = 0.6
    mutation_prob = 0.01
    def __post_init__(self):
        self.population = self._initialize_population()

    def _initialize_population(self):
        return [[random.randrange(2, 10), random.uniform(1.1, 10.0)] for _ in range(self.pop_size)]

    def _fitness(self, params):
        self.fcm.c = params[0]
        self.fcm.m = params[1]
        x = self.fcm.impute()
        y = self.svr.impute()
        fitness = np.power((x - y), 2).sum()
        return fitness
    
    def _crossover(self, chromosome_1, chromosome_2):
        prob = random.randrange(0, 101)
        if prob <= self.crossover_prob * 100:
            index = random.randrange(0, 2)
            chromosome_1[index], chromosome_2[index] = chromosome_2[index], chromosome_1[index]

        return [chromosome_1, chromosome_2]
    
    def _selection(self):
        '''
        parent selection
        '''
        index = random.sample(range(self.pop_size - 1), 2)
        parent_1 = self.population[index[0]]
        parent_2 = self.population[index[1]]

        del self.population[index[0]]
        del self.population[index[1]]

        return [parent_1, parent_2]
    
    def _mutation(self, chromosome):
        prob = random.randrange(0, 101)
        if prob <= self.mutation_prob * 100:
            index = random.randrange(0, 2)
            if index == 0:
                val = random.randrange(2, 10)
                chromosome[index] = val
            else:
                val = random.uniform(1.1, 10.0)
                chromosome[index] = val

        return chromosome
    
    def run(self):
        for _ in range(self.generation):
            parents = self._selection()
            parents = self._crossover(parents[0], parents[1])
            
            index = random.randrange(0, 2)
            parents[index] = self._mutation(parents[index])
            self.population.append(parents[0])
            self.population.append(parents[1])

        self.population.sort(key=self._fitness)
        return self.population[0][0], self.population[0][1]
    
    
def random_data(seed = 42, upperbound = 0.5, num = 100, features = 2):   
    '''
    Generate random data
    '''
    np.random.seed(seed)
    data = np.random.rand(num, features)
    # data[data < upperbound] = np.nan
    return data

    
if __name__ == "__main__":
    print([[random.randrange(2, 10), random.uniform(1.1, 10.0)] for _ in range(5)])
    data = random_data(42, 0.1, 5, 8)
    data[0:2, [1, 3, 6]] = np.nan
    
    
    svrImputer = SVRImputer(data = data)
    fcmImputer = FCMImputer(data = data, num_clusters = 3)
    
    ga = GeneticAlgorithm(fcm = fcmImputer, svr = svrImputer)
    print(ga.run())