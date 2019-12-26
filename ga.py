#!/usr/bin/env python
# coding: utf-8

# In[2]:

#from pygame import mixer
#mixer.init()
#mixer.music.load('C:/Users/pneira/Pictures/metalocalypse_guitone.mp3')
#mixer.music.play() 
import numpy
import funcion_aoptimizar
from sklearn.cluster import DBSCAN
import sklearn.utils
from multiprocessing import Pool
num_processors = 8
p=Pool(processes = num_processors)
from random import randrange
#@jit
def cal_pop_fitness(equation_inputs, pop):
    output = p.starmap(funcion_aoptimizar.func, pop)
    output=numpy.array(output)
    return output

#@jit
def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents
#@jit
def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
#@jit
def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.sign(numpy.random.uniform(-1, 1, 1))
            rand=randrange(100)
            if offspring_crossover[idx, gene_idx] + rand*random_value>  0:
                #print(offspring_crossover)
                offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] +rand*random_value
            else:
                pass    
            
            gene_idx = gene_idx + mutations_counter
    mixer.init()
    mixer.music.load('C:/Users/pneira/Pictures/metalocalypse_guitone.mp3')
    mixer.music.play()
    return offspring_crossover
 # Load the popular external library


# In[ ]:




