import numpy as np
import ga_v3
from random import randrange
from multiprocessing import Pool
import funcion_aoptimizar
from sklearn.cluster import DBSCAN
import sklearn.utils
from time import time
from random import randrange


file = open("best.txt","r+") 
dabest=file.readlines()
file.close()
#open("best.txt", "w").close()

#A=[0.78619835, 0.66167694, 0.22102764, 0.19342388]

# Inputs of the equation.
equation_inputs = [1,1]

# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 10
num_parents_mating = 4


# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.

new_population=[]
for i in range(0,10):
        new_population.append([randrange(1,100),randrange(1,100)])
new_population=np.array(new_population)

##########################################################################################
#HAY VARIAS SOLUCIONES CORRECTAS
##HAY QUE DETERMINAR CUAL ES LA CON FLOTANTE MAXIMO, MINIMO.
## Sería util optimizar el numero de ticks para operar, incluso determinar si hay un horario optimo.
##No hemos ni tocado el parametro de volumen, dado que hay dos fases diferentes, nada impide que haya una tercera.

##########################################################################################
best_outputs = []
num_generations = 100
fitt = open("best_f.txt","r").readline() 
print(fitt)
historical_best_f=float(fitt)
#poblacionales=[]
if len(dabest)==0:
    #print('sin historial de best')
    historical_best_c=[2, 2]
else:
    historical_best_c=dabest[0]

for generation in range(num_generations):
    print("Generation : ", generation)
    print("best fitness : ", historical_best_f)
    # Midiendo fitness
    
    fitness_ = ga_v3.cal_pop_fitness(equation_inputs, new_population)
    best_match_idx = np.where(fitness_ == np.max(fitness_))[0]                
    best_outputs.append(np.max(np.sum(new_population*equation_inputs, axis=1)))
    #print("Best result : ", np.max(fitness_))
    
    #print("Print N°3:\n")
    #print(fitness_)
    # ELITISMO
    #print(fitness_[best_match_idx][0]>historical_best_f)  #por algun motivo da false a esto.
    if  fitness_[best_match_idx][0]>historical_best_f:
        #print('entró')
        historical_best_f=fitness_[best_match_idx][0]
        historical_best_c=new_population[best_match_idx, :][0]
        open("best.txt", "w").close()
        file = open("best.txt","r+") 
        file.write(str((str(historical_best_c[0])+','+str(historical_best_c[1]))))
        file.close()
        open("best_f.txt", "w").close()
        fitt = open("best_f.txt","r+") 
        fitt.write(str(historical_best_f))
        fitt.close()
            
    else:
        print('no entró')
        file = open("best.txt","r").readline() 
        new_population[0]=list(map(float, file.split(",")))  
    
    # Seleccion de adres
    parents = ga_v3.select_mating_pool(new_population, fitness_, 
                                      num_parents_mating)
    
    #print("Print N°4:\n")
    #print(fitness_)

    # Crossover
    offspring_crossover = ga_v3.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    #print("Crossover")
    #print(offspring_crossover)
    #print("Print N°5:\n")
    #print(fitness_)

    # Mutaciones
    offspring_mutation = ga_v3.mutation(offspring_crossover, num_mutations=4)
    #print("Mutation")
    #print(offspring_mutation)

    # Crear nueva población
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("fitness_")
matplotlib.pyplot.show()