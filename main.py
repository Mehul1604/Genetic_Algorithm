import numpy as np
from config import SECRET_KEY
from client import *
import random
import math
import os
import pandas as pd
from tabulate import tabulate

#hyper parameters
POP_SIZE = 16
MIN_WEIGHT = -10
MAX_WEIGHT = 10
MUT_RANGE = 0.1
WT = 0.25
MUT_PROB = 0.3
NUM_GENS = 60
overfit_vector = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
LOWEST_POSS = [13510723304.19212, 368296592820.6967]
LOWEST_POSS_SUM = 381807316124.88885


#Initialize population by mutating a vector with high probability and range
def initialize_population(starting_vector):
    population = np.zeros((POP_SIZE , MAX_DEG) , dtype='float64')
    for i in range(len(population)):
        population[i] = mutate(starting_vector , 0.01 , 0.9)
    
    return population



# Get the training and validation errors
def get_err(vector):
    train_valid_error = get_errors(SECRET_KEY , list(vector))
    return train_valid_error


#Relation of fitness to errors
def calculate_fitness(errors):
    return errors[0]*WT + errors[1]



# Calculate fitness and errors of the population , bind them together and sort them rank wise
def fitness_function(population):
    training_errors = np.zeros(len(population))
    validation_errors = np.zeros(len(population))
    fitness_array = np.zeros(len(population))
    generation = []
    for i in range(len(population)):
        chromosome = population[i]
        error = get_err(chromosome)
        training_errors[i] = error[0]
        validation_errors[i] = error[1]
        fitness_array[i] = calculate_fitness(error)
        generation.append((chromosome , training_errors[i] , validation_errors[i] , fitness_array[i]))
    
    
    generation.sort(key=lambda x:x[3])
    generation = np.array(generation)
    return generation
    

# Select the top chromosomes from the population as parents
def select_parent_pool(generation):
    top_selection = int(math.ceil(len(generation) / 4)) + 1
    parent_generation = generation[:top_selection]
    return parent_generation
    

# From the top pool make an array of fathers and mothers by random weighted choice
def make_parents(parent_pool):
    n = len(parent_pool)
    total = n*(n+1)/2
    weight_arr = np.array(range(1,n+1))[::-1]
    weight_arr = weight_arr/total
    print('WEIGHTS ARE')
    print(weight_arr)
    num_pars = int(POP_SIZE/2)
    parents = []
    while num_pars:
        pars = random.choices(list(enumerate(parent_pool)) , k=2 , weights=weight_arr)
        parents.append(pars)
        num_pars -= 1
    
    parents = np.array(parents)
    return parents
    


# Binary crossover
def binary_crossover(father_chromosome , mother_chromosome):

    u = random.random() 
    n_c = 3 #factor would be changed after some iterations
        
    if (u < 0.5):
        beta = (2 * u)**((n_c + 1)**-1)
    else:
        beta = ((2*(1-u))**-1)**((n_c + 1)**-1)

    child1 = 0.5*((1 + beta) * father_chromosome + (1 - beta) * mother_chromosome)
    child2 = 0.5*((1 - beta) * father_chromosome + (1 + beta) * mother_chromosome)
    child1 = np.array(child1)
    child2 = np.array(child2)

    return (child1, child2)



# Make an array of children vectors
def make_children(parents):
    children_population = []
    for i in range(len(parents)):
        father = parents[i][0][1]
        mother = parents[i][1][1]
        father_chrom = father[0]
        mother_chrom = mother[0]
        child_pair = binary_crossover(father_chrom , mother_chrom)
        children_population.append(child_pair[0])
        children_population.append(child_pair[1])
    
    children_population = np.array(children_population)
    return children_population


# Mutation Function which takes range and probability 
def mutate(vector  ,mut_range , mut_prob):
    mutated_vect = np.copy(vector)
    for i in range(len(mutated_vect)):
        w = mutated_vect[i]
        reduction_fact = random.uniform(-mut_range,mut_range)
        reduction = w*reduction_fact
        if random.random() <= mut_prob:
            w = w + reduction
        mutated_vect[i] = w
    
    return mutated_vect


# make mutated children and calculate their fitness
def make_offspring_population(children_population , mut_range , mut_prob):
    mutated_children = []
    for c in children_population:
        mutated_children.append(mutate(c , mut_range , mut_prob))
    
    mutated_children = np.array(mutated_children)
    return mutated_children



def get_mutated_fitness(mutated_children):
    mutated_generation = fitness_function(mutated_children)
    return mutated_generation


#choose the next population
def make_next_gen(parent_pool , mutated_generation):
    new_pool = np.concatenate((parent_pool[:3] , mutated_generation))
    new_pool = new_pool[np.argsort(new_pool[:,3])]
    new_generation = new_pool[:POP_SIZE]
    return new_generation


#trace all the generations in a text file
def write_logs(all_gens , all_parents , all_children , all_mutated):
    os.makedirs('./Logs' , exist_ok=True)
    fd = open('./Logs/final.txt' , 'w')
    for gen in range(len(all_gens)):
        fd.write('GENERATION: {}\n\n'.format(gen))
        
        fd.write('INITIAL POPULATION\n')
        gen_headers = ["Number" , "Chromosome" , "Training Error" , "Validation Error" , "Fitness (Train + Valid)"]
        gen_table = []
        for i in range(len(all_gens[gen])):
            vector = all_gens[gen][i][0]
            train_err = all_gens[gen][i][1]
            valid_err = all_gens[gen][i][2]
            fitness = all_gens[gen][i][3]
            gen_row = [i , vector , train_err , valid_err , fitness]
            gen_table.append(gen_row)
        
        fd.write(tabulate(gen_table, gen_headers, tablefmt="fancy_grid"))
        fd.write('\n\n')
        
        fd.write('PARENTS SELECTED\n')
        parent_headers = ["Number" , "Parent" , "Index" , "Chromosome" , "Training Error" , "Validation Error" , "Fitness"]
        parent_table = []
        for i in range(len(all_parents[gen])):
            specimen = all_parents[gen][i][0][1]
            father_ind = all_parents[gen][i][0][0]
            vector = specimen[0]
            train_err = specimen[1]
            valid_err = specimen[2]
            fitness = specimen[3]
            parent_row = [i  , 'Father', father_ind, vector , train_err , valid_err , fitness]
            parent_table.append(parent_row)
            
            specimen = all_parents[gen][i][1][1]
            mother_ind = all_parents[gen][i][1][0]
            vector = specimen[0]
            train_err = specimen[1]
            valid_err = specimen[2]
            fitness = specimen[3]
            parent_row = [i  , 'Mother',mother_ind, vector , train_err , valid_err , fitness]
            parent_table.append(parent_row)
        
        fd.write(tabulate(parent_table, parent_headers, tablefmt="fancy_grid"))
        fd.write('\n\n')
        
        fd.write('CROSSOVER CHILDREN\n')
        child_headers = ["Number" , "Parent Index", "Chromosome"]
        child_table = []
        for i in range(len(all_children[gen])):
            vector = all_children[gen][i]
            child_row = [i , (int(i/2)), vector]
            child_table.append(child_row)
        
        fd.write(tabulate(child_table, child_headers, tablefmt="fancy_grid"))
        fd.write('\n\n')
        
        fd.write('MUTATED CHILDREN\n')
        mutated_headers = ["Number" , "Chromosome"]
        mutated_table = []
        for i in range(len(all_mutated[gen])):
            vector = all_mutated[gen][i]
            mutated_row = [i , vector]
            mutated_table.append(mutated_row)
        
        fd.write(tabulate(mutated_table, mutated_headers, tablefmt="fancy_grid"))
        fd.write('\n\n')
        fd.write('---------------------------------------------------------------------------------------------------------------------------------------\n\n')
        


#initialize population using overfit vector
INITIAL_POPULATION = initialize_population(overfit_vector)

#getting initial generation with fitness scores
INITIAL_GENERATION = fitness_function(INITIAL_POPULATION)


current_generation = np.copy(INITIAL_GENERATION)

#arrays to store all generations for logs
all_gens = []
all_parents = []
all_children = []
all_mutated = []


#MAIN LOOP
for iter in range(NUM_GENS):
    all_gens.append(current_generation)
    
    #parent pool
    parent_pool = select_parent_pool(current_generation)

    #parents
    parents = make_parents(parent_pool)
    all_parents.append(parents)
    
    #child vectors
    children_chromosomes = make_children(parents)
    all_children.append(children_chromosomes)
    
    #mutated children
    offspring_population = make_offspring_population(children_chromosomes , MUT_RANGE , MUT_PROB)
    all_mutated.append(offspring_population)
    offspring_generation = get_mutated_fitness(offspring_population)

    #making next generation    
    next_generation = make_next_gen(parent_pool , offspring_generation)
    current_generation = np.copy(next_generation)
    last_generation = np.copy(current_generation)
    
    
#saving logs
write_logs(all_gens , all_parents , all_children , all_mutated)





