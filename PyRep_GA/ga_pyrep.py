import os
import time
import numpy as np
import pickle as pkl
import multiprocessing
from pyrep import PyRep
from argparse import ArgumentParser
from multiprocessing import Process, Manager
from pyrep.backend.utils import suppress_std_out_and_err

class SimulationHelper:
    def __init__(self, pop_size):
        self.num_snakes = 5
        self.PROCESSES = 10
        self.sim_loops = pop_size//(self.num_snakes*self.PROCESSES)

    def run(self, A, w, p, cur_id, return_dict, headless=True):
        my_A = A[cur_id*self.num_snakes:(cur_id+1)*self.num_snakes]
        my_w = w[cur_id*self.num_snakes:(cur_id+1)*self.num_snakes]
        my_p = p[cur_id*self.num_snakes:(cur_id+1)*self.num_snakes]

        with suppress_std_out_and_err():
            prs = PyRep()
            prs.launch('pyrep_testing_scene.ttt', headless=headless)
            prs.start()

            position_initial = []
            for i in range(self.num_snakes):
                try:
                    _  = prs.script_call("run_on_snake@Snake1#"+str(9*i),1,[0,0,0],[my_A[i],my_w[i],my_p[i]],['yes','its','working'],[])
                    position_initial.append(np.array(prs.script_call("get_position@Snake1#"+str(9*i),1,[],[],['yes','its','working'],[])[1]))
                except:
                    print(my_A,my_w,my_p)

            for _ in range(100):
                prs.step()
            
            dist = []
            position_final = []
            for i in range(self.num_snakes):
                position_final.append(np.array(prs.script_call("get_position@Snake1#"+str(9*i),1,[],[],['yes','its','working'],[])[1]))
                diff = abs(position_final[i]-position_initial[i])
                dist.append(diff[1]**2+diff[0]**2)
            return_dict[cur_id] = dist
            prs.stop()
            prs.shutdown()

    def run_simulation(self, A, w, p):

        print("Starting Generation Simulation")
        all_distances = []
        sim_start = time.time()
        for i in range(self.sim_loops):
            headless = np.random.choice([True,False],size=(1,),replace=True,p=[0.7,0.3])
            start = time.time()
            print("\t Loop {}/{}".format(i+1,self.sim_loops), end='\t')
            cA = A[i*self.num_snakes*self.PROCESSES:(i+1)*self.num_snakes*self.PROCESSES] 
            cw = w[i*self.num_snakes*self.PROCESSES:(i+1)*self.num_snakes*self.PROCESSES]
            cp = p[i*self.num_snakes*self.PROCESSES:(i+1)*self.num_snakes*self.PROCESSES]
            return_dict = multiprocessing.Manager().dict()
            distances = np.zeros((self.num_snakes,self.PROCESSES))
            processes = [Process(target=self.run, args=(cA,cw,cp,id,return_dict,headless)) for id in range(self.PROCESSES)]
            [pr.start() for pr in processes]
            [pr.join() for pr in processes]
            for k in return_dict:
                distances[:,k] = return_dict[k]
            all_distances.extend(distances.ravel().tolist())
            print("Time Taken : {} secs".format(time.time()-start))
        print("Done Generation Simulation in {} secs".format(time.time()-sim_start))
        return all_distances

class GA:
    def __init__(self, sol_per_pop, num_mating_parents, num_generations):
        loaded_data = np.load('data/points.npz', allow_pickle=True)
        params = pkl.load(open('data/params_final.pkl','rb'))
        self.save_best_snakes = []
        self.actual_soln = np.array([params['amp']/100.,params['omega']*100.,params['phase']])
        self.num_weights = self.actual_soln.shape[0]
        self.sol_per_pop = sol_per_pop
        self.num_mating_parents = num_mating_parents
        self.num_generations = num_generations
        self.equation_inputs = np.array(loaded_data['cx'][0])
        self.pop_size = (self.sol_per_pop,self.num_weights)
        self.new_population = np.random.uniform(-1.0,1.0,self.pop_size)
        self.sim_help = SimulationHelper(self.sol_per_pop)
    
    def function_valuation(self, weights):
        x = self.equation_inputs
        y = weights[0]*np.sin(weights[1]*x + weights[2])
        return y
    
    def calc_pop_fitness(self, method=1):
        if method == 1:
            true_val = self.function_valuation(self.actual_soln).reshape((-1,1))
            pop_val = []
            for i in range(len(self.new_population)):
                #normalize both distance and Square Error and add alpha beta params
                num = self.function_valuation(self.new_population[i,:]).reshape((-1,1)) - true_val
                pop_val.append(np.sum(np.square(num))-self.dist[i])
            self.fitness = np.array(pop_val).reshape((-1,1))
        else:
            true_val = self.function_valuation(self.actual_soln).reshape((-1,1))
            pop_val = []
            self.dist = np.array(self.dist)
            self.dist = ((self.dist-min(self.dist))/max(self.dist)-min(self.dist)).tolist()
            for i in range(len(self.new_population)):
                #normalize both distance and Square Error and add alpha beta params
                num = self.function_valuation(self.new_population[i,:]).reshape((-1,1)) - true_val
                den = self.function_valuation(self.new_population[i,:]).reshape((-1,1)) + true_val
                pop_val.append(0.6*np.sum(np.square(num/den))-0.4*self.dist[i])
            self.fitness = np.array(pop_val).reshape((-1,1))

    def select_mating_pool(self):
        parents = np.empty((self.num_mating_parents, self.new_population.shape[1]))
        for parent_num in range(self.num_mating_parents):
            max_fitness_idx = np.where(self.fitness == np.min(self.fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = self.new_population[max_fitness_idx, :]
            self.fitness[max_fitness_idx] = 99999999999
        self.parents = parents

    def crossover(self, offspring_size):
        self.offspring = np.empty(offspring_size)
        crossover_point = np.uint8(offspring_size[1]/2)
        for k in range(offspring_size[0]):
            parent1_idx = k%self.parents.shape[0]
            parent2_idx = (k+1)%self.parents.shape[0]
            self.offspring[k, 0:crossover_point] = self.parents[parent1_idx, 0:crossover_point]
            self.offspring[k, crossover_point:] = self.parents[parent2_idx, crossover_point:]

    def mutation(self):
        for idx in range(self.offspring.shape[0]):
            random_value = np.random.uniform(-1.0, 1.0, 1)
            rand_ind = np.random.randint(0,self.num_weights,1)[0]
            self.offspring[idx, rand_ind] = self.offspring[idx, rand_ind] + random_value
    
    def loop_run(self):
        for generation in range(self.num_generations):
            print("-"*70)
            print("Generation : {}/{}".format(generation+1,self.num_generations))
            A,w,p = np.array([s[0] for s in self.new_population]), np.array([s[1] for s in self.new_population]), np.array([s[2] for s in self.new_population])
            self.dist = self.sim_help.run_simulation(A,w,p)
            self.calc_pop_fitness(method=1)
            self.save_best_snakes.append(self.new_population[np.argsort(self.fitness)][:5])
            self.select_mating_pool()
            self.crossover(offspring_size=(self.pop_size[0]-self.parents.shape[0], self.num_weights))
            self.mutation()
            self.new_population[0:self.parents.shape[0], :] = self.parents
            self.new_population[self.parents.shape[0]:, :] = self.offspring
            self.best_match_idx = np.where(self.fitness == np.min(self.fitness))
            print("Best result : ", np.min(self.fitness))
            print("Best solution : ", self.new_population[self.best_match_idx[0][0], :])
            print("Actual solution : ", self.actual_soln)
        self.calc_pop_fitness(method=1)
        self.best_match_idx = np.where(self.fitness == np.min(self.fitness))
        print("-"*70)
        print("Best solution : ", self.new_population[self.best_match_idx[0][0], :])
        print("Best solution fitness : ", self.fitness[self.best_match_idx[0][0]][0])
        pkl.dump(self.save_best_snakes, open('checkpoint/best_snakes.pkl','wb'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--pop_size', type=int, required=True, help='Population Size')
    parser.add_argument('-m', '--no_mating_parents', type=int, required=True,  help='Number of mating parents')
    parser.add_argument('-n', '--no_of_generations', type=int, required=True, help='Number of generations')
    parser.add_argument('-c', '--continue_training', type=str, default="no", help="Continue previous training")
    args = parser.parse_args()
    
    sol_per_pop, num_mating_parents, num_generations = args.pop_size, args.no_mating_parents, args.no_of_generations
    assert sol_per_pop >= 50, "Population size should be atleast 100"
    assert sol_per_pop%50 == 0, "Population size must be a multiple of 100"
    assert num_mating_parents < sol_per_pop, "Mating parents need to be lesser than total population size"
    cont = args.continue_training
    
    try:
        os.mkdir('checkpoint')
    except:
        pass

    if cont == "no":
        ga = GA(sol_per_pop, num_mating_parents, num_generations)
        ga.loop_run()
        pkl.dump(ga, open('checkpoint/ga_object.pkl','wb'))
    else:
        try:
            ga = pkl.load(open('checkpoint/ga_object.pkl','rb'))
            print("Successfully loaded from previous checkpoint!")
            ga.num_generations = num_generations
            ga.loop_run()
            pkl.dump(ga, open('checkpoint/ga_object.pkl','wb'))
        except:
            print("Failed to load from previous checkpoint!")
