import os
import re
import sim
import sys
import math
import time
import numpy as np
import pickle as pkl
from argparse import ArgumentParser

class SimulationHelper:
    def __init__(self):

        sim.simxFinish(-1)
        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

        if self.clientID == -1:
            sys.exit("Couldn't connect to CSim server")
        else:
            print("Connected to Sim Server")

        output = sim.simxGetObjectGroupData(self.clientID,sim.sim_appobj_object_type,0,sim.simx_opmode_blocking)
        object_handles_list = output[1]
        object_names_list = output[-1]

        self.snakes_v_joints = {k:[] for k in ['0','8','17','26','35','44']}
        self.snakes_h_joints = {k:[] for k in ['0','8','17','26','35','44']}

        for i in range(len(object_names_list)):
            if 'vJoint' in str(object_names_list[i]) or 'hJoint' in str(object_names_list[i]):
                try:
                    num = int(re.findall(r'\d+', str(object_names_list[i]))[0])
                except:
                    num = 0
                if num>=8 and num<=16:
                    if 'vJoint' in str(object_names_list[i]):
                        self.snakes_v_joints['8'].append(object_handles_list[i])
                    elif 'hJoint' in str(object_names_list[i]):
                        self.snakes_h_joints['8'].append(object_handles_list[i])
                elif num>=17 and num<=25:
                    if 'vJoint' in str(object_names_list[i]):
                        self.snakes_v_joints['17'].append(object_handles_list[i])
                    elif 'hJoint' in str(object_names_list[i]):
                        self.snakes_h_joints['17'].append(object_handles_list[i])
                elif num>=26 and num<=34:
                    if 'vJoint' in str(object_names_list[i]):
                        self.snakes_v_joints['26'].append(object_handles_list[i])
                    elif 'hJoint' in str(object_names_list[i]):
                        self.snakes_h_joints['26'].append(object_handles_list[i])
                elif num>=35 and num<=43:
                    if 'vJoint' in str(object_names_list[i]):
                        self.snakes_v_joints['35'].append(object_handles_list[i])
                    elif 'hJoint' in str(object_names_list[i]):
                        self.snakes_h_joints['35'].append(object_handles_list[i])
                elif num>=44 and num<=52:
                    if 'vJoint' in str(object_names_list[i]):
                        self.snakes_v_joints['44'].append(object_handles_list[i])
                    elif 'hJoint' in str(object_names_list[i]):
                        self.snakes_h_joints['44'].append(object_handles_list[i])
                else:
                    if 'vJoint' in str(object_names_list[i]):
                        self.snakes_v_joints['0'].append(object_handles_list[i])
                    elif 'hJoint' in str(object_names_list[i]):
                        self.snakes_h_joints['0'].append(object_handles_list[i])

        self.t_const = 0.050000000745058
    
    def run_simulation(self, A, w, p):
        t = 0
        A /= 100.
        w *= 100.

        while t<5:
            t += self.t_const
            time.sleep(self.t_const)

            for ind,j in enumerate(self.snakes_h_joints.keys()):
                for i in range(1,len(self.snakes_v_joints[j])):
                    h_cmd = 0
                    v_cmd = (A[ind]*math.sin(t*p[ind]+i*w[ind]))
                    err_code_h = sim.simxSetJointTargetPosition(self.clientID,self.snakes_h_joints[j][i-1],h_cmd,sim.simx_opmode_oneshot)
                    err_code_v = sim.simxSetJointTargetPosition(self.clientID,self.snakes_v_joints[j][i-1],v_cmd,sim.simx_opmode_oneshot)
        
        return [0]*len(self.snakes_h_joints.keys())
    
class GA:
    def __init__(self, sol_per_pop, num_mating_parents, num_generations):
        loaded_data = np.load('points.npz', allow_pickle=True)
        params = pkl.load(open('params_final.pkl','rb'))
        self.actual_soln = np.array([params['amp'],params['omega'],params['phase']])
        self.num_weights = self.actual_soln.shape[0]
        self.sol_per_pop = sol_per_pop
        self.num_mating_parents = num_mating_parents
        self.num_generations = num_generations
        self.equation_inputs = np.array(loaded_data['cx'][0])
        self.pop_size = (self.sol_per_pop,self.num_weights)
        self.new_population = np.random.uniform(-1.0,1.0,self.pop_size)
        self.sim_help = SimulationHelper()
    
    def function_valuation(self, weights):
        x = self.equation_inputs
        y = weights[0]*np.sin(weights[1]*x + weights[2])
        return y
    
    def calc_pop_fitness(self):
        true_val = self.function_valuation(self.actual_soln).reshape((-1,1))
        pop_val = []
        for i in range(len(self.new_population)):
            pop_val.append(np.sum(np.square(self.function_valuation(self.new_population[i,:]).reshape((-1,1))-true_val))-self.dist[i])
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
            print("Generation : ", generation)
            A,w,p = np.array([s[0] for s in self.new_population]), np.array([s[1] for s in self.new_population]), np.array([s[2] for s in self.new_population])
            self.dist = []
            for _ in range(0,len(A),6):
                self.dist.extend(self.sim_help.run_simulation(A,w,p))
            self.calc_pop_fitness()
            self.select_mating_pool()
            self.crossover(offspring_size=(self.pop_size[0]-self.parents.shape[0], self.num_weights))
            self.mutation()
            self.new_population[0:self.parents.shape[0], :] = self.parents
            self.new_population[self.parents.shape[0]:, :] = self.offspring
            print("Best result : ", np.min(self.fitness))
        self.calc_pop_fitness()
        self.best_match_idx = np.where(self.fitness == np.min(self.fitness))
        print("Best solution : ", self.new_population[self.best_match_idx[0][0], :])
        print("Best solution fitness : ", self.fitness[self.best_match_idx[0][0]])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--pop_size', type=int, required=True, help='Population Size')
    parser.add_argument('-m', '--no_mating_parents', type=int, required=True,  help='Number of mating parents')
    parser.add_argument('-n', '--no_of_generations', type=int, required=True, help='Number of generations')

    args = parser.parse_args()
    print(args)
    sol_per_pop, num_mating_parents, num_generations = args.pop_size, args.no_mating_parents, args.no_of_generations
    ga = GA(sol_per_pop, num_mating_parents, num_generations)
    ga.loop_run()