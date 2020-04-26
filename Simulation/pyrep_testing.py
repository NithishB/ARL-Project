import re
import sim
import sys
import time
import math
import multiprocessing
import numpy as np
import pickle as pkl
from pyrep import PyRep
from multiprocessing import Process

PROCESSES = 10

A = np.random.uniform(0,1,(100,)).tolist()
w = (1+np.random.uniform(0,1,(100,))).tolist()


def run():
    global A, w
    num_snakes = 10
    #cur_id = multiprocessing.current_process()._identity[0] - 1
    #my_A = A[cur_id*num_snakes:(cur_id+1)*num_snakes]
    #my_w = w[cur_id*num_snakes:(cur_id+1)*num_snakes]
    pr = PyRep()
    pr.launch('pyrep_testing_scene.ttt',headless = False)
    pr.start()
    print("Started")
    # params = pkl.load(open('learnt_params.pkl','rb')) 
    position_initial = []
    for i in range(num_snakes):
        #_  = pr.script_call("run_on_snake@Snake1#"+str(9*i),1,[0,0,0],[my_A[i],my_w[i]],['yes','its','working'],[])
        _  = pr.script_call("run_on_snake@Snake1#"+str(9*i),1,[0,0,0],[0.635,1.45],['yes','its','working'],[])
        position_initial.append(np.array(pr.script_call("get_position@Snake1#"+str(9*i),1,[],[],['yes','its','working'],[])[1]))
    #print(position_initial)
    for _ in range(10):
        pr.step()
    position_final = []
    #IPython.embed()
    for i in range(num_snakes):
        position_final.append(np.array(pr.script_call("get_position@Snake1#"+str(9*i),1,[],[],['yes','its','working'],[])[1]))
        diff = abs(position_final[i]-position_initial[i])
        print("Distance moved"+str(i)+" = " + str(diff[1]**2+diff[0]**2))
    IPython.embed()
    print(position_final)
    pr.stop()
    print("Stopped")
    pr.shutdown()

#processes = [Process(target=run, args=()) for i in range(PROCESSES)]
#[p.start() for p in processes]
#[p.join() for p in processes]
run()
