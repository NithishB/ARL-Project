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
    cur_id = multiprocessing.current_process()._identity[0] - 1
    my_A = A[cur_id*num_snakes:(cur_id+1)*num_snakes]
    my_w = w[cur_id*num_snakes:(cur_id+1)*num_snakes]
    pr = PyRep()
    pr.launch('pyrep_testing_scene.ttt',headless = True)
    pr.start()
    print("Started")
    # params = pkl.load(open('learnt_params.pkl','rb')) 
    for i in range(num_snakes):
        _  = pr.script_call("run_on_snake@Snake1#"+str(9*i),1,[0,0,0],[my_A[i],my_w[i]],['yes','its','working'],[])

    for _ in range(100):
        pr.step()
    
    pr.stop()
    print("Stopped")
    pr.shutdown()

processes = [Process(target=run, args=()) for i in range(PROCESSES)]
[p.start() for p in processes]
[p.join() for p in processes]
