import re
import sim
import sys
import pickle as pkl
import IPython
import numpy as np
import math
import time
from pyrep import PyRep

pr = PyRep()
pr.launch('pyrep_testing_scene.ttt',headless = False)
pr.start()
params = pkl.load(open('learnt_params.pkl','rb'))
out  = pr.script_call("run_on_snake@Snake1#18",1,[0,0,0],[0.635,1.435],['yes','its','working'],[]) 
out  = pr.script_call("run_on_snake@Snake1#0",1,[0,0,0],[0.635,1.435],['yes','its','working'],[])
out  = pr.script_call("run_on_snake@Snake1#9",1,[0,0,0],[0.635,1.435],['yes','its','working'],[])

for _ in range(100):
    pr.step() #each is one time step

# out  = pr.script_call("run_on_snake@Plane#0",1,[0,0,0],[0.635,1.435],['yes','its','working'],[]) 
# for _ in range(100):
#     pr.step()

IPython.embed()
pr.stop() #pr.stop resets the simulator

pr.start()
params = pkl.load(open('learnt_params.pkl','rb'))
#out  = pr.script_call("run_on_snake@Snake1#1",1,[0,0,0],[0.635,1.435],['yes','its','working'],[]) 
out  = pr.script_call("run_on_snake@Snake1#0",1,[0,0,0],[0.635,1.435],['yes','its','working'],[])
#out  = pr.script_call("run_on_snake@Snake1#2",1,[0,0,0],[0.635,1.435],['yes','its','working'],[])

for _ in range(100):
    pr.step()
pr.stop() 
pr.shutdown()
