import re
import sim
import sys
import pickle as pkl
import IPython
import numpy as np
import math
import time
from pyrep import PyRep
#pr = PyRep()
#pr.launch('pyrep_testing_scene.ttt',headless = False)
#pr.start()
#out  = pr.script_call("run_on_snake@Snake1",1,[10,11,12],[1.1,2.2,3.3,4.4],['yes','its','working'],[]) 


#IPython.embed()
sim.simxFinish(-1)
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP

params = pkl.load(open('learnt_params.pkl','rb'))

if clientID == -1:
    sys.exit("Couldn't connect to CSim server")
else:
    print("Connected to Sim Server")

output = sim.simxGetObjectGroupData(clientID,sim.sim_appobj_object_type,0,sim.simx_opmode_blocking)
object_handles_list = output[1]
object_names_list = output[-1]
object_handles_dict = dict(zip(object_names_list,object_handles_list))
v_joints = []
h_joints = []
num_joints_per_snake = 8
num_snakes = 6

snakes_v_joints = np.zeros([num_snakes,num_joints_per_snake])
snakes_h_joints = np.zeros([num_snakes,num_joints_per_snake]) 
#IPython.embed()
for i in range(len(object_names_list)):
    if 'vJoint' in str(object_names_list[i]):
        v_joints.append(object_handles_dict[object_names_list[i]])
    elif 'hJoint' in str(object_names_list[i]):
        h_joints.append(object_handles_dict[object_names_list[i]])

j = 0 
k = 0
for i in range(len(v_joints)):
    snakes_v_joints[j][k] = int(v_joints[i])
    snakes_h_joints[j][k] = int(h_joints[i])  
    k+=1        
    if (i+1)%num_joints_per_snake==0:
        print(i)
        j+=1
        k = 0

t_const = 0.050000000745058
t = 0.0
s=0.0
t=0

A = params['amp']/100
w = params['omega']*100
#out  = pr.script_call("run_on_snake@Plane#0",1,[],[A,w],['yes','its','working'],[]) 
#IPython.embed()
# for _ in range(10):

while t<5.0:
    t += t_const
    time.sleep(t_const)
    for j in range(num_snakes):
        for i in range(1,num_joints_per_snake):
            h_cmd = 0
            v_cmd = (A*math.sin(t*2.36+i*w))
            #IPython.embed()        
            err_code_h = sim.simxSetJointTargetPosition(clientID,int(snakes_h_joints[j,i-1]),h_cmd,sim.simx_opmode_oneshot)
            err_code_v = sim.simxSetJointTargetPosition(clientID,int(snakes_v_joints[j,i-1]),v_cmd,sim.simx_opmode_oneshot)
  
#     for s in range(num_snakes):
#         sim.simxRemoveModel(clientID,object_handles_dict['Snake'+str(s+1)],sim.simx_opmode_blocking)
    
#     for s in range(num_snakes):
#         sim.simxLoadModel(clientID,'/home/raoshashank/ARL-Project/Snake_Models/snake'+str(s)+'.ttm',True,sim.simx_opmode_blocking)  
    
#     output = sim.simxGetObjectGroupData(clientID,sim.sim_appobj_object_type,0,sim.simx_opmode_blocking)
#     object_handles_list = output[1]
#     object_names_list = output[-1]
#     object_handles_dict = dict(zip(object_names_list,object_handles_list))
#     v_joints = []
#     h_joints = []
#     num_joints_per_snake = 8
#     num_snakes = 6

#     snakes_v_joints = np.zeros([num_snakes,num_joints_per_snake])
#     snakes_h_joints = np.zeros([num_snakes,num_joints_per_snake]) 
#     #IPython.embed()
#     for i in range(len(object_names_list)):
#         if 'vJoint' in str(object_names_list[i]):
#             v_joints.append(object_handles_dict[object_names_list[i]])
#         elif 'hJoint' in str(object_names_list[i]):
#             h_joints.append(object_handles_dict[object_names_list[i]])

#     j = 0 
#     k = 0
#     for i in range(len(v_joints)):
#         snakes_v_joints[j][k] = int(v_joints[i])
#         snakes_h_joints[j][k] = int(h_joints[i])  
#         k+=1        
#         if (i+1)%num_joints_per_snake==0:
#             print(i)
#             j+=1
#             k = 0


#     print("Finished one run...Resetting")
    

#pr.stop()
#pr.shutdown()