import re
import sim
import sys
import pickle as pkl
import IPython
import numpy as np
import math
import time

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

snakes_v_joints = {k:[] for k in ['0','8','17','26','35','44']}#,'53','62','71','80','89']}
snakes_h_joints = {k:[] for k in ['0','8','17','26','35','44']}#,'53','62','71','80','89']}

for i in range(len(object_names_list)):
    if 'vJoint' in str(object_names_list[i]) or 'hJoint' in str(object_names_list[i]):
        try:
            num = int(re.findall(r'\d+', str(object_names_list[i]))[0])
        except:
            num = 0
        if num>=8 and num<=16:
            if 'vJoint' in str(object_names_list[i]):
                snakes_v_joints['8'].append(object_handles_list[i])
            elif 'hJoint' in str(object_names_list[i]):
                snakes_h_joints['8'].append(object_handles_list[i])
        elif num>=17 and num<=25:
            if 'vJoint' in str(object_names_list[i]):
                snakes_v_joints['17'].append(object_handles_list[i])
            elif 'hJoint' in str(object_names_list[i]):
                snakes_h_joints['17'].append(object_handles_list[i])
        elif num>=26 and num<=34:
            if 'vJoint' in str(object_names_list[i]):
                snakes_v_joints['26'].append(object_handles_list[i])
            elif 'hJoint' in str(object_names_list[i]):
                snakes_h_joints['26'].append(object_handles_list[i])
        elif num>=35 and num<=43:
            if 'vJoint' in str(object_names_list[i]):
                snakes_v_joints['35'].append(object_handles_list[i])
            elif 'hJoint' in str(object_names_list[i]):
                snakes_h_joints['35'].append(object_handles_list[i])
        elif num>=44 and num<=52:
            if 'vJoint' in str(object_names_list[i]):
                snakes_v_joints['44'].append(object_handles_list[i])
            elif 'hJoint' in str(object_names_list[i]):
                snakes_h_joints['44'].append(object_handles_list[i])
        # elif num>=53 and num<=61:
        #     if 'vJoint' in str(object_names_list[i]):
        #         snakes_v_joints['44'].append(object_handles_list[i])
        #     elif 'hJoint' in str(object_names_list[i]):
        #         snakes_h_joints['44'].append(object_handles_list[i])
        # elif num>=62 and num<=70:
        #     if 'vJoint' in str(object_names_list[i]):
        #         snakes_v_joints['44'].append(object_handles_list[i])
        #     elif 'hJoint' in str(object_names_list[i]):
        #         snakes_h_joints['44'].append(object_handles_list[i])
        # elif num>=71 and num<=79:
        #     if 'vJoint' in str(object_names_list[i]):
        #         snakes_v_joints['44'].append(object_handles_list[i])
        #     elif 'hJoint' in str(object_names_list[i]):
        #         snakes_h_joints['44'].append(object_handles_list[i])
        # elif num>=80 and num<=88:
        #     if 'vJoint' in str(object_names_list[i]):
        #         snakes_v_joints['44'].append(object_handles_list[i])
        #     elif 'hJoint' in str(object_names_list[i]):
        #         snakes_h_joints['44'].append(object_handles_list[i])
        else:
            if 'vJoint' in str(object_names_list[i]):
                snakes_v_joints['0'].append(object_handles_list[i])
            elif 'hJoint' in str(object_names_list[i]):
                snakes_h_joints['0'].append(object_handles_list[i])

t_const = 0.050000000745058
t = 0.0
s=0.0
t=0

A = params['amp']/100
w = params['omega']*100

while True:
    t += t_const
    time.sleep(t_const)

    for j in snakes_h_joints.keys():
        for i in range(1,len(snakes_v_joints[j])):
            h_cmd = 0
            v_cmd = (A*math.sin(t*2.36+i*w))        
            err_code_h = sim.simxSetJointTargetPosition(clientID,snakes_h_joints[j][i-1],h_cmd,sim.simx_opmode_oneshot)
            err_code_v = sim.simxSetJointTargetPosition(clientID,snakes_v_joints[j][i-1],v_cmd,sim.simx_opmode_oneshot)