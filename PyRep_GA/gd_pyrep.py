import os
import time
import numpy as np
import pickle as pkl
import multiprocessing
from pyrep import PyRep
from argparse import ArgumentParser
from pyrep.backend.utils import suppress_std_out_and_err
import IPython

class GD:
    def __init__(self,num_iterations,learning_rate,headless):
        params = pkl.load(open('data/params_final.pkl','rb'))
        self.ground_truth = np.array([params['amp']/100.,params['omega']*100.,params['phase']])
        self.thetas = np.random.random([3])
        self.loss = 0
        self.num_iterations = num_iterations
        self.distance  = 0.0
        self.learning_rate = learning_rate
        print("Launching Sim")
        with suppress_std_out_and_err():
            self.pr = PyRep()
            self.pr.launch('GD_snake_env.ttt', headless=headless)
        

    def find_loss(self):
        #solution = [A,w,p]
        return np.sum(np.square(self.ground_truth - self.thetas)) - self.distance
    
    def gradient_step(self):
        #IPython.embed()
        self.thetas-=self.learning_rate*2*np.multiply(self.thetas,abs(self.ground_truth-self.thetas))
    
    def run(self):
        for i in range(self.num_iterations):
            #calc Distance first
            with suppress_std_out_and_err():
                self.pr.start()
                start = time.time()
                _  = self.pr.script_call("run_on_snake@Snake1#0",1,[0,0,0],[self.thetas[0],self.thetas[1],self.thetas[2]],['yes','its','working'],[])
            try:                    
                position_initial = np.array(self.pr.script_call("get_position@Snake1#0",1,[],[],['yes','its','working'],[])[1])
                for _ in range(100):
                    self.pr.step()
                position_final=np.array(self.pr.script_call("get_position@Snake1#0",1,[],[],['yes','its','working'],[])[1])
            except:
                print("sim argument error")
            self.pr.stop()
            diff = abs(position_final-position_initial)
            self.distance = diff[0]**2
            self.loss = self.find_loss()
            print("Iteration "+str(i) + "  Loss:"+str(self.loss))
            self.gradient_step()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--num_iterations', type=int, required=True, help='Population Size')
    parser.add_argument('-a', '--A_coeff', type=float, required=True,  help='Number of mating parents')
    parser.add_argument('-c', '--continue_training', type=bool, default=False, help="Continue previous training")
    parser.add_argument('-o', '--headless', type=str, default=False, help="headless")
    args = parser.parse_args()
    
    num_iterations, alpha,cont,headless = args.num_iterations, args.A_coeff, args.continue_training, args.headless
    try:
        os.mkdir('checkpoint')
    except:
        pass

    
    if not cont:
        gd = GD(num_iterations,alpha,headless)
        gd.run()
        print("Done, Shutting down sim")
        with suppress_std_out_and_err():
            gd.pr.shutdown()
        #pkl.dump(gd, open('checkpoint/gd_object.pkl','wb'))
    else:
        try:
            gd = pkl.load(open('checkpoint/gd_object.pkl','rb'))
            print("Successfully loaded from previous checkpoint!")
            gd.num_iterations = num_generations
            ga.run()
            gd.pr.shutdown()
            pkl.dump(gd, open('checkpoint/gd_object.pkl','wb'))
        except:
            print("Failed to load from previous checkpoint!")
