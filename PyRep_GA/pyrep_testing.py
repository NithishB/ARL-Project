import os
import time
import numpy as np
import pickle as pkl
import tkinter as tk
from pyrep import PyRep

def write(l,name):
    l.config(text=name, fg='red')

def show_screen(name):
    root = tk.Tk()
    root.wm_overrideredirect(True)
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    root.bind("<Button-1>", lambda evt: root.destroy())
    l = tk.Label(text='', font=("Helvetica", 70))
    l.pack(expand=True)
    start = time.time()
    while time.time() - start < 3:
        write(l,name)
        root.update_idletasks()
        root.update()
    root.destroy()

def run_snake(A, w, p):
    num_snakes = 3
    pr = PyRep()
    pr.launch('pyrep_video.ttt',headless = False)
    pr.start()
    for i in range(num_snakes):
        _  = pr.script_call("run_on_snake@Snake1#"+str(9*i),1,[0,0,0],[A[i],w[i],p[i]],['yes','its','working'],[])
    for _ in range(100):
        pr.step()
    pr.stop()
    pr.shutdown()

def run_for_each_snake():
    methods = os.listdir("checkpoint")
    for m in methods:
        best_snakes = pkl.load(open(os.path.join('checkpoint',m,'best_snakes.pkl'),'rb'))
        num_gen = len(best_snakes)
        for gen in range(num_gen):
            top_3_snakes = np.array(best_snakes[gen][:3])
            A,w,p = top_3_snakes[:,0,0].tolist(), top_3_snakes[:,0,1].tolist(), top_3_snakes[:,0,2].tolist()
            name = " ".join(["Method",m,"Generation",str(gen)])
            show_screen(name)
            run_snake(A,w,p)

if __name__ == "__main__":
    run_for_each_snake()