import time
import tkinter as tk
from random import choice

colors = ('red', 'green', 'blue')
def do_stuff(s):
    color = choice(colors)
    l.config(text=s, fg=color)

root = tk.Tk()
root.wm_overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.bind("<Button-1>", lambda evt: root.destroy())

l = tk.Label(text='', font=("Helvetica", 70))
l.pack(expand=True)

start = time.time()
while time.time() - start < 4:
    do_stuff('Hello')
    root.update_idletasks()
    root.update()

print("Hi")