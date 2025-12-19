import os
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
from functools import partial

#
from simulation.dynamic_gap import DynamicGap
from mpc.mpc import MPC
from simulation.animation import SimVisual
#
import matplotlib
matplotlib.use('TkAgg')  
plt.ion()  


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=False,
        help="Save the animation as a video file")
    return parser

def run_mpc(env):
    #
    env.reset()
    t, n = 0, 0
    t0 = time.time()
    while t < env.sim_T:
        t = env.sim_dt * n
        _, _, _, info = env.step()
        t_now = time.time()
        print(t_now - t0)
        #
        t0 = time.time()
        #
        n += 1
        update = False
        if t >= env.sim_T:
            update = True
        yield [info, t, update]

def main():
    #
    args = arg_parser().parse_args()
    #
    plan_T = 2.0   
    plan_dt = 0.1
    so_path = "./mpc/saved/mpc_v1.so" 
    #
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(mpc, plan_T, plan_dt)
    
    #
    sim_visual = SimVisual(env)

   
    run_frame = partial(run_mpc, env)
    ani = animation.FuncAnimation(
        sim_visual.fig, 
        sim_visual.update, 
        frames=run_frame,
        init_func=sim_visual.init_animate, 
        interval=100, 
        blit=True, 
        repeat=False,
        cache_frame_data=False  
    )
    
    
    if args.save_video:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("MPC_0.mp4", writer=writer)
    
    plt.tight_layout()
    
    
    print("\nAnimation running. Close the window to exit.")
    plt.show(block=True)
    
if __name__ == "__main__":
    main()