import numpy as np

from gym_torcs_test import TorcsEnv
import random
import argparse
import tensorflow as tf
from my_config import *

from driver_agent import *
import gc
gc.enable()

import timeit
import math

print( is_training )
print( total_explore )
print( max_eps )
print( max_steps_eps )
print( epsilon_start )
print( port )

def playGame(train_indicator=is_training, p=port):    #1 means Train, 0 means simply Run

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input
    env_name = 'Torcs_Env'
    agent = DriverAgent(env_name, state_dim, action_dim)

    # Generate a Torcs environment
    vision = False
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False,manual_launch=True,  port=p)
    
    EXPLORE = total_explore
    episode_count = max_eps
    max_steps = max_steps_eps
    epsilon = epsilon_start
    done = False
    
    step = 0
    best_reward = -100000

    print("TORCS Experiment Start.")
    for i in range(2):
        ##Occasional Testing
        if i == 1:
            input('Press Enter to Start')
        #relaunch TORCS every 3 episode because of the memory leak error
        ob = env.reset()
            
        # Early episode annealing for out of track driving and small progress
        # During early training phases - out of track and slow driving is allowed as humans do ( Margin of error )
        # As one learns to drive the constraints become stricter
        
        early_stop = 0

        #Initializing the first state
        #s_t = np.hstack((ob.focus, ob.distFromStart, ob.distRaced, ob.racePos, ob.track, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.trackPos, ob.angle))
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        print(len(s_t))
        #Counting the total reward and total steps in the current episode
        total_reward = 0.
        step_eps = 0.

        total_lap_time = 0.
        lap_time = 0.
        prev_lap_time = 1.
        finish_lap = False
        
        for j in range(max_steps):
            #print(np.hstack((ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ)))
            
            #Take noisy actions during training
            a_t = agent.action(s_t)
                
            ob, r_t, done, info = env.step(a_t,early_stop)
            #s_t1 = np.hstack((ob.focus, ob.distFromStart, ob.distRaced, ob.racePos,ob.track, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.trackPos, ob.angle))
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
                
            #Cheking for nan rewards
            if ( math.isnan( r_t )):
                r_t = 0.0
                for bad_r in range( 50 ):
                    print( 'Bad Reward Found' )

            if prev_lap_time >= 0:
                if lap_time != prev_lap_time:
                    prev_lap_time = lap_time
                else:
                    finish_lap = True

            lap_time = ob.curLapTime
            if (lap_time < 0.2) or done or finish_lap:
                total_lap_time += prev_lap_time
                if finish_lap:
                    prev_lap_time = -1
                finish_lap = False

            total_reward += r_t
            s_t = s_t1

            #Displaying progress every 15 steps.
            if ( (np.mod(step,15)==0) ):        
                print("[{:.2f}s] Episode {:d} Step {:d} Epsilon {:.4f} ".format(total_lap_time, i, int(step_eps), epsilon), end='')
                print("Action {0} Reward {1}".format(a_t, r_t))

            step += 1
            step_eps += 1
            if done:
                break
            if i== 0:
                break
                
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    print("Finish.")

if __name__ == "__main__":
    playGame()

