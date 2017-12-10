import numpy as np
np.random.seed(1337)

from gym_torcs import TorcsEnv
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
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False, port=p)
    
    EXPLORE = total_explore
    episode_count = max_eps
    max_steps = max_steps_eps
    epsilon = epsilon_start
    done = False
    
    step = 0
    best_reward = -100000
    best_lap_time = 100000

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        ##Occasional Testing
        if (( np.mod(i, 10) == 0 ) and (i>10)):
            train_indicator= 0
        else:
            train_indicator=is_training

        #relaunch TORCS every 3 episode because of the memory leak error
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   
        else:
            ob = env.reset()
            
        # Early episode annealing for out of track driving and small progress
        # During early training phases - out of track and slow driving is allowed as humans do ( Margin of error )
        # As one learns to drive the constraints become stricter
        
        random_number = random.random()
        eps_early = max(epsilon,0.1)
        if (random_number < (1.0-eps_early)) and (train_indicator == 1):
            early_stop = 1
        else: 
            early_stop = 0
        print("Episode : " + str(i) + ' Early Stopping: ' + str(early_stop) +  ' Epsilon: ' + str(eps_early) +  ' RN: ' + str(random_number)  )

        #Initializing the first state
        #s_t = np.hstack((ob.track, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.trackPos, ob.angle))
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
            
            #Take noisy actions during training
            if (train_indicator):
                epsilon -= 1.0 / EXPLORE
                epsilon = max(epsilon, 0.1)
                a_t = agent.noise_action(s_t,epsilon)
            else:
                a_t = agent.action(s_t)
                
            ob, r_t, done, info = env.step(a_t,early_stop)
            #ob, r_t, done, info = env.step(a_t[0])
            #s_t1 = np.hstack((ob.focus, ob.distFromStart, ob.distRaced, ob.racePos,ob.track, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.trackPos, ob.angle))
            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            
            #train with state_t, state_t+1, actions_t, actions_t+1, and reward
            if (train_indicator) and i > 0:
                agent.train(s_t,a_t,r_t,s_t1,done)
                
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
            #if ( (np.mod(step,15)==0) ):        
#            print("Episode", i, "Step", step_eps,"Epsilon", epsilon , "Action", a_t, "Reward", r_t )
            if ( (np.mod(step,15)==0) ):        
                print("[{:.2f}s] Episode {:d} Step {:d} Epsilon {:.4f} ".format(total_lap_time, i, int(step_eps), epsilon), end='')
                print("Action {0} Reward {1}".format(a_t, r_t))

            step += 1
            step_eps += 1
            if done:
                break
            if i== 0:
                break
                
        #Saving the best model.
        if total_lap_time < best_lap_time and prev_lap_time < 0 and \
                total_lap_time > 170 and not info['error'] and i > 0 :
        #if total_reward >= best_reward and i > 0:
            #if (train_indicator==1):
            print("Now we save model with reward " + str(total_lap_time) + " previous best reward was " + str(best_lap_time))
            best_reward = total_reward
            best_lap_time = total_lap_time
            agent.saveNetwork(i)       
                
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()

