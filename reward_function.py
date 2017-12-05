import numpy as np

def calcReward(obs):
    ## write your own reward
    track = np.array(obs['track'])
    trackPos = np.array(obs['trackPos'])
    sp = np.array(obs['speedX'])
    damage = np.array(obs['damage'])
    rpm = np.array(obs['rpm'])

    progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - \
            sp*np.abs(obs['trackPos'])
    reward = progress

    return reward
