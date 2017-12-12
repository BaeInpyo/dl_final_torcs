import numpy as np

def calcReward(obs):
    ## write your own reward
    trackPos = obs.trackPos
    sp = obs.speedX
    angle = obs.angle

    if np.abs(trackPos) > 0.9:
        progress = sp*np.cos(angle) - np.abs(sp*np.sin(angle)) - \
                sp*(0.5*np.abs(trackPos))
    else:
        progress = sp*np.cos(angle) - np.abs(sp*np.sin(angle))
    reward = progress

    return reward
