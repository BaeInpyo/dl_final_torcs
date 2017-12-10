import numpy as np

def calcReward(obs):
    ## write your own reward
    #track = obs.track
    trackPos = obs.trackPos
    sp = obs.speedX
    angle = obs.angle

    if np.abs(trackPos) > 0.8:
        progress = sp*np.cos(angle) - np.abs(sp*np.sin(angle)) - \
                sp*(np.abs(trackPos))
    else:
        progress = sp*np.cos(angle) - np.abs(sp*np.sin(angle))
    reward = progress

    return reward
