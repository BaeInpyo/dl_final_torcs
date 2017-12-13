0. Environment
tensorflow      --version 1.3.0
python          --version 3.4
linux           --version 4.9.16
torcs simulator --provided

1. Necessary modification
We modified the number of states (training.py / testGame.py)
We modified an observation to be normalized (gym_*.py)
We added the lap time observation to monitor the result. (training.py /gym_*.py)

2. Model
We use DDPG (actor-critic) for this final project. Basically we followed the
paper about Torcs-DDPG, and then we modified some hyperparameters and the reward
function.

3. Result
Our best total lap time is 177.23s and the best single lap time among three laps
is 56.54s.
