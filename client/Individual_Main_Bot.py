from DDQN_Model import ddqnTrainer
from get_environment import GetEnv
from gather_training_points import readReward
import pydirectinput
from time import sleep
from threading import *
import random
import numpy as np


Frames = 5
InFrameDim = (128, 1031, 0, 1920)
InputShape = (Frames, 270, 480, 3)
TotalStepLimit = 5000000
ActionSpace = ("w", "a", "s", "d", "enter", "e")
MouseActionSpace = [[1061, 581], [1053, 619], [1031, 651], [999, 673], [961, 681], [922, 673], [890, 651], [868, 619], [861, 581], [868, 542], [890, 510], [922, 488], [961, 481], [999, 488], [1031, 510], [1053, 542]]
ScoreInDim = (1014, 1029, 875, 1045)
ScoreOutDim = (170, 16)


class Environment:
    def __init__(self):
        self.env = GetEnv(inDim=InFrameDim, outDim=(InputShape[2], InputShape[1]))
        self.out = []
        for i in range(5):
            self.out.append(self.env.takeImage(4, "None"))
        thread = Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        while True:
            self.out.append(self.env.takeImage(4, "None"))
            self.out.pop(0)

    def getEnvironment(self):
        return self.out

    def step(self, action, prev_action):
        print(f"action: {action}")
        print(f"Mouse Action: {MouseActionSpace[action[1]]}")
        upgrade = random.randrange(1, 9)
        pydirectinput.press(str(upgrade))
        pydirectinput.keyUp(ActionSpace[prev_action[0]])
        pydirectinput.keyDown(ActionSpace[action[0]])
        pydirectinput.moveTo(MouseActionSpace[action[1]][0], MouseActionSpace[action[1]][1])


def main():
    run = 0
    total_step = 0
    game_model = ddqnTrainer(InputShape, (len(ActionSpace), len(MouseActionSpace)))
    prev_score = 0
    prev_action = [0, 0]
    while True:
        run += 1
        env = Environment()
        current_state = env.getEnvironment()
        step = 0
        print(total_step)
        score = readReward()
        while total_step <= TotalStepLimit:
            total_step += 1
            step += 1
            print(f"Step: {total_step}")
            action = game_model.move(current_state)
            env.step(action, prev_action)
            prev_action = action
            next_state = env.getEnvironment()
            print(np.array(next_state).shape)
            current_score = score.mainLoop((1013, 1031, 895, 1025), (130, 17))
            reward = current_score - prev_score
            print(f"current_score: {current_score}")
            prev_score = current_score
            game_model.remember(current_state, action, reward, next_state)
            thread = Thread(target=game_model.step_update, args=(total_step,))
            thread.start()
            current_state = next_state


main()
