from DDQN_Model import ddqnTrainer
from get_environment import GetEnv
from gather_training_points import readReward
import pydirectinput
from time import sleep
import threading


Frames = 4
OutFrameSize = 80
InFrameDim = (556, 838, 449, 786)
InputShape = (Frames, InFrameDim[1]-InFrameDim[0], InFrameDim[3] - InFrameDim[2])
TotalStepLimit = 5000000
ActionSpace = ("w", "a", "s", "d", "space")
MouseActionSpace = [[868, 360], [860, 398], [839, 431], [806, 452], [768, 460], [729, 452], [697, 430], [675, 398], [668, 360], [675, 321], [697, 289], [729, 267], [768, 260], [806, 267], [838, 289], [860, 321]]
ScoreInDim = (905, 921, 875, 1045)
ScoreOutDim = (170, 16)


class Environment:
    def __init__(self):
        self.env = GetEnv(inDim=InFrameDim, outDim=(OutFrameSize, OutFrameSize))
        self.out = []
        for i in range(5):
            self.out.append(self.env.takeImage(4, "None"))
        thread = threading.Thread(target=self.run, args=())
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
            current_score = score.mainLoop(ScoreInDim, ScoreOutDim)
            reward = current_score - prev_score
            print(reward)
            prev_score = current_score
            game_model.remember(current_state, action, reward, next_state)
            game_model.step_update(total_step)


main()
