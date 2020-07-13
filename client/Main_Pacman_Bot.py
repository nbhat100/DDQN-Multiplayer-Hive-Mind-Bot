from DDQN_Model import ddqnTrainer
from get_environment import GetEnv
from gather_training_points import readReward
import pyautogui
from time import sleep


Frames = 4
OutFrameSize = 80
InFrameDim = (556, 838, 449, 786)
InputShape = (Frames, InFrameDim[1]-InFrameDim[0], InFrameDim[3] - InFrameDim[2])
TotalStepLimit = 5000000
ActionSpace = ("w", "a", "s", "d", "space", "e", "q")


class Environment:
    def __init__(self):
        self.env = GetEnv(inDim=InFrameDim, outDim=(OutFrameSize, OutFrameSize))
        self.out = []
        for i in range(5):
            self.out.append(self.env.takeImage(4, "None"))
        while True:
            self.out.append(self.env.takeImage(4, "None"))
            self.out.pop(0)

    def getEnvironment(self):
        return self.out

    def step(self, action):
        print(f"action: {action}")
        pyautogui.press(ActionSpace[action])


def main():
    run = 0
    total_step = 0
    game_model = ddqnTrainer(InputShape, len(ActionSpace))
    prev_score = 0
    while True:
        run += 1
        env = Environment()
        current_state = env.getEnvironment()
        step = 0
        score = readReward()
        while total_step <= TotalStepLimit:
            total_step += 1
            step += 1
            print(f"Step: {total_step}")
            action = game_model.move(current_state)
            env.step(action)
            next_state = env.getEnvironment()
            current_score = score.mainLoop()
            reward = current_score - prev_score
            print(reward)
            prev_score = current_score
            game_model.remember(current_state, action, reward, next_state)
            game_model.step_update(total_step)


main()
