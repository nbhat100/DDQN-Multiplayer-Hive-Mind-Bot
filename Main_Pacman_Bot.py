from DDQN_Model import ddqnTrainer
from get_environment import GetEnv
from gather_training_points import readReward


Frames = 4
OutFrameSize = 80
InFrameDim = (556, 838, 449, 786)
InputShape = (Frames, InFrameDim[1]-InFrameDim[0], InFrameDim[3] - InFrameDim[2])
TotalStepLimit = 5000000
ActionSpace = 5


def main():
    run = 0
    total_step = 0
    game_model = ddqnTrainer(InputShape, ActionSpace)
    prev_score = 0
    while True:
        run += 1
        env = GetEnv(inDim=InFrameDim, outDim=(OutFrameSize,OutFrameSize))
        current_state = env.takeImage(4, "None")
        step = 0
        score = readReward()
        while total_step <= TotalStepLimit:
            total_step += 1
            step += 1
            action = game_model.move(current_state)
            env.step(action)
            next_state = env.takeImage(4, "None")
            current_score = score.mainLoop()
            reward = current_score - prev_score
            prev_score = current_score
            game_model.remember(current_state, action, reward, next_state)
            game_model.step_update(total_step)


main()
