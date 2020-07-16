from convolutional_neural_net import ConvNet
import os
import random
import numpy as np

# Hyper Parameters
StartingReplaySize = 0
ExplorationMax = 1
MemorySize = 900000
TrainingFrequency = 4
BatchSize = 32
Gamma = 0.99
ExplorationMin = 0.1
ExplorationSteps = 85000
ExplorationDecay = (ExplorationMax - ExplorationMin)/ExplorationSteps
ModelPersistenceUpdateFrequency = 10
TargetUpdateFrequency = 40


class ddqnModel:

    def __init__(self, input_shape, action_space, model_path):
        self.input_shape = input_shape
        self.action_space = action_space
        self.model_path = model_path
        self.ddqn = ConvNet(self.input_shape, action_space).model
        if os.path.isdir(self.model_path):
            self.ddqn.load_weights(self.model_path)

    def save_model(self):
        self.ddqn.save_weights(self.model_path)


class ddqnTrainer(ddqnModel):
    def __init__(self, input_shape, action_space):
        ddqnModel.__init__(self, input_shape, action_space, './models')
        self.ddqn_target = ConvNet(self.input_shape, action_space).model
        self.reset_target()
        self.epsilon = ExplorationMax
        self.memory = []

    def reset_target(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())

    def move(self, state):
        print(f"memory: {len(self.memory)}")
        if np.random.rand() < self.epsilon or len(self.memory) < StartingReplaySize:
            return random.randrange(self.action_space[0]), random.randrange(self.action_space[1])
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0]), np.argmax(q_values[1])

    def remember(self, current_state, action, reward, next_state):
        self.memory.append({"current_state": current_state,
                            "action": action,
                            "reward": reward,
                            "next_state": next_state})
        if len(self.memory) > MemorySize:
            self.memory.pop(0)

    def train(self):
        batch = np.asarray(random.sample(self.memory, BatchSize))
        if len(batch) < BatchSize:
            return
        current_states = []
        q_values = []
        max_q_values = []
        for entry in batch:
            current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
            current_states.append(current_state)
            next_state = np.expand_dims(np.asarray(entry["next_state"]).astype(np.float64), axis=0)
            next_state_prediction = self.ddqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.ddqn.predict(current_state)[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + Gamma * next_q_value
            q_values.append(q)
            max_q_values.append(np.max(q))
        fitModel = self.ddqn.fit(np.asarray(current_states).squeeze(), np.asarray(q_values).squeeze(), batch_size=BatchSize, verbose=0)
        loss = fitModel.history["loss"][0]
        accuracy = fitModel.history["acc"][0]
        return loss, accuracy, sum(max_q_values)/len(max_q_values)

    def update_epsilon(self):
        self.epsilon -= ExplorationDecay
        self.epsilon = max(ExplorationMin, self.epsilon)

    def step_update(self, totalStep):
        if len(self.memory) < StartingReplaySize:
            return
        if totalStep % TrainingFrequency == 0:
            loss, accuracy, average_max_q = self.train()
            print(f"loss: {loss}, accuracy: {accuracy}, average_max_q: {average_max_q}")
        self.update_epsilon()
        if totalStep % ModelPersistenceUpdateFrequency == 0:
            self.save_model()
        if totalStep % TargetUpdateFrequency == 0:
            self.reset_target()
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(totalStep))
