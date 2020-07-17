import numpy as np
from value_iteration import choose_action


np.random.seed(1)
INF = np.nan_to_num([1 * float('-inf')])


def learning_rate(lr, gradient):
    """
    :param lr: (float) last step learning rate
    :param gradient: (list)
    :return:
    """
    if len(gradient) < 2 or gradient[-2] - gradient[-1] > 1e-3:
        return lr
    else:
        return lr * 3


class MaxEntIRL(object):

    def __init__(self, mdp, irl_rate, n_features=False):
        self.mdp = mdp
        self.n_states, self.n_actions, self.n_steps = mdp.n_states, mdp.n_actions, mdp.n_steps
        self.irl_rate = irl_rate
        # self.alpha = self.mdp.alpha
        if n_features:
            self.n_features = 49

    def feature_expectation(self):
        feature_expectations = np.zeros(self.n_features)
        for episode in self.mdp.demonstrations:
            for i in range(self.n_steps):  # g
                if i < self.n_steps-1:
                    feature_expectations += self.mdp.feature_vector((episode[i], episode[i+1]))
                else:
                    feature_expectations += self.mdp.feature_vector((episode[i], episode[i]))

        feature_expectations /= len(self.mdp.demonstrations)

        return feature_expectations

    def init_state_dist(self):

        path_states = np.array([str(0) + path[0] for path in self.mdp.demonstrations])
        # start_states = np.array([state[0] for state in path_states])
        start_state_count = np.bincount(path_states.astype(int), minlength=self.n_states)

        return start_state_count.astype(float) / len(path_states)

    def state_visitation_frequency(self, policy):
        state_visitation = np.expand_dims(self.init_state_dist(), axis=1)
        sa_visit_t = np.zeros(
            (self.n_states, self.n_actions, self.n_steps)
        )

        for i in range(self.n_steps):
            sa_visit = state_visitation * policy
            sa_visit_t[:, :, i] = sa_visit

            new_state_visitation = np.einsum("ij,ijk->k", sa_visit,
                                             self.mdp.transition_matrix,
                                             optimize=False)

            state_visitation = np.expand_dims(new_state_visitation, axis=1)

        mu_exp = np.sum(sa_visit_t, axis=2)
        # print('mu_exp', mu_exp)
        return mu_exp

    def demo_savf(self):
        savf = np.zeros((self.mdp.n_states, self.n_actions), dtype=np.float32)
        for demo in self.mdp.demonstrations.values():
            for episode in demo:
                state, action = episode[0], episode[1]
                s, a = self.mdp.state_idx[state], self.mdp.action_idx[action]
                savf[s, a] += 1
        # print(savf, len(self.demo))
        savf /= len(self.mdp.demonstrations)
        return savf

    def sample_savf(self, policy, episodes_container, n_iters=25000):
        """
        :param policy:
        :param episodes_container:
        :param n_iters:
        :return:
        """
        savf = np.zeros((self.mdp.n_states, self.n_actions), dtype=np.float32)

        for i in range(n_iters):
            # if i % 100000 == 0:
            #    print('sampled ', i, ' episode')
            state = self.mdp.reset()
            episode = state[0]
            t = 0

            while True:
                action = choose_action(state, policy)
                next_state = self.mdp.step(action)

                # reward = self.mdp.get_reward((state, action))
                savf[int(state), int(action)] += 1

                t += 1
                state = next_state
                # print(episode, state, type(episode), type(state))
                episode += str(state[0])

                if t == self.n_steps-1:
                    savf[int(state), int(state)] += 1
                    break

            episodes_container.append(episode)
        savf /= float(n_iters)

        return savf / (float(self.n_steps))

    def sample_sa_feature(self, policy, episodes_container, n_iters=25000):
        """
        :param policy:
        :param episodes_container: (list)
        :param n_iters:
        :return:
        """

        irl_feature_expectations = np.zeros(self.n_features)

        for i in range(n_iters):
            # if i % 100000 == 0:
            #    print('sampled ', i, ' episode')
            state = self.mdp.reset()
            episode = state[0]
            t = 0

            while True:
                action = choose_action(state, policy)
                next_state = self.mdp.step(action)

                # reward = self.mdp.get_reward((state, action))
                irl_feature_expectations += self.mdp.feature_vector((state, action))

                t += 1
                state = next_state
                # print(episode, state, type(episode), type(state))
                episode += str(state[0])

                if t == self.n_steps-1:
                    break

            episodes_container.append(episode)

        return irl_feature_expectations / float(n_iters)
