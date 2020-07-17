import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from maxent import MaxEntIRL
from value_iteration import solve, choose_action


class DeepIRLFC(MaxEntIRL):
    def __init__(self, mdp, n_input, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_fc'):
        super(DeepIRLFC, self).__init__(mdp, lr)

        # Multilayer perceptron variables
        self.n_input = n_input
        self.lr = lr
        self.n_h1 = n_h1
        self.n_h2 = n_h2

        self.name = name

        self.sess = tf.compat.v1.Session()
        self.input_s, self.reward, self.theta = self._build_network(self.name)
        self.optimizer = tfa.optimizers.AdamW(lr)

        self.grad_r = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
        self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
        self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

        self.grad_norms = tf.compat.v1.global_norm(self.grad_theta)
        self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build_network(self, name):
        input_s = tf.compat.v1.placeholder(tf.float32, [None, self.n_input])
        with tf.compat.v1.variable_scope(name):
            fc1 = layers.Dense(input_s, self.n_h1, activation=tf.nn.tanh, name='fc1')
            layers.Dropout(fc1, rate=0.1)
            fc2 = layers.Dense(fc1, self.n_h2, activation=tf.nn.tanh, name='fc2')
            layers.Dropout(fc2, rate=0.1)
            reward = layers.Dense(fc2, 1, name='reward')
        theta = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

        return input_s, reward, theta

    def get_rewards(self, states):
        rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
        return rewards

    def apply_grads(self, feat_map, grad_r):
        grad_r = np.reshape(grad_r, [-1, 1])
        feat_map = np.reshape(feat_map, [-1, self.n_input])
        _, grad_theta, l2_loss, grad_norms = self.sess.run(
            [self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
            feed_dict={self.grad_r: grad_r, self.input_s: feat_map}
        )

        return grad_theta, l2_loss, grad_norms

    def rewards(self, n_iters=2000):
        mu_D = self.demo_savf()
        # print(np.sum(mu_D))

        feat_map = self.mdp.feature_matrix(deep=True)
        # print('feat_map', feat_map.shape, feat_map)
        policy_container = []

        for iteration in range(n_iters):
            if iteration % (n_iters / 5) == 0:
                print('iteration: {}'.format(iteration))

            rewards = self.normalize(self.get_rewards(feat_map))

            # value iteration
            Q_init = np.random.uniform(size=(self.mdp.n_states, self.mdp.n_actions))
            policy, Q = solve(self.mdp, Q_init, rewards)
            policy_container.append(policy)
            mu_exp = self.state_visitation_frequency(policy)

            grad_r = mu_D - mu_exp
            print('grad_r norm', np.linalg.norm(grad_r))

            grad_theta, l2_loss, grad_norm = self.apply_grads(feat_map, grad_r)
            print('l2_loss {}'.format(l2_loss))
            if np.linalg.norm(grad_r) < 0.01 or l2_loss < 1e-4:
                print('convergence')
                break

        rewards = self.get_rewards(feat_map)
        print(rewards)

        return rewards

    def generate_samples(self, policy, n_iters=10000):
        """
        :return:
        """
        episodes_container = []

        for state in self.mdp.init:
            # if i % 100000 == 0:
            #    print('sampled ', i, ' episode')

            episode = state[-1]
            t = 0

            while True:
                action = choose_action(self.mdp.states_idx[state], policy)[0]
                next_state = self.mdp.step(state, action)

                # reward = self.mdp.get_reward((state, action))
                t += 1
                state = next_state
                # print(episode, state, type(episode), type(state))
                episode += str(state[-1])

                if t == self.n_steps-1:
                    break

            episodes_container.append(episode)

        return episodes_container

    def normalize(self, mat):
        mat = np.reshape(mat, (self.mdp.n_states, self.mdp.n_actions))
        min_val = np.min(mat, axis=1).reshape((self.mdp.n_states, 1))
        max_val = np.max(mat, axis=1).reshape((self.mdp.n_states, 1))
        # diff = max_val - min_val

        mat_ = 8 * (mat-min_val) / (max_val - min_val) - 4
        mat_[np.isnan(mat_)] = 0
        # print('normalization {}'.format(mat_))

        return mat_.reshape((self.mdp.n_states * self.mdp.n_actions, ))
