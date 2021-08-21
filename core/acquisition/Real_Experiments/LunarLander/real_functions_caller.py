import numpy as np
import gym


def funcval_decorator(func, terrain):
    def func_val(design_var, true_val=False):
        return func(design_var, terrain)
    return func_val


class constrained_LunarLanderBenchmark():
    def __init__(self, m_terrains, minimum_reward):
        self.m_terrains = m_terrains
        self.env = gym.make('LunarLander-v2')
        self.minimum_reward = minimum_reward

    def f(self, design_var, true_val=False):

        simulations = self.simulator(design_var)
        mean_response = np.mean(simulations, axis=1)
        return mean_response/50.0

    def c(self, design_var, true_val=True):

        simulations = self.simulator(design_var)
        min_response = np.min(simulations, axis=1)
        return -(min_response - self.minimum_reward)/50.0

    def simulator(self, design_var, true_val=False):

        design_var = np.atleast_2d(design_var)
        self.last_evaluated_design_var = design_var

        m = self.m_terrains

        terrains = range(m)
        averaged_cum_reward_list = []
        cum_reward_matrix = []
        # iterate over all designs.
        for xvar in design_var:

            # for each design, show m different terrains
            cum_reward_list = []
            for i_episode in range(m):

                self.env.seed(seed=terrains[i_episode])  # terrain landscape control
                observation = self.env.reset()
                cum_reward = 0
                counter = 0
                done = False
                while not done:

                    counter+=1
                    action = self.heuristic_Controller(observation, xvar)  # controller
                    observation, reward, done, info = self.env.step(action)  # obervation from environment
                    cum_reward += reward

                cum_reward_list.append(cum_reward)

                self.env.close()

            cum_reward_matrix.append(cum_reward_list)
            averaged_cum_reward = cum_reward_list
            averaged_cum_reward_list.append(averaged_cum_reward)

        return np.array(averaged_cum_reward_list)

    # def c_builder(self):
    #
    #     m = self.m_terrains
    #     terrains = range(m)
    #     constrained_functions = []
    #
    #     for terr in terrains:
    #         constrained_functions.append(funcval_decorator(self.c, terr))
    #
    #     return constrained_functions
    #
    # def c(self, design_var, terrain):
    #
    #     reward_list = self.f(design_var, terrain)
    #     return -(np.array(reward_list) - (self.minimum_reward/50))

    def heuristic_Controller(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a

# LunarLanderBench = constrained_LunarLanderBenchmark(m_terrains=3, minimum_reward= 100)

# weight = np.random.random((4,12))*2
# funval = LunarLanderBench.f(weight)

# print(funval)
#
# constraints = LunarLanderBench.c_builder()
#
#
# print(constraints[1](weight))