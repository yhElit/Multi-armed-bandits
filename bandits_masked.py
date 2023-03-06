# This script requires
import numpy
import math
import matplotlib.pyplot as plt
import copy

"""
The aim of this script is to simulate a situation in which an agent repeatedly has to choose
betweeen the same options, of which there are finitely many.
Each option leads to a reward drawn from a fixed Gaussian distribution: a multi-armed bandit with Gaussian rewards.
The aim of the agent is to repeatedly interact with the environment to learn which arm to pull.
To do so the agent maintains numerical estimates for the value of each arm (action-value method).
"""


class GaussianBandit(object):
    """
    This represents a multi-armed bandit whose arms have normally distributed rewards.
    
    Attributes
    ----------
    means_list : list[float]
        An ordered list of the extected values of the arms.
    stdev_list : list[float]
        An ordered list of the standard deviations of the arms.
    q_values : list[float]
        An ordered list of q-values for the arms.
    action_count : list[int]
        An ordered list whose entries are the number of times an arm has been pulled.
    action_history : list[int]
        A list of the actions that have been taken in the past.
    reward_history : list[float]
        A list of the rewards that have been observed in the past.
    q_history : list[list[float]]
        A list of the historical q-value vectors.        
    """

    def __init__(self, means_list, stdev_list=None):
        if stdev_list != None and len(means_list) != len(stdev_list):
            raise ValueError('The list of means and the list of standard deviations are not of equal lengths.')
        # attributes describing the bandit
        self.means_list = means_list
        if stdev_list is None:
            stdev_list = [1] * len(means_list)
        self.stdev_list = stdev_list
        # attributes describing state of knowledge
        self.q_values = [0] * len(means_list)
        self.action_count = [0] * len(means_list)
        # attributes describing a learning history
        self.action_history = []
        self.reward_history = []
        self.q_history = [[0] * len(means_list)]

    @staticmethod
    def _update_average(old_average, num_obs, new_obs):
        """
        Helper function for online averaging.
        """
        # online averaging formula
        return old_average * num_obs / (num_obs + 1) + new_obs / (num_obs + 1)

    def _num_arms(self):
        "Returns the number of arms."
        return len(self.means_list)

    def reset(self, reset_q_values: bool = True):
        """
        Resets all history attributes, and potentially also the q-values and action-counts, of the object in-place.
        """
        self.action_history = []
        self.reward_history = []
        self.q_history = [[0] * len(self.means_list)]
        if reset_q_values:
            self.q_values = [0] * len(self.means_list)
            self.action_count = [0] * len(self.means_list)

    def reward(self, arm: int):
        """
        Returns the reward of the respective arm.
        Does not modify the object.
        """
        if arm < 0 or arm > self._num_arms() - 1:
            raise ValueError("This arm does not exist.")
        from numpy.random import normal
        return normal(self.means_list[arm], self.stdev_list[arm])

    def eps_greedy_action(self, epsilon: float):
        """
        Returns the index of an arm chosen according to epsilon-greedy action-choice with respect to the q-values of the object.
        Does not modify the object.
        """
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon is not in the interval [0,1].")

        if numpy.random.random() < epsilon:
            # exploration: choose a random arm
            action = numpy.random.choice(self._num_arms())
        else:
            # exploitation: choose the arm with highest q-value
            max_q = max(self.q_values)
            action = numpy.random.choice([i for i, q in enumerate(self.q_values) if q == max_q])
        return action

    def ucb_action(self, exp_factor: float = 1):
        """
        Returns the index of an arm chosen according to UCB action-choice with respect to the q-values and action-counts of the object.
        Does not modify the object.
        """
        if exp_factor < 0:
            raise ValueError("The exploration factor can not be negative.")

        total_actions = sum(self.action_count)
        if 0 in self.action_count:
            # if any arm has not been played, play it
            action = self.action_count.index(0)
        else:
            # compute UCB score for each arm
            ucb_scores = [self.q_values[i] + exp_factor * math.sqrt(math.log(total_actions) / self.action_count[i]) for
                          i in range(self._num_arms())]
            # choose the arm with highest UCB score
            max_ucb = max(ucb_scores)
            action = numpy.random.choice([i for i, ucb in enumerate(ucb_scores) if ucb == max_ucb])
        return action

    def play_eps_greedy(self, num_steps=100, epsilon=0.1):
        """
            Simulates playing the bandit for a given number of steps using epsilon-greedy action-choice.
            Modifies the object.
            """
        self.reset()
        for i in range(num_steps):
            action = self.eps_greedy_action(epsilon)
            reward = self.reward(action)
            self.action_history.append(action)
            self.reward_history.append(reward)
            self.action_count[action] += 1
            self.q_values[action] = self._update_average(self.q_values[action], self.action_count[action] - 1, reward)
            self.q_history.append(list(self.q_values))

    def play_ucb(self, num_steps=100, exp_factor=1):
        """
            Simulates playing the bandit for a given number of steps using UCB action-choice.
            Modifies the object.
            """
        self.reset()
        for i in range(num_steps):
            action = self.ucb_action(exp_factor)
            reward = self.reward(action)
            self.action_history.append(action)
            self.reward_history.append(reward)
            self.action_count[action] += 1
            self.q_values[action] = self._update_average(self.q_values[action], self.action_count[action] - 1, reward)
            self.q_history.append(list(self.q_values))

    """
    plot the historical information of both methods in one plot
    """

    def my_plot_action_history(self, ax, method_title):
        """
        Displays a graphical representation of the action-history to standard output.
        The fraction of times an action has been chosen in the past is displayed.
        """
        ax.set_title(method_title + "Action History")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Fraction of action in past")
        for i in set(self.action_history):
            actions_chosen = [x == i for x in self.action_history]
            actions_counts = [sum(actions_chosen[0:j + 1]) for j in range(len(actions_chosen))]
            actions_percent = [actions_counts[j] / (j + 1) for j in range(len(actions_counts))]
            ax.plot(actions_percent)

    def my_plot_q_history(self, ax, method_title):
        """
        Displays a graphical representation of the q-value-history to standard output.
        """
        ax.set_title(method_title + "Q-Value History")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Q-Value")
        ax.plot(self.q_history)

    def my_plot_reward_history(self, ax, method_title):
        ax.set_title(method_title + "Reward History")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")
        ax.scatter(range(len(self.reward_history)), self.reward_history, s=0.3)

    def my_plot_all(self, old_bandit):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 5))
        old_bandit.my_plot_action_history(axs[0, 0], method_title="Epsilon-greedy: ")
        old_bandit.my_plot_q_history(axs[0, 1], method_title="Epsilon-greedy: ")
        old_bandit.my_plot_reward_history(axs[0, 2], method_title="Epsilon-greedy: ")
        self.my_plot_action_history(axs[1, 0], method_title="Upper confidence bounds: ")
        self.my_plot_q_history(axs[1, 1], method_title="Upper confidence bounds: ")
        self.my_plot_reward_history(axs[1, 2], method_title="Upper confidence bounds: ")
        plt.tight_layout()
        plt.show()


def main():
    # initialize a multi-armed bandit with three arms with known means and standard deviations
    bandit = GaussianBandit([1, 2, 3], [1, 3, 6])

    # define iterations and epsilon
    n_trials = 2000
    epsilon = 0.1

    # perform epsilon-greedy action-value learning
    bandit.play_eps_greedy(num_steps=n_trials, epsilon=epsilon)

    # save bandit
    old_bandit = copy.deepcopy(bandit)

    # reset the multi-armed bandit
    bandit.reset()

    # define iterations and exp factor
    n_trials = 2000
    exp_factor = 1

    # perform action-value learning with upper confidence bounds
    bandit.play_ucb(num_steps=n_trials, exp_factor=exp_factor)

    # plot the historical information of both methods in one plot
    bandit.my_plot_all(old_bandit=old_bandit)


if __name__ == '__main__':
    main()
