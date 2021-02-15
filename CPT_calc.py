import numpy as np 

"""
Some tourists are curious if there are fish in a nearby lake. They are unable to observe whether this is true or not by staring into the lake. 
However, they can observe whether or not there are birds nearby that affect the presence of fish. 
Based on their instincts, the tourists propose the following domain theory:
1. The prior probability of fish nearby (that is, without any observation) is 0.5.
2. The probability of fish nearby on day t is 0.8 given there are fish nearby on day t − 1, and 0.3
if not.
3. The probability of birds nearby on day t if there are fish nearby on the same day is 0.75, and
0.2 if not.
The following evidence is given
• e1 = {birds nearby}
• e2 = {birds nearby}
• e3 = {no birds nearby}
• e4 = {birds nearby}
• e5 = {no birds nearby} 
• e6 = {birds nearby}
We will denote the state variable for fish nearby on day t by Xt.
"""


class HMM_fish:
    def __init__(self, starting_prob_fish, prob_depen_fish, prob_depen_birds, observations):

        self.starting_prob_fish = starting_prob_fish # prob. of fish at t = 0
        self.prob_depen_fish = prob_depen_fish # prob. of fish if/if not fish present at time t-1
        self.prob_depen_birds = prob_depen_birds # prob. of birds if/if not fish present at time t-1
        self.observations = observations # evidence of birds/not birds at time t
        #self.num_t = num_t, rather included in each method

    def filtering(self, num_t):

        if num_t > len(self.observations):
            return print("Not enough evidence!")

        X_t = np.array([]) # the (binary) prob. distribution at time t, list

        T_first_row = self.prob_depen_fish.copy()
        T_second_row = self.prob_depen_fish.copy()
        T_second_row.reverse()
        T_list = [T_first_row, T_second_row]

        T = np.array(T_list)
        #print(T)

        obs_true_list = [[self.prob_depen_birds[0], 0], [0, self.prob_depen_birds[1]]]
        obs_false_list = [[1-self.prob_depen_birds[0], 0], [0, 1-self.prob_depen_birds[1]]]
        O_true = np.array(obs_true_list)
        O_false = np.array(obs_false_list)
        #print(O_true)

        vector = self.starting_prob_fish

        for t in range(num_t):
            vector = np.matmul(vector, T)
            if self.observations[t] == 1:
                vector = np.matmul(vector, O_true)
            elif self.observations[t] == 0:
                vector = np.matmul(vector, O_false)
            else:
                return print('Feil input i observasjoner')

        vector = self.normalize(vector)

        return vector


def umbrella_ex():

    HMM_umbrella = HMM_fish([0.5, 0.5], [0.7, 0.3], [0.9, 0.2], [1, 1])
    result = HMM_umbrella.filtering(2)

    return result


def main():

    return None


if __name__ == '__main__':
    print(main())

