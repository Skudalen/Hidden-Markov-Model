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

        if num_t > self.observations:
            return print("Not enough evidence!")

        X_t = np.array([]) # the (binary) prob. distribution at time t, list

        T = np.array([self.prob_depen_fish, 
                    self.prob_depen_fish.reverse()])

def main():

    A = np.array([[1,0],
                [0,1]])
    B = np.array([[2,2],
                [2,2]])

    return np.matmul(A, B)


if __name__ == main():
    main()

