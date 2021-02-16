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


class HMM:
    def __init__(self, starting_prob, prob_depen_direct, prob_depen_indirect, observations):

        self.starting_prob = starting_prob # prob. of fish at t = 0
        self.prob_depen_direct = prob_depen_direct # prob. of fish if/if not fish present at time t-1
        self.prob_depen_indirect = prob_depen_indirect # prob. of birds if/if not fish present at time t-1
        self.observations = observations # evidence of birds/not birds at time t, t+1, t+2, ...
        #self.num_t = num_t, rather included in each method

    def normalize(self, vector):
        z = sum(vector)
        v = [element * 1 / z for element in vector]  # now sums to 1
        return v

    def filtering_loop(self, num_t):

        if num_t > len(self.observations):
            return print("Not enough evidence!")

        T_list = [self.prob_depen_direct.copy(), self.prob_depen_direct.copy()[::-1]]
        T = np.array(T_list)

        obs_true_list = [[self.prob_depen_indirect[0], 0], [0, self.prob_depen_indirect[1]]]
        obs_false_list = [[1-self.prob_depen_indirect[0], 0], [0, 1-self.prob_depen_indirect[1]]]
        O_true = np.array(obs_true_list)
        O_false = np.array(obs_false_list)

        vector = self.starting_prob # the (binary) prob. distribution at time t, list

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

    def filtering_recursive(self, num_t):

        if num_t > len(self.observations):
            return print("Not enough evidence!")

        T_list = [self.prob_depen_direct.copy(), self.prob_depen_direct.copy()[::-1]]
        T = np.array(T_list)

        obs_true_list = [[self.prob_depen_indirect[0], 0], [0, self.prob_depen_indirect[1]]]
        obs_false_list = [[1 - self.prob_depen_indirect[0], 0], [0, 1 - self.prob_depen_indirect[1]]]
        O_true = np.array(obs_true_list)
        O_false = np.array(obs_false_list)

        if num_t == 0:
            return self.starting_prob

        if self.observations[num_t-1] == 1:
            return self.normalize(np.matmul(np.matmul(O_true, T), self.filtering_recursive(num_t - 1)))
        elif self.observations[num_t-1] == 0:
            return self.normalize(np.matmul(np.matmul(O_false, T), self.filtering_recursive(num_t - 1)))
        else:
            return print('Feil input i observasjoner')

    def prediction(self, num_t):

            T_list = [self.prob_depen_direct.copy(), self.prob_depen_direct.copy()[::-1]]
            T = np.array(T_list)

            obs_true_list = [[self.prob_depen_indirect[0], 0], [0, self.prob_depen_indirect[1]]]
            obs_false_list = [[1 - self.prob_depen_indirect[0], 0], [0, 1 - self.prob_depen_indirect[1]]]
            O_true = np.array(obs_true_list)
            O_false = np.array(obs_false_list)

            vector = self.starting_prob  # the (binary) prob. distribution at time t, list

            for t in range(len(self.observations)):
                vector = np.matmul(vector, T)
                if self.observations[t] == 1:
                    vector = np.matmul(vector, O_true)
                elif self.observations[t] == 0:
                    vector = np.matmul(vector, O_false)
                else:
                    return print('Feil input i observasjoner')


            t_left = num_t - len(self.observations)
            while t_left:
                vector = np.matmul(vector, T)
                t_left -= 1

            vector = self.normalize(vector)

            return vector

    def backwarding(self, T, O_true, O_false, num_e, k):

        if k+1 > num_e:
            return [1, 1]
        elif self.observations[k] == 1:
            return np.matmul(np.matmul(T, O_true), self.backwarding(T, O_true, O_false, k+1, num_e))
        elif self.observations[k] == 0:
            return np.matmul(np.matmul(T, O_false), self.backwarding(T, O_true, O_false, k+1, num_e))
        else:
            print('Error')

    def smoothing(self, num_evidence, k):

            if k > len(self.observations):
                return print("No sense smoothing!")
            elif num_evidence > len(self.observations):
                return print("Not enough evidence!")

            # Skal finne:
            # P (X_k | e_1:t) = P(X_k | e_1:k) x P(e_k+1:t | X_k)

            T_list = [self.prob_depen_direct.copy(), self.prob_depen_direct.copy()[::-1]]
            T = np.array(T_list)
            obs_true_list = [[self.prob_depen_indirect[0], 0], [0, self.prob_depen_indirect[1]]]
            obs_false_list = [[1 - self.prob_depen_indirect[0], 0], [0, 1 - self.prob_depen_indirect[1]]]
            O_true = np.array(obs_true_list)
            O_false = np.array(obs_false_list)

            filtering_result = self.filtering_recursive(k)  # P(X_k | e_1:k)
            backward_result = self.backwarding(T, O_true, O_false, num_evidence, k) # P(e_k+1:t | X_k)

            end_result = [filtering_result[0]*backward_result[0], filtering_result[1]*backward_result[1]]
            print(filtering_result)
            print(backward_result)

            return self.normalize(end_result)

def umbrella_ex():

    HMM_umbrella = HMM([0.5, 0.5], [0.7, 0.3], [0.9, 0.2], [1, 1])
    #result1 = HMM_umbrella.filtering_loop(2)
    #result2 = HMM_umbrella.filtering_recursive(2)

    result = HMM_umbrella.smoothing(2, 1)

    return result


def exercise_1b(HMM, num_t):
    for t in range(num_t):
        print(HMM.filtering_recursive(t+1))
    return None

def exercise_1c(HMM, num_t):
    #print([len(HMM.observations)+1, num_t+1])
    for t in range(len(HMM.observations)+1, num_t+1):
        print(HMM.prediction(t))
    return None

def exercise_1d(HMM, num_e, t):
    for t in range(t):
        print(HMM.smoothing(num_e, t))
    return None


def main():

    print(umbrella_ex())
    HHM_exercise = HMM([0.5, 0.5], [0.8, 0.3], [0.75, 0.2], [1, 1, 0, 1, 0, 1])

    #exercise_1b(HHM_exercise, 6)
    #exercise_1c(HHM_exercise, 30)
    exercise_1d(HHM_exercise, 6, 6)

    return None



if __name__ == '__main__':
    main()

