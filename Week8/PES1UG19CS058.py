import numpy as np
import math 
import os

class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        # TODO

        # Cardinality of the state space
        K = self.A.shape[0]
        # Initialize the priors with default (uniform dist) if not given by caller
        t_len = len(seq)
        temp1 = np.empty((K, t_len), 'd')
        temp2 = np.empty((K, t_len), 'B')

        # Initilaize the tracking tables from first observation
        temp1[:, 0] = self.pi * self.B[:, self.emissions_dict[seq[0]]]
        temp2[:, 0] = 0

        # Iterate throught the observations updating the tracking tables
        for i in range(1, t_len):
            temp1[:, i] = np.max(temp1[:, i - 1] * self.A.T * self.B[np.newaxis, :, self.emissions_dict[seq[i]]].T , 1)
            temp2[:, i] = np.argmax(temp1[:, i - 1] * self.A.T, 1)

        # Build the output, optimal model trajectory
        x = np.empty(t_len, 'B')
        x[-1] = np.argmax(temp1[:, t_len - 1])
        for i in reversed(range(1, t_len)):
            x[i - 1] = temp2[x[i], i]

       
        result = []

        for i in range(0,len(x)):
            result.append(self.states[x[i]])

        return result 