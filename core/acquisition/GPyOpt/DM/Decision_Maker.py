import numpy as np
#import matplotlib.pyplot as plt
#import pygmo as pg
from itertools import permutations


class DM:
    def __init__(self, random=True, l=None):
        self.Data = []
        if random == True:
            l = np.random.uniform()
        self.w = np.array([l, 1 - l])

    def winner(self, Data , w = np.array([])):
        self.Data = Data
        if w.size > 0:
            U = self.calculate_utility(Data, w)
        else:
            U = self.calculate_utility(Data, self.w)
        winner_idx = np.argmax(U)
        winner_pnt = Data[winner_idx]
        return winner_idx

    def __call__(self, Data, weight = None):

        if weight == None:

            winner_idx = self.winner(Data)
        else:

            w = np.array([weight, 1 - weight])
            winner_idx = self.winner(Data, w)
        duals = range(Data.shape[0])
        comb = permutations(duals, 2)
        comb_list = []
        for i in list(comb):
            comb_list.append(np.array(i))
        comb_array = np.array(comb_list)

        bool_aux = np.hstack(comb_array[:, 0] == winner_idx)
        winned_duals_idx = comb_array[bool_aux]

        winned_duals_Data = Data[winned_duals_idx]

        return Data[winner_idx], winned_duals_Data

    def calculate_utility(self, F, w):
        assert len(self.w) == F.shape[1];
        "weights must have same dim of objective space"
        return np.sum(F * w, axis=1)
