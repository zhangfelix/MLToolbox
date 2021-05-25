# coding=utf-8
import torch
import numpy as npy
import tools.distance as dis
import tools.data as tdata

class Knn:
    vote_methods = ("normal","weight_vote")
    def __init__(self, vote_method="normal", filename=null, data=null, k=1, distance=dis.euler_distance):
        if vote_method not in self.vote_methods:
            self.vote_method = "normal"
        else:
            self.vote_method = vote_method
        if filename != null:
            self.date = tdata.date_loader(filename)
        else:
            self.data = data
        self.k = k

    def calculate(self, new_points):
        klist = k_nestest(new_points)
        vote(klist)

    def k_nestest(self, new_points):
        # todo: calculate distance with different methods.
        pass

    def vote(self, klist):
        # vote which class the new point belong to.
        pass
