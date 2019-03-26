"""
Agent
"""
import os
import re
from collections import namedtuple
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
import matplotlib.pyplot as plt

Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])

class FNAgent():
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False

    def save(self, mode_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

# step1
def foo():
    return 1
foo()

# step2
