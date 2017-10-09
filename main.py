import numpy as np 
import tensorflow as tf 
from game import utils

# TODO(v2): upgrade with tensorflow
# TODO(v3): upgrade with genetic algorithm(neuro evolution)
if __name__ == '__main__':
  game = utils.GameState()
  game.Start()
  while True:
    game.Update()
