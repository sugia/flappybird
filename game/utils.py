import numpy as np 
import tensorflow as tf
import random 
import pygame

FPS = 60
SCREEN_WIDTH = 576
SCREEN_HEIGHT = 512
BACKGROUND_SPEED_X = 20
PIPE_PENDING_GAP = 256
INIT_BIRD_X = 80
INIT_BIRD_Y = 256
PIPE_INTERSPACE = 140
PIPE_HEIGHT_DELTA = 50

pygame.init()
FPS_CLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

def LoadImages():
  IMAGES = {}

  IMAGES['numbers'] = (
      pygame.image.load('img/0.png').convert_alpha(),
      pygame.image.load('img/1.png').convert_alpha(),
      pygame.image.load('img/2.png').convert_alpha(),
      pygame.image.load('img/3.png').convert_alpha(),
      pygame.image.load('img/4.png').convert_alpha(),
      pygame.image.load('img/5.png').convert_alpha(),
      pygame.image.load('img/6.png').convert_alpha(),
      pygame.image.load('img/7.png').convert_alpha(),
      pygame.image.load('img/8.png').convert_alpha(),
      pygame.image.load('img/9.png').convert_alpha()
  )

  # background image
  IMAGES['background'] = pygame.image.load('img/background.png')
  IMAGES['player'] = pygame.image.load('img/bird.png')
  IMAGES['pipe_upper'] = pygame.image.load('img/pipe_upper.png')
  IMAGES['pipe_lower'] = pygame.image.load('img/pipe_lower.png')

  return IMAGES

IMAGES = LoadImages()

PLAYER_WIDTH = IMAGES['player'].get_width()
PLAYER_HEIGHT = IMAGES['player'].get_height()
PIPE_WIDTH = IMAGES['pipe_upper'].get_width()
PIPE_HEIGHT = IMAGES['pipe_upper'].get_height()

sess = tf.Session()

class Bird:
  def __init__(self, input_nodes, hidden_nodes, output_nodes):
    self.input_signal = tf.placeholder(shape=[None, input_nodes], dtype=tf.float32)
    self.w1 = self.InitVariable(shape=[input_nodes, hidden_nodes])
    self.b1 = self.InitVariable(shape=[hidden_nodes])
    self.hidden_layer = tf.matmul(self.input_signal, self.w1) + self.b1

    self.w2 = self.InitVariable(shape=[hidden_nodes, output_nodes])
    self.b2 = self.InitVariable(shape=[output_nodes])
    self.output_layer = tf.sigmoid(tf.matmul(self.hidden_layer, self.w2) + self.b2)

    self.output_signal = tf.placeholder(shape=[None, output_nodes], dtype=tf.float32)
    self.loss = tf.reduce_mean(tf.square(self.output_layer - self.output_signal))
    self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    self.x = INIT_BIRD_X
    self.y = INIT_BIRD_Y

    self.alive = True 
    self.gravity = 0
    self.velocity = 15
    self.jump = -45

  def InitVariable(self, shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=10))

  def Train(self, input_signal, output_signal):
    for i in range(100):
      sess.run(self.train, feed_dict = {self.input_signal: input_signal, self.output_signal: output_signal})

  def Predict(self, input_signal):
    return sess.run(self.output_layer, feed_dict = {self.input_signal: input_signal})

  def Flap(self):
    self.gravity = self.jump 

  def Update(self):
    self.gravity += self.velocity
    self.y += self.gravity

  def IsDead(self, pipes):
    # a bird is dead when it gets out of screen
    if self.y >= SCREEN_HEIGHT or self.y + PLAYER_HEIGHT <= 0:
      return True 

    for pipe in pipes:
      if not (self.x + PLAYER_WIDTH < pipe.x or \
              self.x > pipe.x + PIPE_WIDTH or \
              self.y + PLAYER_HEIGHT < pipe.y_start or \
              self.y > pipe.y_start + pipe.y_end):
         return True

class Pipe:
  def __init__(self, x, y_start, y_end):
    self.x = x
    self.y_start = y_start
    self.y_end = y_end
    self.passed = False

  def Update(self):
    self.x -= BACKGROUND_SPEED_X

  def IsOut(self):
    if self.x + PIPE_WIDTH < 0:
      return True 

  def IsUpperPipe(self):
    return self.y_start == 0

  def IsLowerPipe(self):
    return self.y_start != 0

class GameState:
  def __init__(self):
    self.score = 0

    self.pipes = []
    self.birds = []
    
    self.pipe_pending = 0
    self.gen = []
    self.alives = 0
    self.background_x = 0
    self.max_score = 0

  def Start(self):
    self.pipe_pending = 0
    self.score = 0
    self.pipes = []
    self.birds = []
    self.gen = 1

    for i in range(self.gen):
      bird = Bird(input_nodes=2, hidden_nodes=8, output_nodes=1)
      self.birds.append(bird)

    self.alives = len(self.birds)

    init = tf.global_variables_initializer()
    sess.run(init)

    self.Display()

  def Update(self):
    self.background_x += BACKGROUND_SPEED_X
    pipe_smallest_x = SCREEN_WIDTH
    next_upper_pipe_y_end = 0
    for pipe in self.pipes:
      if pipe.IsUpperPipe() and pipe.x + PIPE_WIDTH > INIT_BIRD_X and pipe.x < pipe_smallest_x:
        next_upper_pipe_y_end = pipe.y_end
        pipe_smallest_x = pipe.x
      if pipe.IsUpperPipe() and not pipe.passed and pipe.x < INIT_BIRD_X:
        pipe.passed = True
        self.score += 1


    # do actions for each bird and check their statuses
    for i in range(len(self.birds)):
      if self.birds[i].alive:
        bird_height_rate = 1.0 * (self.birds[i].y + PLAYER_HEIGHT) / SCREEN_HEIGHT
        interspace_height_rate = 1.0 * (next_upper_pipe_y_end + PIPE_INTERSPACE / 1.3) / SCREEN_HEIGHT
        input_signal = [[bird_height_rate, interspace_height_rate]]
        output_signal = [[0 if bird_height_rate < interspace_height_rate else 1]]

        self.birds[i].Train(input_signal, output_signal)

        #prediction = self.birds[i].Predict(input_signal)
        #print(output_signal, prediction)
        #if prediction > 0.5:
        #  self.birds[i].Flap()
        if bird_height_rate > interspace_height_rate:
          self.birds[i].Flap()

        self.birds[i].Update()
        
        if self.birds[i].IsDead(self.pipes):
          self.birds[i].alive = False 
          self.alives -= 1

    # move each pipe and add pipe if necessary
    out_index = []
    for i in range(len(self.pipes)):
      self.pipes[i].Update()
      if self.pipes[i].IsOut():
        out_index.append(i)
    for index in out_index[::-1]:
      self.pipes.pop(index)

    if self.pipe_pending == 0:
      upper_pipe_y_end = round(random.random() * (SCREEN_HEIGHT - PIPE_HEIGHT_DELTA * 2 - PIPE_INTERSPACE)) + PIPE_HEIGHT_DELTA
      self.pipes.append(Pipe(x = SCREEN_WIDTH, y_start = 0, y_end = upper_pipe_y_end))
      self.pipes.append(Pipe(x = SCREEN_WIDTH, y_start = upper_pipe_y_end + PIPE_INTERSPACE, y_end = SCREEN_HEIGHT))

    self.pipe_pending += BACKGROUND_SPEED_X
    if self.pipe_pending >= PIPE_PENDING_GAP:
      self.pipe_pending = 0

    # check score and show score
    if self.score > self.max_score:
      self.max_score = self.score 

    # check game status
    if self.IsEnd():
      self.Start()

    self.Display()

  def IsEnd(self):
    for bird in self.birds:
      if bird.alive:
        return False 
    return True 

  def Display(self): 
    SCREEN.blit(IMAGES['background'], (0, 0))

    for bird in self.birds:
      SCREEN.blit(IMAGES['player'], (bird.x, bird.y))

    for pipe in self.pipes:
      if pipe.IsUpperPipe():
        SCREEN.blit(IMAGES['pipe_upper'], (pipe.x, pipe.y_end - PIPE_HEIGHT))
      elif pipe.IsLowerPipe():
        SCREEN.blit(IMAGES['pipe_lower'], (pipe.x, pipe.y_start))
    
    self.ShowScore()
    pygame.display.update()
    FPS_CLOCK.tick(FPS)


  def ShowScore(self):
    # shows score in the middle of screen
    scoreDigits = [int(x) for x in list(str(self.score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREEN_WIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREEN_HEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()
































