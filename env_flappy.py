import os
import sys
import numpy as np
import pygame

#add FlapPyBird to path so we can import its files
sys.path.append(os.path.abspath("../FlapPyBird"))

from src.flappy_env import FlappyBirdGame

class FlappyBirdEnv:
    def __init__(self):
        #initialize game instance
        self.game = FlappyBirdGame()

        #initialize done flag
        self.done = False
        self.current_obs = None

    def reset(self):
        #resetting the entire game
        self.game.reset_game()

        #updating the done flag
        self.done = False

        #getting the initial observation
        self.current_obs = self._get_observation()

        return self.current_obs

    def _get_observation(self):
        #getting the bird's vertical position and velocity
        bird_y = self.game.player_y
        bird_vel = self.game.player_vel_y

        #finding the next pipe ahead of the bird
        pipes_ahead = [p for p in self.game.upper_pipes_list if p['x'] + self.game.pipe_width > self.game.player_x]
        next_pipe = pipes_ahead[0] if pipes_ahead else {'y': 200, 'x': 300}

        #calculating the vertical distance to the center of the pipe gap
        pipe_center_y = next_pipe['y'] + self.game.pipe_gap_size / 2
        relative_pipe_y = pipe_center_y - bird_y

        #return as np array
        return np.array([bird_y, bird_vel, relative_pipe_y], dtype=np.float32)

    def step(self, action):
        #applying action
        if action == 1:
            self.game.flap()
        else:
            self.game.no_flap()

        #moving forward one frame
        reward, done = self.game.play_frame()
        self.done = done

        #getting the next observation
        obs = self._get_observation()

        #assigning the rewards
        reward = -100 if done else 1

        return obs, reward, done
