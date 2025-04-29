import numpy as np
import random

#flappy bird environment simulator
class FlappyBirdEnv:
    def __init__(self, gravity=0.5, flap_power=8, pipe_gap=100, pipe_spacing=200, pipe_width=80):
        #initialize basic physics and game config
        self.gravity = gravity                  #how fast the bird falls
        self.flap_power = flap_power            #how strong the flap is (negative velocity)
        self.pipe_gap = pipe_gap                #vertical gap between upper and lower pipes
        self.pipe_spacing = pipe_spacing        #not used here but useful for multiple pipes
        self.pipe_width = pipe_width            #width of each pipe
        self.bird_y = 250                       #bird's vertical starting position
        self.bird_velocity = 0                  #initial vertical speed
        self.pipe_x = 400                       #pipe starts on the right side
        self.pipe_y = self._generate_pipe_y()   #random vertical position for pipe gap
        self.ground_y = 512                     #ground level (bottom of screen)
        self.bird_radius = 12                   #size of bird (used for collision)
        self.done = False                       #flag to check if game is over

    def _generate_pipe_y(self):
        #randomly set the vertical center of the pipe gap
        return random.randint(100, 400)

    def reset(self):
        #reset environment to the initial state
        self.bird_y = 250                       #reset bird height
        self.bird_velocity = 0                  #reset bird speed
        self.pipe_x = 400                       #reset pipe to starting x
        self.pipe_y = self._generate_pipe_y()   #generate new pipe gap height
        self.done = False                       #game is not over
        return self._get_observation()          #return initial observation

    def _get_observation(self):
        #return partial observation for the agent
        relative_pipe_y = self.pipe_y - self.bird_y  #vertical distance to pipe center
        return np.array([self.bird_y, self.bird_velocity, relative_pipe_y], dtype=np.float32)

    def _check_collision(self):
        #determine if the bird has crashed into a pipe or the ground
        bird_top = self.bird_y - self.bird_radius     #top edge of bird
        bird_bottom = self.bird_y + self.bird_radius  #bottom edge of bird
        pipe_top = self.pipe_y - self.pipe_gap / 2    #top edge of pipe gap
        pipe_bottom = self.pipe_y + self.pipe_gap / 2 #bottom edge of pipe gap

        #check if bird is at pipe x-position and not in the gap
        hit_pipe = (self.pipe_x < self.bird_radius + 50) and not (pipe_top <= self.bird_y <= pipe_bottom)
        
        #check if bird hit the ground or flew off the top
        hit_ground = self.bird_y >= self.ground_y or self.bird_y <= 0

        return hit_pipe or hit_ground

    def step(self, action):
        #apply an action: 1 = flap, 0 = no flap
        if action == 1:
            self.bird_velocity = -self.flap_power  #flap: give upward speed
        else:
            self.bird_velocity += self.gravity     #no flap: gravity increases downward speed

        self.bird_y += self.bird_velocity          #update bird's position
        self.pipe_x -= 3                           #move the pipe left (towards the bird)

        #if pipe is off screen, reset it to the right with a new gap
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = 400
            self.pipe_y = self._generate_pipe_y()

        self.done = self._check_collision()        #check if the bird hit something

        reward = -100 if self.done else 1          # -100 if crash, else +1 for surviving

        observation = self._get_observation()      #get new observation

        return observation, reward, self.done      #return results of the action
