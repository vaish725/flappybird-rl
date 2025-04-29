import os
import sys
import numpy as np
import pygame

#add FlapPyBird to path
sys.path.append(os.path.abspath("../FlapPyBird"))

from env_flappy import FlappyBirdEnv

#initialize pygame
pygame.init()
screen_width = 288
screen_height = 512
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird RL Manual Play')
clock = pygame.time.Clock()

#initialize environment
env = FlappyBirdEnv()

#number of episodes to run
num_episodes = 10

#initialize variables to track performance
episode_rewards = []

#colors
white = (255, 255, 255)
blue = (0, 150, 255)
green = (0, 255, 0)
black = (0, 0, 0)

for episode in range(num_episodes):
    #reset environment at the start of each episode
    obs = env.reset()
    score = 0
    passed_pipe = False

    total_reward = 0
    done = False

    while not done:
        action = 0  #default: no flap

        for event in pygame.event.get():
            #check for quit event
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            #check if space or up arrow is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    action = 1  #flap

        #apply action and get next observation, reward, done
        next_obs, reward, done = env.step(action)
        
        #check if bird crosses the pipe
        next_pipe = env.game.upper_pipes_list[0]
        if env.game.player_x > next_pipe['x'] + env.game.pipe_width and not passed_pipe:
            score += 1
            passed_pipe = True

        #reset passed_pipe when new pipe appears
        if next_pipe['x'] + env.game.pipe_width < env.game.player_x and len(env.game.upper_pipes_list) > 0:
            passed_pipe = False


        #accumulate reward
        total_reward += reward

        #fill background
        screen.fill(blue)

        #draw pipes
        for pipe in env.game.upper_pipes_list:
            pipe_rect_top = pygame.Rect(pipe['x'], 0, env.game.pipe_width, pipe['y'])
            pipe_rect_bottom = pygame.Rect(pipe['x'], pipe['y'] + env.game.pipe_gap_size, env.game.pipe_width, screen_height)
            pygame.draw.rect(screen, green, pipe_rect_top)
            pygame.draw.rect(screen, green, pipe_rect_bottom)

        #draw bird
        bird_x = env.game.player_x
        bird_y = env.game.player_y
        pygame.draw.circle(screen, black, (int(bird_x), int(bird_y)), 10)

        #draw floor
        pygame.draw.rect(screen, black, (0, screen_height - 20, screen_width, 20))

        #update display
        pygame.display.update()
        
        #draw the score text
        font = pygame.font.SysFont(None, 40)
        score_surface = font.render(f"Score: {score}", True, white)
        score_rect = score_surface.get_rect(center=(screen_width//2, 30))
        screen.blit(score_surface, score_rect)


        #control frame rate
        clock.tick(30)

    #store total reward for this episode
    episode_rewards.append(total_reward)

    #print episode summary
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

#after all episodes, print average reward
average_reward = np.mean(episode_rewards)
print(f"\nAverage Reward over {num_episodes} episodes: {average_reward}")
