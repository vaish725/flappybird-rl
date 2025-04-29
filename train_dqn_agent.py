import os
import sys
import numpy as np
import pygame
import torch
import matplotlib.pyplot as plt
#add FlapPyBird to path
sys.path.append(os.path.abspath("../FlapPyBird"))

from env_flappy import FlappyBirdEnv
from agent_dqn import DQNAgent
import logging
from datetime import datetime

#create logs folder if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

#generate unique log filename
log_filename = f'logs/train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

#configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

#define logprint instead of print
def logprint(*args, **kwargs):
    logging.info(' '.join(map(str, args)))

#initialize pygame
pygame.init()
screen_width = 288
screen_height = 512
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird RL DQN Agent')
clock = pygame.time.Clock()

#initialize environment
env = FlappyBirdEnv()

#initialize agent
state_dim = 3  #bird_y, bird_vel_y, relative_pipe_y
action_dim = 2  #flap or no flap
agent = DQNAgent(state_dim, action_dim)

#number of episodes to train
num_episodes = 500

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
    total_reward = 0
    done = False
    score = 0
    passed_pipe = False

    while not done:
        for event in pygame.event.get():
            #check for quit event
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        #select action using agent
        action = agent.select_action(obs)

        #apply action and get next observation, reward, done
        next_obs, reward, done = env.step(action)

        #store experience
        agent.store_transition(obs, action, reward, next_obs, done)

        #train agent
        agent.train_step()

        #accumulate reward
        total_reward += reward

        #update observation
        obs = next_obs

        #check if bird crosses the pipe
        next_pipe = env.game.upper_pipes_list[0]
        if env.game.player_x > next_pipe['x'] + env.game.pipe_width and not passed_pipe:
            score += 1
            passed_pipe = True

        #reset passed_pipe when new pipe appears
        if next_pipe['x'] + env.game.pipe_width < env.game.player_x and len(env.game.upper_pipes_list) > 0:
            passed_pipe = False

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

        #draw score text
        font = pygame.font.SysFont(None, 40)
        score_surface = font.render(f"Score: {score}", True, white)
        score_rect = score_surface.get_rect(center=(screen_width//2, 30))
        screen.blit(score_surface, score_rect)

        #update display
        pygame.display.update()

        #control frame rate
        clock.tick(30)

    #update target network every 10 episodes
    if episode % 10 == 0:
        agent.update_target_network()

    #store total reward for this episode
    episode_rewards.append(total_reward)

    #print episode summary
    logprint(f"Episode {episode+1}: Total Reward = {total_reward}, Score = {score}, Epsilon = {agent.epsilon:.3f}")

#after all episodes, print final result
average_reward = np.mean(episode_rewards)
logprint(f"\nAverage Reward over {num_episodes} episodes: {average_reward}")

#save final trained model
torch.save(agent.policy_net.state_dict(), "saved_model.pth")
logprint("\nModel saved as saved_model.pth")

#plot reward curve
plt.figure(figsize=(10,6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress: Reward vs Episode')
plt.grid()
plt.savefig('reward_plot.png')
plt.show()

logprint("\nReward plot saved as reward_plot.png")
