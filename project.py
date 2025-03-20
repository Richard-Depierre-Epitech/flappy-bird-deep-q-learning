import flappy_bird_gymnasium
import gymnasium

# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
env = gymnasium.make("CartPole-v1", render_mode="human")

number_of_episodes = 1

obs, _ = env.reset()
while True:
    # Actions:
    # 0 - do nothing
    # 1 - flap
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Rendering:
    env.render()
    # Rewards:
    #+0.1 - every frame it stays alive
    #+1.0 - successfully passing a pipe
    #-1.0 - dying
    #−0.5 - touch the top of the screen

    # Checking if the player is still alive
    if terminated:
        break

env.close()