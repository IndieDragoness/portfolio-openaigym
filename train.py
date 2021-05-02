import random
import sys
import gym
import numpy as np
# Tensorflow model building: https://www.tensorflow.org/api_docs/python/tf/keras/Model
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
# Keras Reinforcement Learning models: https://keras-rl.readthedocs.io/en/latest/ (DQNAgent in this case)
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy # Choose between Value & Policy based RL: Policy
from rl.memory import SequentialMemory # Allows our agent to 'remember' things

def train():
    """ Setup training.
    """
    # Initialize OpenAI Gym Environment
    env = gym.make('CartPole-v0')
    states = env.observation_space.shape[0] # Observing all the states available in our environment
    actions = env.action_space.n # Specific number of actions available
    episodes = 10

    # Build the neural network model
    model = build_model(states, actions)
    model.summary() # Print the shape of the neural network

    # Build the Agent/DQN Model
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # Save the fitted weights for the model
    dqn.save_weights("dqn_weights.h5f", overwrite=True)

def test_random_input():
    """ Test environment with random inputs (no neural network).
    """
    # Initialize OpenAI Gym Environment
    env = gym.make('CartPole-v0')
    states = env.observation_space.shape[0] # Observing all the states available in our environment
    actions = env.action_space.n # Specific number of actions available
    episodes = 10

    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0

        # Continuously take steps until episode completes
        while not done:
            env.render()
            action = random.choice([0,1])
            n_state, reward, done, info = env.step(action)

            # Accumulate reward
            score+=reward
        print('Episode: {} Score: {}'.format(episode, score))

    # Clean up the environment
    env.close()

def test(model_filepath):
    """ Test environment with trained neural network. Compare to test_random_input().
    """
    # Initialize OpenAI Gym Environment
    env = gym.make('CartPole-v0')
    states = env.observation_space.shape[0] # Observing all the states available in our environment
    actions = env.action_space.n # Specific number of actions available
    episodes = 10

    # Load model
    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Load the weights
    dqn.load_weights("dqn_weights.h5f")


    scores = dqn.test(env, nb_episodes=5, visualize=True)
    print(np.mean(scores.history['episode_reward']))

def build_model(states, actions):
    """ Utilize Tensorflow to build the model.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(1, states))) # Pass through states at the top
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear')) # Pass out actions at the bottom
    return model

def build_agent(model, actions):
    """ Utilize Keras-RL to build the Agent.
    """
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)

    dqn = DQNAgent(model=model,
                   memory=memory,
                   policy=policy,
                   nb_actions=actions,
                   nb_steps_warmup=10,
                   target_model_update=1e-2)

    return dqn

if __name__ == "__main__":
    if "train" in sys.argv[1]:
        train()
    elif "random" in sys.argv[1]:
        test_random_input()
    elif "test" in sys.argv[1]:
        test(model_filepath="fitted.model")
    else:
        print("Arguments:")
        print("  --train - Train a neural network on the given OpenAI Gym environment.")
        print("  --random - Test the OpenAI Gym environment with random inputs.")
        print("  --test - Test the OpenAI Gym environment with the trained neural network.")
    