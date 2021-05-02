# Portfolio OpenAIGym Documentation
Gym is a toolkit for developing and comparing reinforcement learning algorithms. This repo contains my notes and learning efforts.

[Official Docs for OpenAIGym](https://gym.openai.com/docs/)

# Table of Contents
* [Quickstart](#quickstart)
* [Summary](#summary)
* [Directory Structure](#directory-structure)
* [Register a Custom Environment](#register-a-custom-environment)
* [Virtual Env](#virtual-env)
  * [Create Virtual Environment](#create-virtual-environment)
  * [Load the Virtual Environment](#load-the-virtual-environment)
  * [Return to Normal Environment](#return-to-normal-environment)
  * [Installing Modules in Virtual Environment](#installing-modules-in-virtual-environment)

# Quickstart
Use this example to do a test run of Open AI Gym, to make sure it's installed properly.

```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```

# Summary
It is recommended that you install the gym and any dependencies in a [virtualenv](https://pythonbasics.org/virtualenv/).

# Directory Structure
```
gym_dir
  - __init__.py # Register environments here
  - envs
    - __init__.py # Import the environment and any custom libraries here
    - custom_env.py # Define the environment here
  


```

# Register a Custom Environment
```
gym_dir
  - envs
  - __init__.py # Register environments here
      

# To register:
from gym.envs.registration import register

register(
    id='<your_env_id>',
    entry_point='gym_dir.envs:CustomEnv',
    max_episode_steps=<your_max_steps>,
)
```

# Register an Environment

# Virtual Env
Source: [virtualenv](https://pythonbasics.org/virtualenv/).

Every project should have it's own `virtualenv`.

## Create Virtual Environment
This creates the folder `.venv` with these sub directories (Linux): `bin`, `include`, `lib`, and `share`.

Or, (Windows): 
```
py -3 -m venv .venv
```

## Load the Virtual Environment
```
cd .venv
bin/activate
```

## Return to Normal Environment
```
deactivate
```

## Installing Modules in Virtual Environment
This `activate`'s the virtual environment and then installs `gym` (Linux).
```
source .venv/bin/activate
pip install -U gym[all]
```

# Open AI Gym Basics
Activate the `.venv`.

While in the environment, 