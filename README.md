# Model Predictive Control of an Autonomous Vehicle

We utilize model predictive controller to perform lane following and obstacle avoidance.

<p align="center">
  <img width="450" height="300" src="https://github.com/coldhenry/Model-Predictive-Control-of-Autonomous-Car/blob/main/Multi-Purpose-MPC-master/Test/simple-bicycle-example/mpc.gif"/><br/>
  <em>Agent running trying to avoid obstacles.</em>
</p>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Simulator

[highway-env](https://github.com/eleurent/highway-env)

### Built With

* Python 3.6.10

* PyTorch >= 1.7.0

* gym 0.17.3

* numpy >= 1.16.2

* matplotlib >= 3.1.1

## Code Organization

```
.
├── Highway                       # Scripts for highway environment
│   ├── dueling_dqn.py            # Dueling DQN
│   ├── double_dqn.py             # Double DQN
│   ├── double_dqn_cnn.py         # Double DQN with CNN architecture
│   └── double_dqn_prioritized.py # Double DQN with Prioritized Buffer
├── Intersection                  # Scripts for intersection environment
├── doc                           # Detailed info
└── README.md
```

## How to Run

There are 4 methods you can try, namely *dueling DQN*, *double DQN*, *double DQN with cnn*, and *double DQN with prioritized replay buffer*, with corresponding file name.

ex. if you want to try double DQN in highway environment, just do
```
cd Highway
python double_dqn.py
```

### Configure the Environment

We configured the highway environment in the following way, you can also read the [documentation online](https://highway-env.readthedocs.io/en/latest/quickstart.html#configuring-an-environment) for other settings.

```python
env = gym.make("highway-v0")
env.config["lanes_count"] = 4
env.config["duration"] = 100
env.config["vehicles_count"] = 10
env.config["vehicles_density"] = 1.3
env.config["policy_frequency"] = 2
env.config["simulation_frequency"] = 10
env.reset()
```

## Results

<p align="center">
  <img width="640" height="160" src="https://github.com/arthur960304/dqn-dense-traffic/blob/main/doc/highway.gif"/><br/>
</p>
<p align="center">
  <em>Agent running in the highway environment.</em>
</p>


## Authors

* **Arthur Hsieh** - <i>Initial Works</i> - [arthur960304](https://github.com/arthur960304)
* **Henry Liu** - <i>Initial Works</i> - [coldhenry](https://github.com/coldhenry)
