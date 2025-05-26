# FW-FlightControl

Control algorithms, training and testing scripts for a x8 Fixed Wing UAV. Uses a JSBSim simulator based RL compatible framework: [FW-JSBSim](https://github.com/Akkodis/FW-JSBGym).

## Installation
Requires Python 3.9

- Install [FW-JSBSim](https://github.com/Akkodis/FW-JSBGym)
- Install `fw_flightcontrol` as a pip package with: `pip install -e .`

## Directory organization
- `agents/`: Contains the agents.
    - `tdmpc2/`: Contains the TD-MPC2 agent NN structure, logic and training/update routines. Adapted from the official [TD-MPC2](https://github.com/nicklashansen/tdmpc2) repository.
- `config/`: Hydra compatible configuration files
    - `env/jsbsim`: Environment relative config setting up the simulation: wind, gusts, turbulences, simulation frequence...
    - `env/task`: Task relative config describing Markov Decision Process configuration
    - `env/reward`: Config for different rewards functions and weights.
    - `rl/`: Config relative to the RL algorithms
- `eval/`:
    - Evaluation scripts: `*_eval_simple.py`.
    - `outputs/`: Evaluation results .csv.
- `models/`: NN model saves.
    - `icinco/`: NN models used in the ICINCO 2024 paper.
- `train/`: Training scripts for each control algorithm: PPO, SAC and TD-MPC2 (there's also a legacy TD3 script, not working)
- `utils/`: Utilitary
    - `eval_utils.py`: Creates reference sequence for evaluating agents and saves it to a .npy file.
    - `gym_utils.py`: Modifies of the `NormalizeObservation` gym wrapper.
    - `train_utils.py`: Contains common methods for training agents, like periodic evaluation, make environment and save the models.

## Usage
### Training
For example, to train a PPO agent on the `ACBohnNoVaIErr-v0` env, with only gusts for 750 000 timesteps:
```
python train/ppo_train.py rl.PPO.env_id=ACBohnNoVaIErr-v0 rl.PPO.exp_name=gustsonly env/jsbsim=gustsonly
```

### Evaluating
To evaluate, a PPO agent:
```
python eval/ppo_eval_simple.py rl.PPO.env_id=ACBohnNoVaIErr-v0 env/jsbsim=gustsonly model_path=models/icinco/ppo/gustonly/ppo_bohn_caps_nohist_easymedrefs_gustsonly_1_06-06_14:42:46.pt res_file=ppo_gustsonly_1 ref_file=simple_easy
```
If you wish to have a plot view: add `env.jsbsim.render_mode=plot` and if you wish to have a FlightGear view: add `env.jsbsim.render_mode=fgear`.


## Citation
```
@inproceedings{olivares2024mfvsmb,
  title={Model-Free versus Model-Based Reinforcement Learning for Fixed-Wing UAV Attitude Control Under Varying Wind Conditions}, 
  author={David Olivares and Pierre Fournier and Pavan Vasishta and Julien Marzat},
  booktitle={International Conference on Informatics in Control, Automation and Robotics (ICINCO)},
  year={2024}
}
```