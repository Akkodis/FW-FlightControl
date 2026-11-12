# FW-FlightControl

Control algorithms, training and testing scripts for a x8 Fixed Wing UAV. Uses a JSBSim simulator based RL compatible framework: [FW-JSBSim](https://github.com/Akkodis/FW-JSBGym).

## Paper reproducibility

Start with the [reproducibility guide](./reproducibility/README.md): current
status, the four reports and the reproduction commands. These document the
validated ICINCO slices and remaining work; they do not certify reproduction of
either complete paper.

## Installation
For ICINCO checkpoint evaluations, create the dedicated Linux x86-64
environment once, from this directory:

```bash
conda env create -f environment-icinco2024.yml
conda activate fw-icinco2024
```

Keep FW-JSBGym as a sibling checkout and initialize the TD-MPC2 submodule.
Neural evaluations also require an NVIDIA GPU and a CUDA 12.4-compatible
driver. See the [runtime guide](./reproducibility/README.md#icinco-evaluation-environment)
for the exact-lock alternative and limitations.

For other project work, continue using `fw-310`; do not change its JSBSim
version. The general installation instructions are to install
[FW-JSBSim](https://github.com/Akkodis/FW-JSBGym), then run `pip install -e .`.

## Directory organization
- `agents/`: Contains the agents.
    - `tdmpc2/`: Contains the TD-MPC2 agent NN structure, logic and training/update routines. Adapted from the official [TD-MPC2](https://github.com/nicklashansen/tdmpc2) repository.
- `config/`: Hydra compatible configuration files
    - `env/jsbsim`: Environment relative config setting up the simulation: wind, gusts, turbulences, simulation frequence...
    - `env/task`: Task relative config describing Markov Decision Process configuration
    - `env/reward`: Config for different rewards functions and weights.
    - `rl/`: Config relative to the RL algorithms
- `eval/`:
    - `attitude_control/`: Hydra-configured generic evaluation harness, target arrays and legacy outputs.
    - `waypoint_tracking/`: Waypoint tracking evaluation scripts.
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
See the [attitude-control evaluation guide](./fw_flightcontrol/eval/attitude_control/README.md)
for configurable targets, checkpoints, simulator/policy seeds, atmosphere
severities, and visualization. For example, use the historical ICINCO preset
to evaluate PID:

```bash
conda activate fw-icinco2024
python -m fw_flightcontrol.eval.attitude_control experiment=icinco2024 \
    'evaluation.targets=[{path:fw_flightcontrol/eval/attitude_control/targets/easy.npy}]' \
    'evaluation.aggregation.target_groups=[]' \
    evaluation.output=reproducibility/runs/pid-easy
```

For a neural controller, explicitly configure the checkpoint so model
selection stays independent from atmosphere and target choices:

```bash
python -m fw_flightcontrol.eval.attitude_control algorithm=ppo checkpoints=single \
    "checkpoints.entries.0.path=/absolute/path/to/model.pt" \
    'evaluation.targets=[{path:fw_flightcontrol/eval/attitude_control/targets/easy.npy}]' \
    evaluation.output=reproducibility/runs/ppo-easy
```

The harness refuses a non-empty output directory and records configuration,
runtime versions, source revisions, and hashes. For an interactive
visualization, set `evaluation.visualization.mode=matplotlib` and
`evaluation.visualization.interactive=true`:

```bash
python -m fw_flightcontrol.eval.attitude_control experiment=icinco2024 \
    evaluation.visualization.mode=matplotlib \
    evaluation.visualization.interactive=true \
    evaluation.output=reproducibility/runs/pid-preview
```


## Citation
```
@inproceedings{olivares2024mfvsmb,
  title={Model-Free versus Model-Based Reinforcement Learning for Fixed-Wing UAV Attitude Control Under Varying Wind Conditions}, 
  author={David Olivares and Pierre Fournier and Pavan Vasishta and Julien Marzat},
  booktitle={International Conference on Informatics in Control, Automation and Robotics (ICINCO)},
  year={2024}
}
```