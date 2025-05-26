import os
import sys
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../agents/tdmpc2/tdmpc2/')

from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.parser import parse_cfg
from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.buffer import Buffer
from fw_flightcontrol.agents.tdmpc2.tdmpc2.envs import make_env
from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2
from fw_flightcontrol.agents.tdmpc2.tdmpc2.trainer.offline_trainer import OfflineTrainer
from fw_flightcontrol.agents.tdmpc2.tdmpc2.trainer.online_trainer import OnlineTrainer
from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='tdmpc2_default', config_path='../config')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.rl.steps > 0, 'Must train for at least 1 step.'
	cfg.rl = parse_cfg(cfg.rl)
	os.chdir(hydra.utils.get_original_cwd())

	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.rl.work_dir)
	trainer_cls = OfflineTrainer if cfg.rl.multitask else OnlineTrainer
	env = make_env(cfg)
	cfg_rl = update_cfg(cfg.rl)
	trainer = trainer_cls(
		cfg=cfg_rl,
		cfg_all=cfg,
		env=env,
		agent=TDMPC2(cfg_rl),
		buffer=Buffer(cfg_rl),
		logger=Logger(cfg, cfg_rl),
	)
	trainer.train()
	print('\nTraining completed successfully')

def update_cfg(cfg_rl):
	# if we don't use the encoder, therefore no latent space, so latent_dim = obs dimension
	if not cfg_rl.use_enc:
		cfg_rl.latent_dim = cfg_rl.obs_shape['state'][0]
		cfg_rl.simnorm_dim = 7 # adjust this based on the observation space
	if cfg_rl.latent_dim == 14:
		cfg_rl.simnorm_dim = 7
	# if not using CAPS loss, set ts_coef to 0
	if not cfg_rl.use_caps:
		cfg_rl.ts_coef = 0
	# return cfg_to_dataclass(cfg_rl)
	return cfg_rl

if __name__ == '__main__':
	train()
