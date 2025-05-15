import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='config')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.rl.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg.rl = parse_cfg(cfg.rl)
	print(colored(f'Task: {cfg.rl.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.rl.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.rl.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.rl.multitask and ('mt80' in cfg.rl.checkpoint or 'mt30' in cfg.rl.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	os.chdir(hydra.utils.get_original_cwd())
	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg.rl)
	assert os.path.exists(cfg.rl.checkpoint), f'Checkpoint {cfg.rl.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.rl.checkpoint)
	
	# Evaluate
	if cfg.rl.multitask:
		print(colored(f'Evaluating agent on {len(cfg.rl.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.rl.task}:', 'yellow', attrs=['bold']))
	if cfg.rl.save_video:
		video_dir = os.path.join(cfg.rl.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.rl.tasks if cfg.rl.multitask else [cfg.rl.task]

	# initial roll and pitch references
	roll_limit = np.deg2rad(60)
	pitch_limit = np.deg2rad(30)

	# setting a fixed random seed for JSBSim atmo sampling
	cfg.env.jsbsim.eval_sim_options.seed = 1

	for task_idx, task in enumerate(tasks):
		if not cfg.rl.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		# for i in range(cfg.rl.eval_episodes):
		for i in range(0,1):
			# obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			obs, info = env.reset(options=cfg.env.jsbsim.eval_sim_options)
			obs, info, done, ep_reward, t = obs, info, False, 0, 0
			# roll_ref = np.random.uniform(-roll_limit, roll_limit)
			# pitch_ref = np.random.uniform(-pitch_limit, pitch_limit)
			roll_ref = np.deg2rad(30) # 25
			pitch_ref = np.deg2rad(15) # 15
			print(f"Env Done, new ref : roll = {roll_ref}, pitch = {pitch_ref} sampled")
			if cfg.rl.save_video:
				frames = [env.render()]
			while not done:
				# Set roll and pitch references
				env.set_target_state(roll_ref, pitch_ref)
				action = agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.rl.save_video:
					frames.append(env.render())
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.rl.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.rl.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.rl.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
