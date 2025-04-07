from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from common import layers, math, init


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.

		if self.cfg.use_enc:
			self._encoder = layers.enc(cfg)
		else: # no encoder, just identity in a module dict with key 'state'
			self._encoder = nn.ModuleDict({'state': nn.Identity()})

		if self.cfg.use_decoder:
			self._decoder = layers.mlp(cfg.latent_dim , max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim],
										cfg.obs_shape['state'][0] + cfg.task_dim, act=None)

		if self.cfg.gaussian_dyn:
			# mlps with 2x output size for mean and log_std
			self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.latent_dim, act=layers.SimNorm(cfg))
		else:
			self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))

		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._terminated = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1)
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		if self.cfg.multitask:
			self._action_masks = self._action_masks.to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		for p in self._Qs.parameters():
			p.requires_grad_(mode)
		if self.cfg.multitask:
			for p in self._task_emb.parameters():
				p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)
	
	def sample_gaussian_nn(self, nn_out):
		"""	
		Takes the output of a neural network predicting the mean and log_std of a Gaussian distribution,
		applies scaling and squashing functions and returns a rsample of the distribution.
		!! Doesn't support multitask !!
		"""
		mu, log_std = nn_out.chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		normal_dist = torch.distributions.Normal(mu, log_std.exp())
		sample = normal_dist.rsample() # reparametrization trick
		logprob = normal_dist.log_prob(sample).sum(-1, keepdim=True)
		mu, sample, logprob = math.squash(mu, sample, logprob) # TODO: Remove ?
		return sample, logprob, mu


	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)


	def decode(self, z, task):
		"""
		Decodes a latent representation into an observation.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return self._decoder(z)


	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		dyna_out = self._dynamics(z)
		# Sample from Gaussian distribution if stochastic dynamics model
		if self.cfg.gaussian_dyn:
			z_next, _, __ = self.sample_gaussian_nn(dyna_out)
		else: # else just take the direct output of the deterministic dynamics model
			z_next = dyna_out
		return z_next
	
	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	

	def terminated(self, z, task):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return torch.sigmoid(self._terminated(z))


	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		if self.cfg.multitask: # Mask out unused action dimensions
			mu = mu * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std

	def Q(self, z, a, task, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)
			
		z = torch.cat([z, a], dim=-1)
		out = (self._target_Qs if target else self._Qs)(z)

		if return_type == 'all':
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
