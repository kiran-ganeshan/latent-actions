import copy, os
from torch.nn.modules.module import T
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from numbers import Number


class Encoder(nn.Module):
    
	def __init__(self, action_dim, latent_dim, device):
		super(Encoder, self).__init__()
		self.e1 = nn.Linear(action_dim, 400)
		self.e2 = nn.Linear(400, 400)

		self.mean = nn.Linear(400, latent_dim)
		self.log_std = nn.Linear(400, latent_dim)
		self.device = device
  
	def forward(self, action):
		z = F.relu(self.e1(action))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		return z, mean, std
  

class DecoderCritic(nn.Module):
    
	def __init__(self, state_dim, latent_dim, device):
		super(DecoderCritic, self).__init__()
		self.l1 = nn.Linear(state_dim + latent_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + latent_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)

		self.latent_dim = latent_dim
		self.device = device

	def forward(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
   
		q1 = F.relu(self.l1(torch.cat([state, z], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, z], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def q1(self, state, z=None):
     	# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
   
		q1 = F.relu(self.l1(torch.cat([state, z], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
    
    
class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, num_samples, device, discount=0.99, tau=0.005, lmbda=0.75, beta=0.5, temp=1):
		latent_dim = 2

		self.encoder = Encoder(action_dim, latent_dim, device).to(device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters())

		self.critic = DecoderCritic(state_dim, latent_dim, device).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.max_action = max_action
		self.N = num_samples
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.beta = beta
		self.temp = temp
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.vae.decode(state)
			q1 = self.critic.q1(state, self.encoder(action)[0])
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations, step, batch_size=100, no_tqdm=False):
		metrics = {'critic_loss': list(), 'critic_kl_loss': list(), 
                   'bellman_loss': list(), 'vae_loss': list(), 'vae_kl_loss': list(),
                   'reconst_loss': list(), 'latent': list()}
		for it in tqdm(range(iterations), disable=no_tqdm):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			VKL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * VKL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state)

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			latent, mean, std = self.encoder(action)
			current_Q1, current_Q2 = self.critic(state, latent)
			KL_loss	= - 0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			bellman_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			critic_loss = bellman_loss + self.beta * KL_loss

			self.critic_optimizer.zero_grad()
			self.encoder_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			self.encoder_optimizer.step()

			metrics['critic_loss'].append(critic_loss)
			metrics['bellman_loss'].append(bellman_loss)
			metrics['critic_kl_loss'].append(KL_loss)
			metrics['latent'].append(latent)
			metrics['vae_loss'].append(vae_loss)
			metrics['vae_kl_loss'].append(VKL_loss)
			metrics['reconst_loss'].append(recon_loss)

			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for key, value in metrics.items(): 
			value = [v.cpu().detach().numpy() for v in value]
			if value[0].size == 1:
				print(f"{key}: {sum(value)}")
				metrics[key] = [sum(value)]
			else:
				metrics[key] = [np.concatenate(value, axis=0)]
		#wandb.log({'critic_loss': total_critic_loss, 
		#		   'actor_loss': total_actor_loss, 
		#		   'vae_loss': total_vae_loss}, step=int(step))
		return metrics

	def save(self, filename, step):
		torch.save(self.critic.state_dict(), os.path.join(filename, "critic"))
		torch.save(self.critic_optimizer.state_dict(), os.path.join(filename, "critic_optimizer"))
		
		torch.save(self.vae.state_dict(), os.path.join(filename, "vae"))
		torch.save(self.vae_optimizer.state_dict(), os.path.join(filename, "vae_optimizer"))

		torch.save(self.encoder.state_dict(), os.path.join(filename, "encoder"))
		torch.save(self.encoder_optimizer.state_dict(), os.path.join(filename, "encoder_optimizer"))

		torch.save(step, os.path.join(filename, "step"))


	def load(self, filename):
		self.critic.load_state_dict(torch.load(os.path.join(filename, "critic")))
		self.critic_optimizer.load_state_dict(torch.load(os.path.join(filename, "critic_optimizer")))
		self.critic_target = copy.deepcopy(self.critic)

		self.vae.load_state_dict(torch.load(os.path.join(filename, "vae")))
		self.vae_optimizer.load_state_dict(torch.load(os.path.join(filename, "vae_optimizer")))

		self.encoder.load_state_dict(torch.load(os.path.join(filename, "encoder")))
		self.encoder_optimizer.load_state_dict(torch.load(os.path.join(filename, "encoder_optimizer")))

		step = torch.load(os.path.join(filename, "step"))
		return step