import numpy as np
import torch
import gym
import argparse
from tqdm import tqdm
import os
import d4rl
import utils
import TD3_BC


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, step, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	ep_rewards = []
	for _ in range(eval_episodes):
		total_reward = 0
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			total_reward += reward
		ep_rewards.append(total_reward)

	avg_reward = sum(ep_rewards) / len(ep_rewards)
	std_reward = np.sqrt(sum([(reward - mean) ** 2 for reward in ep_rewards]) / len(ep_rewards))
	d4rl_score = eval_env.get_normalized_score(avg_reward)
	d4rl_std = eval_env.get_normalized_score(std_reward)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=int(5e3), type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=int(1e6), type=int)   # Max time steps to run environment
	parser.add_argument("--save", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load", action="store_true")        # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--no_tqdm", action="store_true")		
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--output_dir", type=str, default='/output')
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"td3_bc_vib_seed{args.seed}_{args.env}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)
	make_empty_dict = lambda: {'critic_loss': [], 'reconst_loss': [], 'kl_loss': [], 'latent': [], 'actor_loss': []}
	metrics, epoch_metrics = make_empty_dict(), make_empty_dict()
	evaluations = []
	T = 0
	if args.load and os.path.isfile(os.path.join(args.output_dir, "step")):
		T = policy.load(args.output_dir)
		evaluations = np.load(os.path.join(args.output_dir, "reward.npy")).tolist()
		for key, lst in metrics.items():
			lst.extend(np.load(os.path.join(args.output_dir, key + ".npy"), allow_pickle=True).tolist())
 
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	pbar = tqdm(total=args.eval_freq, disable=args.no_tqdm)
	for t in range(T, int(args.max_timesteps)):
		batch_metrics = policy.train(replay_buffer, args.batch_size)
		pbar.update(1)
		for metric, value in batch_metrics.items():
			epoch_metrics[metric].append(value)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			pbar.close()
			print(f"Time steps: {t+1}")
			evaluations.append(eval_policy(policy, args.env, args.seed, t, mean, std))
			for key, value in epoch_metrics.items(): 
				value = [v.cpu().detach().numpy() for v in value]
				if value[0].size == 1:
					print(f"{key}: {sum(value)}")
					metrics[key].append(sum(value))
				else:
					metrics[key].append(np.concatenate(value, axis=0))
			for key, value in metrics.items():
				np.save(os.path.join(args.output_dir, key), value)
			np.save(os.path.join(args.output_dir, "reward"), evaluations)
			if args.save: policy.save(args.output_dir, t)
			epoch_metrics = make_empty_dict()
			pbar = tqdm(total=args.eval_freq, disable=args.no_tqdm)
