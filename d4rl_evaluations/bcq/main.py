import argparse
import gym
import numpy as np
import os
import torch
import d4rl
import uuid
import json
import continuous_bcq.BCQ
import continuous_bcq.utils as utils

# Trains BCQ offline
def train_BCQ(env, state_dim, action_dim, max_action, device, output_dir, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"

    # Initialize policy
    policy = continuous_bcq.BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
    
        
    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    print('Loading buffer!')
    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        # Don't apply terminals on the last step of an episode
        if episode_step == env._max_episode_steps - 1:
            episode_step = 0
            continue  
        if done_bool:
            episode_step = 0
        replay_buffer.add(obs, action, new_obs, reward, done_bool)
        episode_step += 1
    print('Loaded buffer')
    #replay_buffer.load(f"./buffers/{buffer_name}")
    
    evaluations = []
    episode_num = 0 
    done = True 
    training_iters = 0
    
    metrics = {'critic_loss': list(), 'vae_kl_loss': list(),
               'vae_loss': list(), 'reconst_loss': list()}
    
    while training_iters < args.max_timesteps: 
            print('Train step:', training_iters, flush=True)
            batch_metrics = policy.train(replay_buffer, 
                                         iterations=int(args.eval_freq), 
                                         step=training_iters, 
                                         batch_size=args.batch_size,
                                         no_tqdm=args.no_tqdm)
            for key, value in batch_metrics.items():
                metrics[key].extend(value)
                np.save(os.path.join(output_dir, key), metrics[key])

            evaluations.append(eval_policy(policy, args.env, args.seed, training_iters))
            np.save(os.path.join(output_dir, f"reward"), evaluations)

            training_iters += args.eval_freq
            print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, step, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    ep_rewards = []
    for _ in range(eval_episodes):
            ep_rewards.append(0.)
            state, done = eval_env.reset(), False
            while not done:
                    action = policy.select_action(np.array(state))
                    state, reward, done, _ = eval_env.step(action)
                    ep_rewards[-1] += reward
    mean = sum(ep_rewards) / len(ep_rewards)
    std = np.sqrt(sum([(reward - mean) ** 2 for reward in ep_rewards]) / len(ep_rewards))
    d4rl_score = eval_env.get_normalized_score(mean)
    d4rl_std = eval_env.get_normalized_score(std)
    #wandb.log({'mean_return': d4rl_score, 'std_return': d4rl_std}, step=int(step))
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-random-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                          # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")                      # Prepends name to filename
    parser.add_argument("--eval_freq", default=5e3, type=float)                 # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)               # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3, type=int)            # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3, type=float)             # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)              # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)                  # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                             # Discount factor
    parser.add_argument("--tau", default=0.005)                                 # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                                # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0)                                     # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--beta", default=0.5)                                  # Weighting factor for KL Divergence in loss
    parser.add_argument("--temp", default=1)                                    # Temperature for AWAC-style actor
    parser.add_argument("--num_samples", "-N", default=10)                      # Number of samples to take for target
    parser.add_argument("--output_dir", default="/output")
    parser.add_argument("--no_tqdm", action="store_true")
    args = parser.parse_args()
    d4rl.set_dataset_path('./datasets')
    print("---------------------------------------")	
    print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'params.json'), 'w') as params_file:
        json.dump({
        'env_name': args.env,
        'seed': args.seed,
        }, params_file)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    print("start gym env creation", flush=True)
    env = gym.make(args.env)
    print("end gym env creation", flush=True)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_BCQ(env, state_dim, action_dim, max_action, device, results_dir, args)
