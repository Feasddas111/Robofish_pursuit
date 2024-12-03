import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
USE_CUDA = False
def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def save_training_config(config, curr_run):
    filepath= (Path('./models') / config.env_id / config.model_name /
                  curr_run) / 'train_config.txt'
    with open(filepath, 'a') as file:
        file.write("———" + str(curr_run) + "th train:" + '\n')
        file.write('\t' + "env = "+ str(config.env_id) + '\n')
        file.write('\t' + "model = " + str(config.model_name) + '\n')
        file.write('\t' + "load_existing_model = " + str(config.load_existing_model) + '\n')
        file.write('\t' + "load_existing_model_path = " + str(config.load_existing_model_path) + '\n')
        file.write('\t' + "episodes = " + str(config.n_episodes) + '\n')
        file.write('\t' + "episode_length = " + str(config.episode_length) + '\n')
        file.write('\t' + "steps_per_update = " + str(config.steps_per_update) + '\n')
        file.write('\t' + "learning_rate = " + str(config.lr) + '\n')
        file.write('\t' + "seed = " + str(config.seed) + '\n' + '\n')
        file.write('\t' + "action = "  + '\n'+ '\n')
        file.write('\t' + "用时： " + '\n')

def save_training_info(config,info,curr_run):
    filepath= (Path('./models') / config.env_id / config.model_name /
                  curr_run) / 'train_training.txt'
    with open(filepath, 'a') as file:
        file.write(str(info) + '\n')

def save_reward_info(config,info,curr_run):
    filepath= (Path('./models') / config.env_id / config.model_name /
                  curr_run) / 'train_reward.txt'
    with open(filepath, 'a') as file:
        file.write(str(info) + '\n')

def save_colliside_info(config,info,curr_run):
    filepath= (Path('./models') / config.env_id / config.model_name /
                  curr_run) / 'train_colliside.txt'
    with open(filepath, 'a') as file:
        file.write(str(info) + '\n')

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    if config.load_existing_model:
            model_path = (Path('./models')  / config.env_id / config.model_name / config.load_existing_model_path )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    if config.load_existing_model:
        maddpg = MADDPG.init_from_save(model_path)
    else:
        maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                      adversary_alg=config.adversary_alg,
                                      tau=config.tau,
                                      lr=config.lr,
                                      hidden_dim=config.hidden_dim)

    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    returns = [0.0] * 3
    collisides = [0.0] * 3
    reward_list = []
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        maddpg.prep_rollouts(device='cpu')
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()
        ep_infos = np.zeros(3)
        for et_i in range(config.episode_length):
            print("Step %i" % (et_i))
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            ep_infos += np.array(infos[0]['n'])
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        reward_list.append(ep_rews)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')
        returns += np.array(ep_rews) / config.training_evaluate_steps
        if (ep_i + 1) % config.training_evaluate_steps == 0:
            ep_returns = returns.tolist()
            info = f"{ep_i + 1 - config.training_evaluate_steps}-{ep_i + 1} Episod_ep_return： {ep_returns}"
            save_reward_info(config, info, curr_run)
            save_training_info(config, info, curr_run)
            returns = np.zeros(len(env.action_space))
        collisides += np.array(ep_infos) / config.training_evaluate_steps
        if (ep_i + 1) % config.training_evaluate_steps == 0:
            ep_collisides = collisides.tolist()
            info_colliside = f"{ep_i + 1 - config.training_evaluate_steps}-{ep_i + 1} Episode_ep_info： {ep_collisides}"
            save_colliside_info(config, info_colliside, curr_run)
            save_training_info(config, info_colliside, curr_run)
            collisides = np.zeros(len(env.action_space))
    reward_npy = np.array(reward_list)
    np.save(run_dir / 'incremental'/ 'reward_result.npy', reward_npy)

    maddpg.save(run_dir / 'model.pt')
    save_training_config(config, curr_run)
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",default="simple_underwater_tag", type =str, help="Name of environment")
    parser.add_argument("--model_name",default="MADDPG", type =str,
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1497504, type=int,
                        help="Random seed")     #1497504  7867839 4826283
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=30000, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--training_evaluate_steps", default=1000, type=int)
    parser.add_argument("--n_exploration_eps", default=30000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--load_existing_model", default=False, type=bool, help="Load the existing model and continue training")
    parser.add_argument("--load_existing_model_path", default="run8\model.pt", type=str)
    config = parser.parse_args()
    run(config)
