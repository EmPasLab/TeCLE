# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO-AMA agent on MuJoCo continuous control environments.

From the paper "How to Stay Curious while Avoiding Noisy TVs"

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.
"""
from absl import app
from absl import flags
from absl import logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import torch
import copy

# pylint: disable=import-error
from networks.policy import GaussianActorMlpNet, GaussianCriticMlpNet
from networks.curiosity import GaussianAmaMlpNet
from checkpoint import PyTorchCheckpoint
from schedule import LinearSchedule
import main_loop
import gym_env
from ppo_ama import agent
import greedy_actors

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'HalfCheetah-v4',
    'MuJoCo continuous control environment name, e.g. HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, Humanoid-v4.',
)
flags.DEFINE_integer('hidden_size', 256, 'Number of units in the MLP hidden layers.')
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0003, 'Learning rate for policy network.')
flags.DEFINE_float('baseline_learning_rate', 0.0003, 'Learning rate for critic.')
flags.DEFINE_float('ama_learning_rate', 0.0003, 'Learning rate for AMA module.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('entropy_coef', 0.001, 'Coefficient for the entropy loss.')
flags.DEFINE_float('clip_epsilon_begin_value', 0.2, 'PPO clip epsilon begin value.')
flags.DEFINE_float('clip_epsilon_end_value', 0.1, 'PPO clip epsilon final value.')
flags.DEFINE_float('max_abs_obs', 10.0, 'Clip observation values to [-max_abs_obs, max_abs_obs].')
flags.DEFINE_float('max_abs_reward', 10.0, 'Clip reward values to [-max_abs_reward, max_abs_reward].')

flags.DEFINE_float(
    'intrinsic_lambda',
    0.1,
    'Scaling factor for AMA intrinsic reward: lambda * max(0, mse - variance).',
)
flags.DEFINE_float(
    'ama_beta',
    0.2,
    'Weights inverse model loss against the forward NLL loss in AMA module.',
)
flags.DEFINE_float(
    'policy_loss_coef',
    1.0,
    'Weights policy loss against the AMA module loss.',
)

flags.DEFINE_integer('unroll_length', 1024, 'Collect N transitions (cross episodes) before send to learner, per actor.')
flags.DEFINE_integer('update_k', 4, 'Run update k times when do learning.')
flags.DEFINE_integer('num_iterations', 10, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(1e5), 'Number of training env steps to run per iteration, per actor.')
flags.DEFINE_integer('num_eval_steps', int(2e4), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', False, 'Run actors on GPU. Default off for MuJoCo (CPU is usually fine).')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/ppo_ama_mujoco_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')
flags.DEFINE_string('load_checkpoint', '', 'Path to checkpoint file to restore from, default empty (no restore).')


def main(argv):
    """Trains PPO-AMA agent on MuJoCo continuous control tasks."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs PPO-AMA agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create environment builder.
    def environment_builder():
        return gym_env.create_mujoco_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**10),
            max_abs_obs=FLAGS.max_abs_obs,
            max_abs_reward=FLAGS.max_abs_reward,
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Test environment and state shape.
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (state_dim,)

    # Create policy network.
    policy_network = GaussianActorMlpNet(state_dim=state_dim, action_dim=action_dim, hidden_size=FLAGS.hidden_size)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    critic_network = GaussianCriticMlpNet(state_dim=state_dim, hidden_size=FLAGS.hidden_size)
    critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=FLAGS.baseline_learning_rate)

    # Create AMA module.
    ama_network = GaussianAmaMlpNet(state_dim=state_dim, action_dim=action_dim)
    ama_optimizer = torch.optim.Adam(ama_network.parameters(), lr=FLAGS.ama_learning_rate)

    # Test network output.
    s = torch.from_numpy(obs[None, ...]).float()
    pi_mu, pi_sigma = policy_network(s)
    assert pi_mu.shape == (1, action_dim)
    assert pi_sigma.shape == (1, action_dim)

    clip_epsilon_scheduler = LinearSchedule(
        begin_t=0,
        end_t=int(
            (FLAGS.num_iterations * int(FLAGS.num_train_steps * FLAGS.num_actors)) / FLAGS.unroll_length
        ),
        begin_value=FLAGS.clip_epsilon_begin_value,
        end_value=FLAGS.clip_epsilon_end_value,
    )

    # Create queue to share transitions between actors and learner.
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 8)
    manager = multiprocessing.Manager()
    shared_params = manager.dict({'policy_network': None})

    # Create PPO-AMA Gaussian learner agent instance.
    learner_agent = agent.GaussianLearner(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        critic_network=critic_network,
        critic_optimizer=critic_optimizer,
        ama_network=ama_network,
        ama_optimizer=ama_optimizer,
        clip_epsilon=clip_epsilon_scheduler,
        discount=FLAGS.discount,
        gae_lambda=FLAGS.gae_lambda,
        total_unroll_length=int(FLAGS.unroll_length * FLAGS.num_actors),
        update_k=FLAGS.update_k,
        intrinsic_lambda=FLAGS.intrinsic_lambda,
        ama_beta=FLAGS.ama_beta,
        policy_loss_coef=FLAGS.policy_loss_coef,
        entropy_coef=FLAGS.entropy_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
        shared_params=shared_params,
    )

    # Create actor environments and actor instances.
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    actor_devices = ['cpu'] * FLAGS.num_actors
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    actors = [
        agent.GaussianActor(
            rank=i,
            data_queue=data_queue,
            policy_network=copy.deepcopy(policy_network),
            unroll_length=FLAGS.unroll_length,
            device=actor_devices[i],
            shared_params=shared_params,
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance.
    eval_agent = greedy_actors.GaussianPolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='PPO-AMA-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name='PPO-AMA', save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))
    checkpoint.register_pair(('critic_network', critic_network))
    checkpoint.register_pair(('ama_network', ama_network))

    if FLAGS.load_checkpoint:
        checkpoint.restore(FLAGS.load_checkpoint)
        logging.info(f'Restored checkpoint from "{FLAGS.load_checkpoint}"')

    # Run parallel training N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=data_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    app.run(main)
