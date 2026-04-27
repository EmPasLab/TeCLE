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
PPO-RND agent on MuJoCo continuous control environments.

From the paper "Exploration by Random Network Distillation"
https://arxiv.org/abs/1810.12894
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
from networks.policy import GaussianRndActorCriticMlpNet
from networks.curiosity import RndMlpNet
from ppo_rnd import agent
from checkpoint import PyTorchCheckpoint
from schedule import LinearSchedule
import main_loop
import gym_env
import greedy_actors

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'HalfCheetah-v4',
    'MuJoCo continuous control environment name, e.g. HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4.',
)
flags.DEFINE_integer('hidden_size', 256, 'Number of units in the MLP hidden layers.')
flags.DEFINE_integer('rnd_latent_dim', 128, 'Latent dimension for RND embedding networks.')
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm when clipping.')
flags.DEFINE_float('learning_rate', 0.0003, 'Learning rate for policy network.')
flags.DEFINE_float('rnd_learning_rate', 0.0003, 'Learning rate for RND predictor network.')
flags.DEFINE_float('ext_discount', 0.99, 'Discount rate for extrinsic environment reward.')
flags.DEFINE_float('int_discount', 0.99, 'Discount rate for intrinsic reward.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for GAE general advantage estimation.')
flags.DEFINE_float('entropy_coef', 0.001, 'Coefficient for the entropy loss.')
flags.DEFINE_float('value_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_float('clip_epsilon_begin_value', 0.2, 'PPO clip epsilon begin value.')
flags.DEFINE_float('clip_epsilon_end_value', 0.1, 'PPO clip epsilon final value.')
flags.DEFINE_integer(
    'init_rnd_obs_steps',
    128,
    'Number of random steps to generate statistics for RND observation normalizer.',
)
flags.DEFINE_integer('rnd_obs_clip', 5, 'Observation normalization clip range for RND.')
flags.DEFINE_float('max_abs_obs', 10.0, 'Clip observation values to [-max_abs_obs, max_abs_obs].')
flags.DEFINE_float('max_abs_reward', 10.0, 'Clip reward values to [-max_abs_reward, max_abs_reward].')

flags.DEFINE_integer('unroll_length', 1024, 'Collect N transitions before sending to learner, per actor.')
flags.DEFINE_integer('update_k', 4, 'Run update k times when doing learning.')
flags.DEFINE_integer('num_iterations', 10, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(1e5), 'Number of training steps to run per iteration, per actor.')
flags.DEFINE_integer('num_eval_steps', int(2e4), 'Number of evaluation steps to run per iteration.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', False, 'Run actors on GPU. Default off for MuJoCo (CPU is usually fine).')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/ppo_rnd_mujoco_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')
flags.DEFINE_string('load_checkpoint', '', 'Path to checkpoint file to restore from, default empty (no restore).')


def main(argv):
    """Trains PPO-RND agent on MuJoCo continuous control tasks."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs PPO-RND-Gaussian agent on {runtime_device}')
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
    policy_network = GaussianRndActorCriticMlpNet(
        state_dim=state_dim, action_dim=action_dim, hidden_size=FLAGS.hidden_size
    )

    # Create RND target and predictor networks (operate on 1-D state vectors).
    rnd_target_network = RndMlpNet(state_dim=state_dim, is_target=True, latent_dim=FLAGS.rnd_latent_dim)
    rnd_predictor_network = RndMlpNet(state_dim=state_dim, is_target=False, latent_dim=FLAGS.rnd_latent_dim)

    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)
    rnd_optimizer = torch.optim.Adam(rnd_predictor_network.parameters(), lr=FLAGS.rnd_learning_rate)

    # Test network output.
    s = torch.from_numpy(obs[None, ...]).float()
    network_output = policy_network(s)
    pi_mu = network_output.pi_mu
    pi_sigma = network_output.pi_sigma
    ext_baseline = network_output.ext_baseline
    int_baseline = network_output.int_baseline
    assert pi_mu.shape == (1, action_dim)
    assert pi_sigma.shape == (1, action_dim)
    assert ext_baseline.shape == int_baseline.shape == (1, 1)

    clip_epsilon_scheduler = LinearSchedule(
        begin_t=0,
        end_t=int((FLAGS.num_iterations * int(FLAGS.num_train_steps * FLAGS.num_actors)) / FLAGS.unroll_length),
        begin_value=FLAGS.clip_epsilon_begin_value,
        end_value=FLAGS.clip_epsilon_end_value,
    )

    # Create queue to share transitions between actors and learner.
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 8)
    manager = multiprocessing.Manager()
    shared_params = manager.dict({'policy_network': None})

    # Create PPO-RND Gaussian learner agent instance.
    learner_agent = agent.GaussianLearner(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        rnd_optimizer=rnd_optimizer,
        rnd_obs_clip=FLAGS.rnd_obs_clip,
        clip_epsilon=clip_epsilon_scheduler,
        ext_discount=FLAGS.ext_discount,
        int_discount=FLAGS.int_discount,
        gae_lambda=FLAGS.gae_lambda,
        total_unroll_length=int(FLAGS.num_actors * FLAGS.unroll_length),
        update_k=FLAGS.update_k,
        rnd_experience_proportion=min(1.0, 32 / FLAGS.num_actors),
        entropy_coef=FLAGS.entropy_coef,
        value_coef=FLAGS.value_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        state_dim=state_dim,
        device=runtime_device,
        shared_params=shared_params,
    )

    # Generate random observations to initialise the RND observation normaliser.
    obs = eval_env.reset()
    logging.info(f'Generating {FLAGS.init_rnd_obs_steps} random observations for RND normaliser')
    random_obs = []

    for _ in range(FLAGS.init_rnd_obs_steps):
        a_t = eval_env.action_space.sample()
        s_t, _, done, _ = eval_env.step(a_t)
        random_obs.append(s_t)
        if done:
            eval_env.reset()

    learner_agent.init_rnd_obs_stats(random_obs)

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
    eval_agent = greedy_actors.GaussianRndPolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='PPO-RND-Gaussian-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name='PPO-RND-Gaussian', save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))
    checkpoint.register_pair(('rnd_target_network', rnd_target_network))
    checkpoint.register_pair(('rnd_predictor_network', rnd_predictor_network))

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
