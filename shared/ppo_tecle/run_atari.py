import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

from absl import app
from absl import flags
from absl import logging
import multiprocessing
import numpy as np
import torch
import copy


from networks.policy import RndActorCriticConvNet
from networks.curiosity import TecleCVAEConvNet as cVAE, TecleInverseConvNet as InverseNet
from ppo_tecle import agent
from checkpoint import PyTorchCheckpoint
from schedule import LinearSchedule
import main_loop
import gym_env
import greedy_actors

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name', 'BankHeist', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.'
) 
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_integer('num_actors', 32, 'Number of worker processes to use.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 10.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('TeCLE_learning_rate', 0.01, 'Learning rate for CVAE.')
flags.DEFINE_float('inv_learning_rate', 0.01, 'Learning rate for inverse network.')
flags.DEFINE_float('ext_discount', 0.999, 'Discount rate for extrinsic environment reward.')
flags.DEFINE_float('int_discount', 0.99, 'Discount rate intrinsic reward.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('entropy_coef', 0.001, 'Coefficient for the entropy loss.')
flags.DEFINE_float('value_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_float('clip_epsilon_begin_value', 0.1, 'PPO clip epsilon begin value.')
flags.DEFINE_float('clip_epsilon_end_value', 0.1, 'PPO clip epsilon final value.')
flags.DEFINE_float('beta', 0.0, 'TeCLE noise beta')
flags.DEFINE_integer(
    'init_obs_steps',
    100,
    'Warm up random steps to take in order to generate statistics for observation.',
)
flags.DEFINE_integer('obs_clip', 10, 'Observation normalization clip range.')

flags.DEFINE_integer('unroll_length', 128, 'Collect N transitions (cross episodes) before send to learner, per actor.')
flags.DEFINE_integer('update_k', 4, 'Run update k times when do learning.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(5e5), 'Number of training steps (environment steps or frames) to run per iteration, per actor.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 18000, 'Maximum steps (before frame skip) per episode, which is 4500 x 4.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', False, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/ppo_TeCLE_BankHeist_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')
flags.DEFINE_string('algo', 'TeCLE', 'Path for checkpoint directory.')
flags.DEFINE_bool('sticky', False, 'Path for checkpoint directory.')


def main(argv):
    
    print("noise beta : ", FLAGS.beta)
    print("log_directory : ", FLAGS.results_csv_path)
    FLAGS.algo = FLAGS.algo + '_' + 'st=' + str(FLAGS.sticky) + '_' + str(FLAGS.beta) + '_' + FLAGS.environment_name
    print("algorithm : ", FLAGS.algo)
    print("sticky_action : ", FLAGS.sticky)
    
    """Trains PPO-TeCLE agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs {FLAGS.algo} agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)

    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            seed=random_state.randint(1, 2**10),
            terminal_on_life_loss=False,
            sticky_action = FLAGS.sticky,
            noop_max=30,
            max_episode_steps=FLAGS.max_episode_steps,
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n
    TeCLE_state_dim = (1,) + state_dim[1:]  
    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    policy_network = RndActorCriticConvNet(state_dim=state_dim, action_dim=action_dim)
    inv_network = InverseNet(TeCLE_state_dim, action_dim)
    TeCLE_network = cVAE(TeCLE_state_dim, action_dim, noise_beta=FLAGS.beta)

    policy_optimizer = torch.optim.Adam(
        policy_network.parameters(),
        lr=FLAGS.learning_rate,
    )

    TeCLE_optimizer = torch.optim.Adam(
        TeCLE_network.parameters(),
        lr=FLAGS.TeCLE_learning_rate,
    )
    inv_optimizer = torch.optim.Adam(
        inv_network.parameters(),
        lr=FLAGS.inv_learning_rate,
    )

    s = torch.from_numpy(obs[None, ...]).float()
    network_output = policy_network(s)
    pi_logits = network_output.pi_logits
    ext_baseline = network_output.ext_baseline
    int_baseline = network_output.int_baseline
    assert pi_logits.shape == (1, action_dim)
    assert ext_baseline.shape == int_baseline.shape == (1, 1)

    clip_epsilon_scheduler = LinearSchedule(
        begin_t=0,
        end_t=int(
            (FLAGS.num_iterations * int(FLAGS.num_train_steps * FLAGS.num_actors)) / FLAGS.unroll_length
        ),
        begin_value=FLAGS.clip_epsilon_begin_value,
        end_value=FLAGS.clip_epsilon_end_value,
    )

    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    manager = multiprocessing.Manager()

    shared_params = manager.dict({'policy_network': None})

    learner_agent = agent.Learner(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        TeCLE_network = TeCLE_network,
        inv_network = inv_network,
        TeCLE_optimizer = TeCLE_optimizer,
        inv_optimizer = inv_optimizer,
        obs_clip=FLAGS.obs_clip,
        clip_epsilon=clip_epsilon_scheduler,
        ext_discount=FLAGS.ext_discount,
        int_discount=FLAGS.int_discount,
        gae_lambda=FLAGS.gae_lambda,
        total_unroll_length=int(FLAGS.num_actors * FLAGS.unroll_length),
        update_k=FLAGS.update_k,
        entropy_coef=FLAGS.entropy_coef,
        value_coef=FLAGS.value_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
        shared_params=shared_params,
        algo=FLAGS.algo
    )

    obs = eval_env.reset()
    logging.info(f'Generating {FLAGS.init_obs_steps} random observations for normalizer')
    random_obs = []

    for _ in range(FLAGS.init_obs_steps):
        a_t = eval_env.action_space.sample()
        s_t, _, done, _ = eval_env.step(a_t)

        random_obs.append(s_t[-1:, :, :]) 
        
        if done:
            eval_env.reset()

    learner_agent.init_obs_stats(random_obs)

    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    actor_devices = ['cpu'] * FLAGS.num_actors

    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            policy_network=copy.deepcopy(policy_network),
            unroll_length=FLAGS.unroll_length,
            device=actor_devices[i],
            shared_params=shared_params,
            algo=FLAGS.algo
        )
        for i in range(FLAGS.num_actors)
    ]

    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name=FLAGS.algo + '-greedy',
    )

    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name=FLAGS.algo, save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))
    checkpoint.register_pair(('TeCLE_network', TeCLE_network))
    checkpoint.register_pair(('inv_network', inv_network))
    
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
