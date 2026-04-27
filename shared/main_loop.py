from typing import Iterable, List, Tuple, Text, Mapping, Any
import itertools
import collections
import sys
import time
import signal
import queue
import math
import multiprocessing
import threading
from absl import logging
import gymnasium as gym


import trackers as trackers_lib
import type as types_lib
from log import CsvWriter
from checkpoint import PyTorchCheckpoint
import gym_env


def run_env_loop(
    agent: types_lib.Agent, env: gym.Env
) -> Iterable[Tuple[gym.Env, types_lib.TimeStep, types_lib.Agent, types_lib.Action]]:
    if not isinstance(agent, types_lib.Agent):
        raise RuntimeError('Expect agent to be an instance of types_lib.Agent.')

    while True:
        agent.reset()
        observation = env.reset() 
        reward = 0.0
        done = loss_life = False
        first_step = True
        info = {}

        while True:
            timestep_t = types_lib.TimeStep(
                observation=observation,
                reward=reward,
                done=done or loss_life,
                first=first_step,
                info=info,
            )
            a_t = agent.step(timestep_t)
            yield env, timestep_t, agent, a_t

            a_tm1 = a_t
            observation, reward, done, info = env.step(a_tm1)
            first_step = False

            loss_life = False
            if 'loss_life' in info and info['loss_life']:
                loss_life = info['loss_life']

            if done:

                timestep_t = types_lib.TimeStep(
                    observation=observation,
                    reward=reward,
                    done=True,
                    first=False,
                    info=info,
                )
                unused_a = agent.step(timestep_t) 
                yield env, timestep_t, agent, None
                break


def run_env_steps(num_steps: int, agent: types_lib.Agent, env: gym.Env, trackers: Iterable[Any]) -> Mapping[Text, float]:
    seq = run_env_loop(agent, env)
    seq_truncated = itertools.islice(seq, num_steps)
    stats = trackers_lib.generate_statistics(trackers, seq_truncated)
    return stats


def run_single_thread_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    train_agent: types_lib.Agent,
    train_env: gym.Env,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    checkpoint: PyTorchCheckpoint,
    csv_file: str,
    use_tensorboard: bool,
    tag: str = None,
    debug_screenshots_interval: int = 0,
) -> None:
    writer = CsvWriter(csv_file)

    train_tb_log_prefix = (
        get_tb_log_prefix(train_env.spec.id, train_agent.agent_name, tag, 'train') if use_tensorboard else None
    )
    train_trackers = trackers_lib.make_default_trackers(train_tb_log_prefix, debug_screenshots_interval)

    should_run_evaluator = False
    eval_trackers = None
    if num_eval_steps > 0 and eval_agent is not None and eval_env is not None:
        should_run_evaluator = True
        eval_tb_log_prefix = (
            get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, tag, 'eval') if use_tensorboard else None
        )
        eval_trackers = trackers_lib.make_default_trackers(eval_tb_log_prefix, debug_screenshots_interval)

    for iteration in range(1, num_iterations + 1):
        logging.info(f'Training iteration {iteration}')

        train_stats = run_env_steps(num_train_steps, train_agent, train_env, train_trackers)

        checkpoint.set_iteration(iteration)
        saved_ckpt = checkpoint.save()

        if saved_ckpt:
            logging.info(f'New checkpoint created at "{saved_ckpt}"')

        log_output = [
            ('iteration', iteration, '%3d'),
            ('train_step', iteration * num_train_steps, '%5d'),
            ('train_episode_return', train_stats['mean_episode_return'], '%2.2f'),
            ('train_num_episodes', train_stats['num_episodes'], '%3d'),
            ('train_step_rate', train_stats['step_rate'], '%4.0f'),
            ('train_duration', train_stats['duration'], '%.2f'),
        ]

        if should_run_evaluator is True:
            logging.info(f'Evaluation iteration {iteration}')

            eval_stats = run_env_steps(num_eval_steps, eval_agent, eval_env, eval_trackers)

            eval_output = [
                ('eval_step', iteration * num_eval_steps, '%5d'),
                ('eval_episode_return', eval_stats['mean_episode_return'], '% 2.2f'),
                ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
                ('eval_step_rate', eval_stats['step_rate'], '%4.0f'),
                ('eval_duration', eval_stats['duration'], '%.2f'),
            ]
            log_output.extend(eval_output)

        log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)
        writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    writer.close()


def run_parallel_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    learner_agent: types_lib.Learner,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    actors: List[types_lib.Agent],
    actor_envs: List[gym.Env],
    data_queue: multiprocessing.Queue,
    checkpoint: PyTorchCheckpoint,
    csv_file: str,
    use_tensorboard: bool,
    tag: str = None,
    debug_screenshots_interval: int = 0,
) -> None:
    iteration_count = multiprocessing.Value('i', 0)
    start_iteration_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    log_queue = multiprocessing.SimpleQueue()

    learner = threading.Thread(
        target=run_learner,
        args=(
            num_iterations,
            num_eval_steps,
            learner_agent,
            eval_agent,
            eval_env,
            data_queue,
            log_queue,
            iteration_count,
            start_iteration_event,
            stop_event,
            checkpoint,
            len(actors),
            use_tensorboard,
            tag,
        ),
    )
    learner.start()
    

    logger = threading.Thread(
        target=run_logger,
        args=(log_queue, csv_file),
    )
    logger.start()

    num_actors = len(actors)
    actor_tb_log_prefixes = [None for _ in range(num_actors)]
    if use_tensorboard:

        _step = 1 if num_actors <= 8 else math.ceil(num_actors / 8)
        for i in range(0, num_actors, _step):
            actor_tb_log_prefixes[i] = get_tb_log_prefix(actor_envs[i].spec.id, actors[i].agent_name, tag, 'train')

    processes = []
    for actor, actor_env, tb_log_prefix in zip(actors, actor_envs, actor_tb_log_prefixes):
        p = multiprocessing.Process(
            target=run_actor,
            args=(
                actor,
                actor_env,
                data_queue,
                log_queue,
                num_train_steps,
                iteration_count,
                start_iteration_event,
                stop_event,
                tb_log_prefix,
                debug_screenshots_interval,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        p.close()

    logger.join()

    data_queue.close()


def run_actor(
    actor: types_lib.Agent,
    actor_env: gym.Env,
    data_queue: multiprocessing.Queue,
    log_queue: multiprocessing.SimpleQueue,
    num_train_steps: int,
    iteration_count: multiprocessing.Value,
    start_iteration_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    tb_log_prefix: str = None,
    debug_screenshots_interval: int = 0,
) -> None:
    if not isinstance(actor, types_lib.Agent):
        raise RuntimeError('Expect actor to be a instance of types_lib.Agent.')

    init_absl_logging()

    handle_exit_signal()

    actor_trackers = trackers_lib.make_default_trackers(tb_log_prefix, debug_screenshots_interval)

    while not stop_event.is_set():

        if not start_iteration_event.is_set():
            continue

        logging.info(f'Starting {actor.agent_name} ...')
        iteration = iteration_count.value

        train_stats = run_env_steps(num_train_steps, actor, actor_env, actor_trackers)

        data_queue.put('PROCESS_DONE')

        if start_iteration_event.is_set():
            start_iteration_event.clear()

        log_output = [
            ('iteration', iteration, '%3d'),
            ('role', actor.agent_name, '%2s'),
            ('step', iteration * num_train_steps, '%5d'),
            ('episode_return', train_stats['mean_episode_return'], '% 2.2f'),
            ('num_episodes', train_stats['num_episodes'], '%3d'),
            ('step_rate', train_stats['step_rate'], '%4.0f'),
            ('duration', train_stats['duration'], '%.2f'),
        ]

        log_queue.put(log_output)


def run_learner(
    num_iterations: int,
    num_eval_steps: int,
    learner: types_lib.Learner,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    data_queue: multiprocessing.Queue,
    log_queue: multiprocessing.SimpleQueue,
    iteration_count: multiprocessing.Value,
    start_iteration_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    checkpoint: PyTorchCheckpoint,
    num_actors: int,
    use_tensorboard: bool,
    tag: str = None,
) -> None:
    if not isinstance(learner, types_lib.Learner):
        raise RuntimeError('Expect learner to be a instance of types_lib.Learner.')

    learner_tb_log_prefix = get_tb_log_prefix(eval_env.spec.id, learner.agent_name, tag, 'train') if use_tensorboard else None
    learner_trackers = trackers_lib.make_learner_trackers(learner_tb_log_prefix)
    for tracker in learner_trackers:
        tracker.reset()

    should_run_evaluator = False
    eval_trackers = None
    if num_eval_steps > 0 and eval_agent is not None and eval_env is not None:
        should_run_evaluator = True
        eval_tb_log_prefix = (
            get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, tag, 'eval') if use_tensorboard else None
        )
        eval_trackers = trackers_lib.make_default_trackers(eval_tb_log_prefix)

    for iteration in range(1, num_iterations + 1):
        logging.info(f'Training iteration {iteration}')
        logging.info(f'Starting {learner.agent_name} ...')

        iteration_count.value = iteration

        start_iteration_event.set()
        learner.reset()
        run_learner_loop(learner, data_queue, num_actors, learner_trackers)
        start_iteration_event.clear()
        checkpoint.set_iteration(iteration)
        saved_ckpt = checkpoint.save()

        if saved_ckpt:
            logging.info(f'New checkpoint created at "{saved_ckpt}"')

        if should_run_evaluator is True:
            logging.info(f'Evaluation iteration {iteration}')

            eval_stats = run_env_steps(num_eval_steps, eval_agent, eval_env, eval_trackers)

            log_output = [
                ('iteration', iteration, '%3d'),
                ('role', 'evaluation', '%3s'),
                ('step', iteration * num_eval_steps, '%5d'),
                ('episode_return', eval_stats['mean_episode_return'], '%2.2f'),
                ('num_episodes', eval_stats['num_episodes'], '%3d'),
                ('step_rate', eval_stats['step_rate'], '%4.0f'),
                ('duration', eval_stats['duration'], '%.2f'),
            ]
            log_queue.put(log_output)

        time.sleep(5)

    stop_event.set()

    log_queue.put('PROCESS_DONE')


def run_learner_loop(
    learner: types_lib.Learner,
    data_queue: multiprocessing.Queue,
    num_actors: int,
    learner_trackers: Iterable[Any],
) -> None:
    
    num_done_actors = 0

    while True:

        try:
            item = data_queue.get()
            if item == 'PROCESS_DONE':
                num_done_actors += 1
            else:
                learner.received_item_from_queue(item)
        except queue.Empty:
            pass
        except EOFError:
            pass

        if num_done_actors == num_actors:
            break

        stats_sequences = learner.step()
        if stats_sequences is not None:

            for stats in stats_sequences:
                for tracker in learner_trackers:
                    tracker.step(stats)


def run_logger(log_queue: multiprocessing.SimpleQueue, csv_file: str):
    writer = CsvWriter(csv_file)

    while True:
        try:
            log_output = log_queue.get()
            if log_output == 'PROCESS_DONE':
                break
            log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
            logging.info(log_output_str)
            writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
        except queue.Empty:
            pass
        except EOFError:
            pass


def run_evaluation_iterations(
    num_iterations: int,
    num_eval_steps: int,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    use_tensorboard: bool,
    recording_video_dir: str = None,
):
    test_tb_log_prefix = get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, None, 'test') if use_tensorboard else None
    test_trackers = trackers_lib.make_default_trackers(test_tb_log_prefix)

    if num_iterations > 0 and num_eval_steps > 0:
        for iteration in range(1, num_iterations + 1):
            logging.info(f'Testing iteration {iteration}')

            eval_stats = run_env_steps(num_eval_steps, eval_agent, eval_env, test_trackers)

            log_output = [
                ('iteration', iteration, '%3d'),
                ('step', iteration * num_eval_steps, '%5d'),
                ('episode_return', eval_stats['mean_episode_return'], '% 2.2f'),
                ('num_episodes', eval_stats['num_episodes'], '%3d'),
                ('step_rate', eval_stats['step_rate'], '%4.0f'),
                ('duration', eval_stats['duration'], '%.2f'),
            ]

            log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
            logging.info(log_output_str)
            iteration += 1

    if recording_video_dir is not None and recording_video_dir != '':
        gym_env.play_and_record_video(eval_agent, eval_env, recording_video_dir)


def get_tb_log_prefix(env_id: str, agent_name: str, tag: str, suffix: str) -> str:
    tb_log_prefix = f'{env_id}-{agent_name}'
    if tag is not None and tag != '':
        tb_log_prefix += f'-{tag}'
    tb_log_prefix += f'-{suffix}'
    return tb_log_prefix


def init_absl_logging():
    logging._warn_preinit_stderr = 0
    logging.set_verbosity(logging.INFO)
    logging.use_absl_handler()


def handle_exit_signal():
    def shutdown(signal_code, frame):
        del frame
        logging.info(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
