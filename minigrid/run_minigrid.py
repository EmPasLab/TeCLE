import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ALGORITHMS = [
    'a2c',
    'ppo',
    'icm',
    'rnd_rev',
    'TeCLE',
]

ENVIRONMENTS = [
    'MiniGrid-Empty-8x8-v0',
    'MiniGrid-Empty-16x16-v0',
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-DoorKey-16x16-v0',
    'MiniGrid-KeyCorridorS3R3-v0',
    'MiniGrid-Unlock-v0',
    'MiniGrid-LavaCrossingS9N3-v0',
    'MiniGrid-LavaCrossingS11N5-v0',
    'MiniGrid-MultiRoom-N2-S4-v0',
]


ORCH_DIR = Path(__file__).parent.resolve()


_rf_env_override = os.environ.get('RF_ENV_PYTHON')
PYTHON = _rf_env_override if _rf_env_override and Path(_rf_env_override).exists() else sys.executable


def build_command(
    algo: str,
    env: str,
    seed: int,
    frames: int,
    procs: int,
    log_interval: int,
    save_interval: int,
    noisy_tv: str,
    noise_beta: float,
    heatmap: bool,
) -> list[str]:
    model = f'{algo}_{env}_{frames}_noisy_tv_{noisy_tv}_noise_beta_{noise_beta}_seed_{seed}'

    cmd = [
        PYTHON, '-m', 'scripts.train',
        '--algo', algo,
        '--env', env,
        '--seed', str(seed),
        '--frames', str(frames),
        '--procs', str(procs),
        '--log-interval', str(log_interval),
        '--save-interval', str(save_interval),
        '--model', model,
        '--noisy_tv', noisy_tv,
        '--noise_beta', str(noise_beta),
    ]
    if heatmap:
        cmd.append('--heatmap')
    return cmd


def run_experiment(
    algo: str,
    env: str,
    seed: int,
    frames: int,
    procs: int,
    log_interval: int,
    save_interval: int,
    noisy_tv: str,
    noise_beta: float,
    heatmap: bool,
    dry_run: bool,
) -> int:
    cmd = build_command(algo, env, seed, frames, procs, log_interval, save_interval, noisy_tv, noise_beta, heatmap)
    cmd_str = ' '.join(str(c) for c in cmd)

    print(f'\n{"=" * 70}')
    print(f'  Algorithm   : {algo}')
    print(f'  Environment : {env}')
    print(f'  Seed        : {seed}')
    print(f'  Noisy TV    : {noisy_tv}')
    print(f'  Noise beta  : {noise_beta}')
    print(f'  Command     : {cmd_str}')
    print(f'{"=" * 70}')

    if dry_run:
        return 0

    start = time.time()
    result = subprocess.run(cmd, cwd=ORCH_DIR)
    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)

    status = 'DONE' if result.returncode == 0 else f'FAILED (code {result.returncode})'
    print(f'  [{status}] Elapsed: {minutes}m {seconds}s')
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run MiniGrid hard-exploration benchmarks.')
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=ALGORITHMS,
                        help='Algorithms to run (default: all enabled in ALGORITHMS list).')
    parser.add_argument('--envs', nargs='+', default=['MiniGrid-Empty-8x8-v0'],
                        help='Environments to run (default: MiniGrid-Empty-8x8-v0).')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 3, 5, 10, 15],
                        help='Random seeds (default: 1 3 5 10 15).')
    parser.add_argument('--frames', type=int, default=5_000_000,
                        help='Number of training frames per run (default: 5_000_000).')
    parser.add_argument('--procs', type=int, default=16,
                        help='Number of parallel envs (default: 16).')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging frequency in updates (default: 10).')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Checkpoint frequency in updates (default: 50).')
    parser.add_argument('--noisy_tv', nargs='+', choices=['True', 'False'], default=['False'],
                        help='Whether to apply the Noisy-TV wrapper (default: False).')
    parser.add_argument('--noise_beta', nargs='+', type=float, default=[-1.0],
                        help='Colored-noise beta values to sweep (default: -1.0).')
    parser.add_argument('--heatmap', action='store_true',
                        help='Enable exploration heatmap logging.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing them.')
    args = parser.parse_args()

    total = len(args.algorithms) * len(args.envs) * len(args.seeds) * len(args.noisy_tv) * len(args.noise_beta)
    print(f'Running {total} experiments:')
    print(f'  Algorithms   : {args.algorithms}')
    print(f'  Environments : {args.envs}')
    print(f'  Seeds        : {args.seeds}')
    print(f'  Noisy TV     : {args.noisy_tv}')
    print(f'  Noise beta   : {args.noise_beta}')

    failures = []
    count = 0
    for algo in args.algorithms:
        for env in args.envs:
            for seed in args.seeds:
                for noisy in args.noisy_tv:
                    for beta in args.noise_beta:
                        count += 1
                        print(f'\n[{count}/{total}]', end='')
                        rc = run_experiment(
                            algo, env, seed,
                            frames=args.frames,
                            procs=args.procs,
                            log_interval=args.log_interval,
                            save_interval=args.save_interval,
                            noisy_tv=noisy,
                            noise_beta=beta,
                            heatmap=args.heatmap,
                            dry_run=args.dry_run,
                        )
                        if rc != 0:
                            failures.append((algo, env, seed, noisy, beta))

    print(f'\n{"=" * 70}')
    if failures:
        print(f'FAILED experiments ({len(failures)}):')
        for algo, env, seed, noisy, beta in failures:
            print(f'  {algo} | {env} | seed={seed} | noisy={noisy} | beta={beta}')
        sys.exit(1)
    else:
        print(f'All {total} experiments completed successfully.')


if __name__ == '__main__':
    main()
