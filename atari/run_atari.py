import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ALGORITHMS = [
    'ppo',
    'ppo_icm',
    'ppo_rnd',
    'ppo_noveld',
    'ppo_ama',
    'ppo_tecle',
    'ppo_noisy',
]

ENVIRONMENTS = [
    'MontezumaRevenge',
    'Gravitar',
    'PrivateEye',
    'Pitfall',
    'BankHeist',
]


ORCH_DIR = Path(__file__).parent.resolve()
SHARED_DIR = ORCH_DIR.parent / 'shared'
LOGS_DIR = ORCH_DIR / 'logs'
CHECKPOINTS_DIR = ORCH_DIR / 'checkpoints'


_rf_env_override = os.environ.get('RF_ENV_PYTHON')
PYTHON = _rf_env_override if _rf_env_override and Path(_rf_env_override).exists() else sys.executable


_env = os.environ.copy()
_prev = _env.get('PYTHONPATH', '')
_env['PYTHONPATH'] = str(SHARED_DIR) + (os.pathsep + _prev if _prev else '')


def build_command(
    algo: str,
    env: str,
    seed: int,
    num_iterations: int = None,
    num_actors: int = None,
    beta: float = None,
    sticky: bool = False,
    load_checkpoint: str = None,
) -> list[str]:
    script = SHARED_DIR / algo / 'run_atari.py'
    csv_path = LOGS_DIR / f'{algo}_{env}_seed{seed}_results.csv'
    ckpt_dir = CHECKPOINTS_DIR / algo / env / f'seed{seed}'

    cmd = [
        PYTHON, str(script),
        f'--environment_name={env}',
        f'--seed={seed}',
        f'--results_csv_path={csv_path}',
        f'--checkpoint_dir={ckpt_dir}',
        '--use_tensorboard=true',
    ]

    if num_iterations is not None:
        cmd.append(f'--num_iterations={num_iterations}')

    if num_actors is not None:
        cmd.append(f'--num_actors={num_actors}')

    if algo == 'ppo_tecle':
        if beta is not None:
            cmd.append(f'--beta={beta}')
        if sticky:
            cmd.append('--sticky=true')

    if load_checkpoint:
        cmd.append(f'--load_checkpoint={load_checkpoint}')

    return cmd


def run_experiment(
    algo: str,
    env: str,
    seed: int,
    num_iterations: int = None,
    num_actors: int = None,
    beta: float = None,
    sticky: bool = False,
    load_checkpoint: str = None,
    dry_run: bool = False,
) -> int:
    cmd = build_command(algo, env, seed, num_iterations, num_actors, beta, sticky, load_checkpoint)
    cmd_str = ' '.join(str(c) for c in cmd)

    print(f'\n{"=" * 70}')
    print(f'  Algorithm  : {algo}')
    print(f'  Environment: {env}')
    print(f'  Seed       : {seed}')
    print(f'  Command    : {cmd_str}')
    print(f'{"=" * 70}')

    if dry_run:
        return 0

    start = time.time()
    result = subprocess.run(cmd, cwd=ORCH_DIR, env=_env)
    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)

    status = 'DONE' if result.returncode == 0 else f'FAILED (code {result.returncode})'
    print(f'  [{status}] Elapsed: {minutes}m {seconds}s')
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run Atari hard-exploration benchmarks.')
    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=['ppo', 'ppo_icm', 'ppo_rnd', 'ppo_noveld', 'ppo_ama', 'ppo_tecle', 'ppo_noisy'],
        default=ALGORITHMS,
        help='Algorithms to run (default: all enabled in ALGORITHMS list).',
    )
    parser.add_argument(
        '--envs',
        nargs='+',
        default=ENVIRONMENTS,
        help='Environments to run (default: all in ENVIRONMENTS list).',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[1, 3, 5, 10, 15],
        help='Random seeds (default: 1 3 5 10 15).',
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=None,
        help='Override num_iterations for each experiment (default: use script default).',
    )
    parser.add_argument(
        '--num_actors',
        type=int,
        default=None,
        help='Override num_actors for each experiment (default: use script default).',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='Colored noise beta for PPO-TeCLE (0=white, 1=pink, 2=red). Default: use script default.',
    )
    parser.add_argument(
        '--sticky',
        action='store_true',
        help='Enable sticky actions for PPO-TeCLE.',
    )
    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to restore from (default: no restore).',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print commands without executing them.',
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(args.algorithms) * len(args.envs) * len(args.seeds)
    print(f'Running {total} experiments:')
    print(f'  Algorithms  : {args.algorithms}')
    print(f'  Environments: {args.envs}')
    print(f'  Seeds       : {args.seeds}')
    if args.beta is not None:
        print(f'  Noise beta  : {args.beta}')
    if args.sticky:
        print(f'  Sticky      : True')

    failures = []
    count = 0
    for algo in args.algorithms:
        for env in args.envs:
            for seed in args.seeds:
                count += 1
                print(f'\n[{count}/{total}]', end='')
                rc = run_experiment(
                    algo, env, seed,
                    num_iterations=args.num_iterations,
                    num_actors=args.num_actors,
                    beta=args.beta,
                    sticky=args.sticky,
                    load_checkpoint=args.load_checkpoint,
                    dry_run=args.dry_run,
                )
                if rc != 0:
                    failures.append((algo, env, seed))

    print(f'\n{"=" * 70}')
    if failures:
        print(f'FAILED experiments ({len(failures)}):')
        for algo, env, seed in failures:
            print(f'  {algo} | {env} | seed={seed}')
        sys.exit(1)
    else:
        print(f'All {total} experiments completed successfully.')


if __name__ == '__main__':
    main()
