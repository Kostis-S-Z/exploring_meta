"""
Code taken directly from https://github.com/openai/train-procgen

Logger explanation:

fps: how many frames per second it manages to play calculated based of number of steps & time elapsed
misc/nupdates: number of epochs / iterations / updates per environment
misc/serial_timesteps: number of total steps it took across epochs / iterations / updates per environment
misc:total_timesteps: number of total steps across iterations across all envs (serial_timesteps * num_envs)
"""

import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

import wandb

use_wandb = False

LOG_DIR = '~/Projects/KTH/Thesis/exploring_meta/procgen'

params = dict(
    env_name="starpilot",  # coinrun, starpilot
    mode="easy",  # easy, hard
    num_lvls=1,  # 0-unlimited, 1-test, 200-easy, 500-hard
    num_envs=4,  # max should be 32env ~ 7gb VRAM, 8 is probably optimal
    learning_rate=5e-4,
    ent_coef=.01,
    gamma=.999,
    lam=.95,
    nsteps=256,
    nminibatches=8,
    ppo_epochs=3,
    clip_range=.2,
    timesteps_per_proc=25_000_000,
    use_vf_clipping=True,
    log_interval=100,  # per update = timesteps / batch
    save_interval=10_000,  # per update
)


def main():

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default=params['env_name'])
    parser.add_argument('--distribution_mode', type=str, default=params['mode'],
                        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=params['num_lvls'])
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)

    args = parser.parse_args()

    test_worker_interval = args.test_worker_interval

    if use_wandb:
        import datetime
        date = datetime.datetime.now().strftime("%d_%m_%Hh%M")
        wandb_run = wandb.init(project="l2l", id=args.env_name + '_' + date, config=params)
    else:
        wandb_run = None

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=LOG_DIR, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=params['num_envs'], env_name=args.env_name, num_levels=num_levels, start_level=args.start_level,
                      distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)


    logger.info("training")
    ppo2.learn(
        wandb_run=wandb_run,
        env=venv,
        network=conv_fn,
        total_timesteps=params['timesteps_per_proc'],
        save_interval=params['save_interval'],
        nsteps=params['nsteps'],
        nminibatches=params['nminibatches'],
        lam=params['lam'],
        gamma=params['gamma'],
        noptepochs=params['ppo_epochs'],
        log_interval=params['log_interval'],
        ent_coef=params['ent_coef'],
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=params['use_vf_clipping'],
        comm=comm,
        lr=params['learning_rate'],
        cliprange=params['clip_range'],
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )


if __name__ == '__main__':
    main()
