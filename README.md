# Experiments on Meta Learning algorithms

Exploring the effects of Meta Learning algorithms (MAML & ANIL) on Vision and RL tasks through a different lens.

Track runs & results in [weights&biases](https://app.wandb.ai/kosz/l2l)

## Installing

1. Install Cython:

```pip install cython```

2. Install my forked version of [learn2learn](https://github.com/learnables/learn2learn) specifically modified for experiments for this repo:

```pip install -e git+https://github.com/Kostis-S-Z/learn2learn.git@metaworld#egg=learn2learn```

_Note: There is a [bug](https://stackoverflow.com/questions/26193365/pycharm-does-not-recognize-modules-installed-in-development-mode) in PyCharm that packages installed in development mode might not be recognised at first and you need to re-open the project in order for it to be properly indexed._

3. Install core dependencies

```pip install -r requirements.txt```

#### Dependencies for RL & Meta-World experiments

4. Install [OpenAI's Gym](https://github.com/openai/gym):

```pip install gym==0.15.4```

5. Install [cherry](https://github.com/learnables/cherry):

```pip install git+https://github.com/Kostis-S-Z/cherry@master#egg=cherry-rl```

6. Install [metaworld](https://github.com/rlworkgroup/metaworld) :

```pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld```

#### Dependencies for Procgen experiments

5. Install [Procgen](https://github.com/openai/procgen):

```pip install procgen==0.9.2```


6. Install [baselines](https://github.com/openai/baselines):

```pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip```


7. Install [tensorflow v1.15](https://www.tensorflow.org/):

```pip install tensorflow==1.15.0```


8. Install [mpi4py](https://github.com/openai/baselines):

```pip install mpi4py==3.0.3```

#### Optional dependencies

8. Track results with [W&B](https://www.wandb.com/):

```pip install wandb```

9. For development:

```pip install pytest```


## Guide

For a vision walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/vision/VISION_CODE_WALKTHROUGH.md)

For an RL walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/rl/RL_CODE_WALKTHROUGH.md)

## Run

Bug: Currently the project depends on running the scripts through PyCharm by setting the project root as sources root. This will be fixed soon...

~~Simply run the python scripts like so:~~ `python3 maml_vision.py`

Change hyper-parameters / experiment settings like so:
```
python3 maml_rl.py --dataset omni
                   --ways 5
                   --shots 5
                   --outer_lr 0.1
```

For scripts that can use MPI run:
```mpiexec -np 8 python3 maml_rl.py ```

## Acknowledgements

A lot of the code for the implementations of MAML & ANIL depends on the [learn2learn](https://github.com/learnables/learn2learn) library.
