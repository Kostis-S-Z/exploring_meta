# Experiments on Meta Learning algorithms

Exploring the effects of Meta Learning algorithms (MAML & ANIL) on Vision and RL tasks through a different lens.

Track runs & results in [weights&biases](https://app.wandb.ai/kosz/l2l)

## Installing

1. Install Cython:

```pip install cython```

2. Install my forked version of [learn2learn](https://github.com/learnables/learn2learn) specifically modified for experiments for this repo:

```pip install -e git+https://github.com/Kostis-S-Z/learn2learn.git@exploring_meta#egg=learn2learn```

_Note: There is a [bug](https://stackoverflow.com/questions/26193365/pycharm-does-not-recognize-modules-installed-in-development-mode) in PyCharm that packages installed in development mode might not be recognised at first and you need to re-open the project in order for it to be properly indexed._

3. Install github version of torch summary (because PyPI package hasn't been updated as of [now](https://github.com/sksq96/pytorch-summary/issues/115) to support summary_string function)

```pip install git+https://github.com/sksq96/pytorch-summary.git@4ee5ac5#egg=torchsummary```

#### RL dependencies

4. Install [cherry](https://github.com/learnables/cherry):

```pip install cherry-rl```

5. Install [baselines](https://github.com/openai/baselines) (Only for Procgen):

```pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip```

6. Install [metaworld](https://github.com/rlworkgroup/metaworld) :

```pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld```

7. Install the rest of dependencies:

```pip install -r requirements.txt```


## Guide

For a vision walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/VISION_CODE_WALKTHROUGH.md)

For an RL walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/RL_CODE_WALKTHROUGH.md)

## Run

Simply run the python scripts like so: `python3 maml_vision.py`

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
