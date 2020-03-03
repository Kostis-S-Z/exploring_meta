# Experiments on Meta Learning algorithms

_add info here_


## Installing

1. Install Cython:

```pip install cython```

2. Install my forked version of [learn2learn](https://github.com/learnables/learn2learn) specifically modified for experiments for this repo:

```pip install -e git+https://github.com/Kostis-S-Z/learn2learn.git@exploring_meta#egg=learn2learn```

_Note: There is a [bug](https://stackoverflow.com/questions/26193365/pycharm-does-not-recognize-modules-installed-in-development-mode) in PyCharm that packages installed in development mode might not be recognised at first and you need to re-open the project in order for it to be properly indexed._

3. Install github version of torch summary (because PyPI package hasn't been updated as of [now](https://github.com/sksq96/pytorch-summary/issues/115) to support summary_string function)

```pip install git+https://github.com/sksq96/pytorch-summary.git@4ee5ac5#egg=torchsummary```

4. Install some extra dependencies for better visualization / logging:

```pip install -r requirements.txt```