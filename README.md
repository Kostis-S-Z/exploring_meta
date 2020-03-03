# Experiments on Meta Learning algorithms

_add info here_


## Installing

1. Install Cython:

```pip install cython```

2. Install my forked version of [learn2learn](https://github.com/learnables/learn2learn) specifically modified for experiments for this repo:

```pip install -e git://github.com/Kostis-S-Z/learn2learn.git@exploring_meta#egg=learn2learn```

3. Install github version of torch summary (because PyPI package hasn't been updated as of now to support summary_string function

```pip install -e git://github.com/sksq96/pytorch-summary.git@4ee5ac5#egg=torchsummary```

4. Install some extra dependencies for better visualization / logging:

```pip install -r requirements.txt```