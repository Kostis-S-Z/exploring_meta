[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


# Experiments on Model-Agnostic Meta-Learning

## Table of Contents

* [About](#about)
  * [Built With](#built-with)
  * [Roadmap Overview](#roadmap-overview)
* [Installation](#Installation)
* [Usage](#usage)
  * [Project Structure](#project-structure)
  * [Training example](#training-example)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


<!-- ABOUT THE PROJECT -->
## About

Exploring the effects of the Model-Agnostic Meta-Learning algorithms MAML & ANIL on Vision and meta-RL tasks. Including Representation similarity experiments for further insights on the network dynamics during meta-testing (adaptation) and Continual Learning experiments to test their ability to _adapt without forgetting_.

Check out some of our runs & results in our [W&B Project](https://app.wandb.ai/kosz/l2l)

### Built With

* [PyTorch](https://pytorch.org/)
* [learn2learn](https://github.com/learnables/learn2learn/)
* [Meta-World](https://github.com/rlworkgroup/metaworld)


<!-- Installation -->
## Installation

*_NOTE_*: This repo was build with Meta-World experiments in mind and thus depends on the proprietary environment [MuJoCo](http://mujoco.org/). So you would need to first get a license for that and install it before you can run the experiments of this repo (Yes, even for the vision or RL-but-not-Meta-World experiments it's required, I am [working on](https://github.com/Kostis-S-Z/exploring_meta/issues/45) separating this dependency)

1. Clone the repo
```sh
git clone https://github.com/github_username/repo_name.git
```

1b. _(Optional, but highly recommended)_ Make a virtual environment

```python3 -m venv meta_env``` or ``` virtualenv meta_env```

2. Install Cython:

```pip3 install cython```

3. Install core dependencies

```pip3 install -r requirements.txt```


4. _(Optional)_ You can easily track models with [W&B](https://www.wandb.com/)

```pip install wandb```


### Roadmap Overview

Currently implemented are the following:

#### Vision: Omniglot, Mini-ImageNet
- [x] MAML with CNN
- [x] ANIL with CNN

#### RL: Particles2D, MuJoCo Ant, Meta-World
- [X] MAML-PPO
- [X] MAML-TRPO
- [X] ANIL-PPO
- [X] ANIL-TRPO
- [X] Baselines: PPO, TRPO & VPG

#### Possible future extensions

_Check branches:_
- [ ] Procgen (it works, but incredibly difficult to train)
- [ ] MAML/ANIL - VPG (very unstable)


<!-- Usage -->
## Usage

For a vision walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/vision/VISION_CODE_WALKTHROUGH.md)

For an RL walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/rl/RL_CODE_WALKTHROUGH.md)

### Project Structure

#### Modules (core dependencies)

**core_functions**: _Functions necessary to train & evaluate RL and Vision_

**utils**: _Functions for data processing, environment making etc (not related to algorithms)_
 
#### Running scripts

**baselines**: Scripts to train & evaluate RL and vision 

**rl**: Scripts to train & evaluate meta-RL

**vision**: Scripts to train & evaluate meta-vision

**misc_scripts**: Scripts to run Continual Learning & Representation Change experiments or to render trained policies in Meta-World


### Training example

```
python3 rl/maml_trpo.py --outer_lr 0.1 --adapt_steps 3
```

_If you get import errors, add the project's root to PYTHONPATH. Make sure the content root is added to the PYTHONPATH in the configuration_ or _in the .bashrc file add_ `export PYTHONPATH="${PYTHONPATH}:~/Path/to/project"`

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

Konstantinos Saitas - Zarkias (Kostis S-Z)

_Feel free to open an issue for anything related to this repo._

Project Link: [https://github.com/github_username/repo_name](https://github.com/Kostis-S-Z/exploring_meta)

## Acknowledgements

Many thanks to fellow researchers & colleagues at [RISE](https://www.ri.se/en), [KTH](https://www.kth.se/en) and Séb Arnold from the [learn2learn](https://learn2learn.net/) team for insightful discussions about this project.  
