# B-GAP-Behavior-Guided-Action-Prediction-for-Autonomous-Navigation
This repository contains code and technical details for the paper we submitted to ICRA 2021:

**[B-GAP: Behavior-Guided Action Prediction for Autonomous Navigation](https://arxiv.org/abs/2011.03748)**

Authors: Angelos Mavrogiannis, Rohan Chandra, and Dinesh Manocha

Please cite our work if you found it useful:

```
@article{mavrogiannis2020bgap,
  title={B-GAP: Behavior-Guided Action Prediction for Autonomous Navigation},
  author={Mavrogiannis, Angelos and Chandra, Rohan and Manocha, Dinesh},
  journal={arXiv preprint arXiv:2011.03748},
  year={2020}
}
```

## Overview
We present a novel learning algorithm for action prediction and local navigation for autonomous driving. Our approach classifies the driver behavior of other vehicles or road-agents (aggressive or conservative) and takes that into account for decision making and safe driving. We present a behavior-driven simulator that can generate trajectories corresponding to different levels of aggressive behaviors and use our simulator to train a policy using graph convolutional networks. We use a reinforcement learning-based navigation scheme that uses a proximity graph of traffic agents and computes a safe trajectory for the ego-vehicle that accounts for aggressive driver maneuvers such as overtaking, over-speeding, weaving, and sudden lane changes. We have integrated our algorithm with OpenAI gym-based "Highway-Env" simulator and demonstrate the benefits in terms of improved navigation in different scenarios.

<p align="center">
<img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-for-Autonomous-Navigation/blob/master/images/offline.png" height="180" width="2000">
</p>

**Offline Training** : We use a behavior-rich simulator that can generate aggressive or conservative driving styles. In Step 1,we use the CMetric behavior classification algorithm to compute a set of parameters that characterize aggressive behaviors such as overspeeding, overtaking, and sudden lane-changing. In Step 2, we use these parameters to train a behavior-based action class navigation policy for action prediction and local navigation.

<p align="center">
<img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-for-Autonomous-Navigation/blob/master/images/online.png" height="480" width="800">
</p>

**Online Training** : We use our behavior-guided trained policy and the final simulation parameters computed using offline training. During an episode at runtime, we use the trained policy to predict the next action of the ego vehicle given the current state of the traffic environment, which is represented in the form of a traffic-graph. The predicted action (in this case, \`\`turn left\'\') is converted into the final local trajectory using the internal controls of the simulator, modified by the parameters that take into account the behavior of traffic agents.

## Dependencies
```python
Python: ">=3.6"
PyTorch: ">=1.4.0"
PyTorch Geometric
TensorBoard
```

### Installing PyTorch Geometric

To install the binaries for PyTorch 1.7.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` and `torch-1.7.0` should be replaced by `torch-1.4.0`, `torch-1.5.0`, `torch-1.5.1`, or `torch-1.6.0` depending on your PyTorch installation.

## Usage
### Simulator Environment
- To use the behavior-rich simulator including conservative and aggressive vehicles use [master](/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-for-Autonomous-Navigation/tree/master) branch.
- To use the default OpenAI gym-based simulator switch to the [default_sim](/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-for-Autonomous-Navigation/tree/default_sim) branch.

### Build
First, build the code using the following commands:

```
cd highway-env/
sudo python3 setup.py install
cd ../rl-agents/
sudo python3 setup.py install
```

In case `python3` points to an older python version, either update the default python3 version, or replace `python3` with the path to the python version you intend to use.

### Training and Testing
To train the GCN agent run the following command by navigating to the `rl-agents/scripts/` subdirectory:

```
python experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/gcn.json --train --episodes=2000 --name-from-config
```

To test the GCN agent run the following command from the same directory:

```
python experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/gcn.json --test --episodes=10 --name-from-config --recover-from=/path/to/output/folder
```

where `/path/to/output/folder` should correspond to the output file of the trained model. Trained models are saved in the subdirectory `/rl-agents/scripts/out/HighwayEnv/DQNAgent/`. Add `--no-display` to disable rendering of the environment.

To change the structure of the GCN or modify the parameters for the training, modify [gcn.json](/rl-agents/scripts/configs/HighwayEnv/agents/DQNAgent/gcn.json).

## Results
