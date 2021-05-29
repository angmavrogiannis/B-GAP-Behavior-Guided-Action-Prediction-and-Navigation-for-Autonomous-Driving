# B-GAP-Behavior-Guided-Action-Prediction-for-Autonomous-Navigation
This repository contains code and technical details for the paper we submitted to IROS 2021:

**[B-GAP: Behavior-Guided Action Prediction and Navigation for Autonomous Driving](https://arxiv.org/abs/2011.03748)**

Authors: Angelos Mavrogiannis, Rohan Chandra, and Dinesh Manocha

Please cite our work if you found it useful:

```
@article{mavrogiannis2021bgap,
  title={B-GAP: Behavior-Guided Action Prediction and Navigation for Autonomous Driving},
  author={Mavrogiannis, Angelos and Chandra, Rohan and Manocha, Dinesh},
  journal={arXiv preprint arXiv:2011.03748},
  year={2021}
}
```

<p align="center">
  <img src="/images/aggro.gif" alt="animated" />
</p>

## Overview
We present an algorithm for behaviorally-guided action prediction and local navigation for autonomous driving in dense traffic scenarios. Our approach classifies the driver behavior of other vehicles or road-agents (aggressive or conservative) and considers that information for decision-making and safe driving. We present a behavior-driven simulator that can generate trajectories corresponding to different levels of aggressive behaviors, and we use our simulator to train a reinforcement learning policy using a multilayer perceptron neural network. We use our reinforcement learning-based navigation scheme to compute safe trajectories for the ego-vehicle accounting for aggressive driver maneuvers such as overtaking, over-speeding, weaving, and sudden lane changes. We have integrated our algorithm with the OpenAI gym-based "Highway-Env" simulator and demonstrate the benefits of our navigation algorithm in terms of reducing collisions by 3.25−26.90% and handling scenarios with 2.5× higher traffic density.

<p align="center">
<img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/bgap_offline.PNG" height="180" width="2000">
</p>

**Offline Training** : We highlight our behavior-guided navigation policy for autonomous driving. We use a behavior-rich simulator that can generate aggressive or conservative driving styles. In *Step 1*, we use the CMetric behavior classification algorithm to compute a set of parameters that characterize aggressive behaviors such as over-speeding, overtaking, and sudden lane changing. In *Step 2*,  we use these parameters to train a behavior-based action class navigation policy for action prediction and local navigation.

<p align="center">
<img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/bgap_runtime.PNG" height="480" width="800">
</p>

**Online Training** : We use our behavior-guided trained policy and the final simulation parameters computed using offline training. During an episode at runtime, we use the trained policy to predict the next action of the ego-vehicle given the current state of the traffic environment, which is represented in the form of a feature matrix. The predicted action (in this case, "turn left") is converted into the final local trajectory using the internal controls of the simulator, modified by the parameters that take into account the behavior of traffic agents.

## Dependencies
```python
Python: ">=3.6"
PyTorch: ">=1.4.0"
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
- To use the behavior-rich simulator including conservative and aggressive vehicles use [master](/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/tree/master) branch.
- To use the *[default](https://github.com/eleurent/highway-env)* OpenAI gym-based simulator switch to the [default_sim](/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/tree/default_sim) branch. (Credits to Edouard Leurent)

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
To train the Graph Convolutional Network (GCN) agent run the following command by navigating to the `rl-agents/scripts/` subdirectory:

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
We evaluate two models -- Multi-layer Perceptron (MLP) and GCN in dense traffic scenarios, varying the number of aggressive and conservative agents, and compare the performance of the trained GCN agent in the behavior-rich simulator and the default simulator. We apply two metrics for evaluation over 100 episodes and averaged at test time:

- **Average Speed (Avg. Spd.)** of the ego-vehicle, since it captures the distance per second covered in a varying time interval.
- **Number of lane changes (#LC)** performed by the ego-vehicle on average during the given duration. In general, fewer lane changes imply that the ego vehicle can cover the same distance with fewer maneuvers. We notice that our approach based on GCN results in approximately 50% reduction in the number of lane changes as compared to MLP.

### Dense Traffic (n = 40 vehicles)

<table>
  <tr>
    <td>Collision frequency</td>
     <td>Average speed</td>
     <td>Average number of lane changes</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/barcharts/crash40.png"></td>
    <td valign="top"><img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/barcharts/vel40.png"></td>
    <td valign="top"><img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/barcharts/lc40.png"></td>
  </tr>
</table>

### Sparse Traffic (n = 5 vehicles)

<table>
  <tr>
    <td>Collision frequency</td>
     <td>Average speed</td>
     <td>Average number of lane changes</td>
  </tr>
  <tr>
    <td valign="top"><img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/barcharts/crash5.png"></td>
    <td valign="top"><img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/barcharts/vel5.png"></td>
    <td valign="top"><img src="https://github.com/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-and-Navigation-for-Autonomous-Driving/blob/master/images/barcharts/lc5.png"></td>
  </tr>
</table>

### Evaluation
The behavior-rich policy is compared with the default policy in traffic scenarios of varying density and driver behavior. The behavior-rich policy leads to improved navigation and reduced number of collisions by adjusting the speed of the ego-vehicle to the behaviors of its neighbors (higher average speed for higher percentages of aggressive neighbors) and decreasing its lane changes to respect unpredictable drivers. When *n=5* in the aggressive scenario, the number of lane changes is higher, as the ego-vehicle is more confident to perform lane-changing maneuvers due to the sparsity of the traffic.

For extended results and comparison with prior work, please refer to our [paper](https://arxiv.org/abs/2011.03748).

For visualized results and comparison, please watch the **[video](https://youtu.be/AKa0esw88sQ)** submitted as supplementary material.
