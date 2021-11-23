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

<p align="center">
  <img src="/images/aggro.gif" alt="animated" />
</p>

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
- To use the *[default](https://github.com/eleurent/highway-env)* OpenAI gym-based simulator switch to the [default_sim](/angmavrogiannis/B-GAP-Behavior-Guided-Action-Prediction-for-Autonomous-Navigation/tree/default_sim) branch. (Credits to Edouard Leurent)

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

### Dense Traffic (n = 20 vehicles)
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan=2>Default</th>
            <th colspan=2>Conservative</th>
            <th colspan=2>Aggressive</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><i>(n=20)</i></td>
            <td><i>Avg. Spd. (m/s)</i></td>
            <td><i>#LC</i></td>
            <td><i>Avg. Spd. (m/s)</i></td>
            <td><i>#LC</i></td>
            <td><i>Avg. Spd. (m/s)</i></td>
            <td><i>#LC</i></td>
        </tr>
        <tr>
            <td style="text-align: center">MLP</td>
            <td style="text-align: center">22.65</td>
            <td style="text-align: center">4.10</td>
            <td style="text-align: center">19.70</td>
            <td style="text-align: center">2.90</td>
            <td style="text-align: center">28.80</td>
           <td style="text-align: center">2.60</td>
        </tr>
        <tr>
            <td style="text-align: center">GCN</td>
            <td style="text-align: center">22.26</td>
            <td style="text-align: center">0.82</td>
            <td style="text-align: center">18.90</td>
            <td style="text-align: center">2.33</td>
            <td style="text-align: center">29.00</td>
            <td style="text-align: center">1.40</td>
        </tr>
    </tbody>
</table>

### Sparse Traffic (n = 10 vehicles)
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan=2>Default</th>
            <th colspan=2>Conservative</th>
            <th colspan=2>Aggressive</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><i>(n=20)</i></td>
            <td><i>Avg. Spd. (m/s)</i></td>
            <td><i>#LC</i></td>
            <td><i>Avg. Spd. (m/s)</i></td>
            <td><i>#LC</i></td>
            <td><i>Avg. Spd. (m/s)</i></td>
            <td><i>#LC</i></td>
        </tr>
        <tr>
            <td style="text-align: center">MLP</td>
            <td style="text-align: center">23.75</td>
            <td style="text-align: center">6.25</td>
            <td style="text-align: center">21.40</td>
            <td style="text-align: center">3.50</td>
            <td style="text-align: center">29.16</td>
           <td style="text-align: center">2.06</td>
        </tr>
        <tr>
            <td style="text-align: center">GCN</td>
            <td style="text-align: center">23.60</td>
            <td style="text-align: center">0.35</td>
            <td style="text-align: center">20.60</td>
            <td style="text-align: center">1.60</td>
            <td style="text-align: center">28.90</td>
            <td style="text-align: center">1.30</td>
        </tr>
    </tbody>
</table>

### Advantages of Our Behavior-Rich Simulator
As the default simulator does not contain any behavior-rich trajectories, the ego-vehicle is not able to adjust its average speed and maintains a speed that is approximately the average of the average speeds obtained when operating in conservative and aggressive environments. Furthermore, the ego-vehicle does not follow a realistic lane-changing pattern in the default environment as it performs excessive number of lane changes using the MLP, but mostly follows the traffic with very few lane changes using the GCN.

### Behavior-Guided Actions
- **Conservative Traffic**: The ego-vehicle learns to confidently overtake slow-moving traffic. This reduces its average speed, but increases the number of lane changes.
- **Aggressive Traffic**: The ego-vehicle learns to act carefully around aggressive drivers, choosing to stay in the same lane. This result in fewer lane changes and safer navigation, as compared to the one observed in current AVs.


For visualized results and comparison, please watch the **[video](https://youtu.be/AKa0esw88sQ)** submitted as supplementary material.
