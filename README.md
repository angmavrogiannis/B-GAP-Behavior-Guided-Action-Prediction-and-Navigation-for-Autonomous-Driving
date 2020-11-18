# Autonomous Decision Making Dense Traffic

# Dependencies
This project requires the following libraries:
- python3 
- pytorch
- tensorboardX

To install the binaries for PyTorch 1.6.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-geometric
```

where `${CUDA}` should be replaced by `cpu` depending on your PyTorch installation.

For instructions compatible with older PyTorch versions, documentation and examples, go *[here](https://github.com/rusty1s/pytorch_geometric)*.

# Build instructions
First build the code using the following commands:
```
cd highway-env/
sudo python3 setup.py install
cd ../rl-agents/
sudo python3 setup.py install
```
# Run instructions
To train the dqn agent run the following command by navigating to the `rl-agents/scripts/` subdirectory:
```
python3 experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/DQNAgent/dqn.json --train --episodes=2000 --name-from-config
```

To train the agent using fitted Q iteration run the following command from the same subdirectory:
```
python3 experiments.py evaluate configs/HighwayEnv/env.json configs/HighwayEnv/agents/FTQAgent/baseline.json --train --episodes=2000 --name-from-config
```
