from __future__ import division, print_function, absolute_import
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle
 

class HighwayEnv(AbstractEnv):
    """
        A highway driving environment.

        The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -10
    """ The reward received when colliding with a vehicle."""
    RIGHT_LANE_REWARD = 0.2
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    HIGH_VELOCITY_REWARD = 1
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = 0.5
    """ The reward received at each lane change action."""

    def default_config(self):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "lanes_count": 4,
            "vehicles_count": 40,
            "controlled_vehicles":1,
            # aggressive vehicles
            "aggressive_vehicle_type": "highway_env.vehicle.behavior.AggressiveCar",
            "aggressive_vehicle_type2": "highway_env.vehicle.behavior.VeryAggressiveCar",
            "num_aggressive": 0,
            "duration": 40,  # [s]
            "initial_spacing": 1,
            "collision_reward": self.COLLISION_REWARD
        })
        return config

    def reset(self):
        self._create_road()
        self._create_vehicles()
        self.steps = 0
        return super(HighwayEnv, self).reset()

    def step(self, action):
        self.steps += 1
        state_copy = self.simplify()
        vehicles_list = state_copy.road.vehicles
        # print(vehicles_list)
        pos_list = []
        for v in vehicles_list:
            pos_list.append([v.position, v.counter])
        # removed pos_list from the returned arguments
        return super(HighwayEnv, self).step(action)

    def _create_road(self):
        """
            Create a road composed of straight adjacent lanes.
        """
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = MDPVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        self.road.vehicles.append(self.vehicle)

        # create conservative cars on the road
        vehicles_type1 = utils.class_from_path(self.config["other_vehicles_type"])
        vehicles_type2 = utils.class_from_path(self.config["aggressive_vehicle_type"])
        vehicles_type3 = utils.class_from_path(self.config["aggressive_vehicle_type2"])
        # add some aggressive vehicles in the road
        count_aggressive = 0
        for _ in range(self.config["vehicles_count"]+self.config["num_aggressive"]):
            a = np.random.randint(low=1, high=5)
            if a==1:
                count_aggressive += 1
                self.road.vehicles.append(vehicles_type2.create_random(self.road))
                if count_aggressive < 3:
                    self.road.vehicles.append(vehicles_type3.create_random(self.road))
                    
            else:
                self.road.vehicles.append(vehicles_type1.create_random(self.road))
        
        print("number of aggressive vehicles ",count_aggressive)

        # create an empty list and then insert randomly
        

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        action_reward = {0: self.LANE_CHANGE_REWARD, 1: 0, 2: self.LANE_CHANGE_REWARD, 3: 0, 4: 0}
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        state_reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * self.vehicle.target_lane_index[2] / (len(neighbours) - 1) \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
        return utils.remap(action_reward[action] + state_reward,
                           [self.config["collision_reward"], self.HIGH_VELOCITY_REWARD+self.RIGHT_LANE_REWARD],
                           [0, 1])

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _cost(self, action):
        """
            The cost signal is the occurrence of collision
        """
        return float(self.vehicle.crashed)


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)
