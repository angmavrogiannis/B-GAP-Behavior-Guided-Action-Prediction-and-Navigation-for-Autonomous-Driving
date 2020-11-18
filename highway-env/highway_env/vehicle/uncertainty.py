import copy

import numpy as np

from highway_env import utils
from highway_env.interval import polytope, vector_interval_section, integrator_interval, \
    interval_negative_part, intervals_diff, intervals_product, LPV
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.vehicle.control import MDPVehicle


class IntervalVehicle(LinearVehicle):
    """
        Estimator for the interval-membership of a LinearVehicle under parameter uncertainty.

        The model trajectory is stored in a model_vehicle, and the lower and upper bounds of the states are stored
        in a min_vehicle and max_vehicle. Note that these vehicles do not follow a proper Vehicle dynamics, and
        are only used for storage of the bounds.
    """
    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None,
                 theta_a_i=None,
                 theta_b_i=None):
        """
        :param theta_a_i: The interval of possible acceleration parameters
        :param theta_b_i: The interval of possible steering parameters
        """
        super(IntervalVehicle, self).__init__(road,
                                              position,
                                              heading,
                                              velocity,
                                              target_lane_index,
                                              target_velocity,
                                              route,
                                              enable_lane_change,
                                              timer)
        self.theta_a_i = theta_a_i if theta_a_i is not None else LinearVehicle.ACCELERATION_RANGE
        self.theta_b_i = theta_b_i if theta_b_i is not None else LinearVehicle.STEERING_RANGE

        self.interval = VehicleInterval(self)
        self.trajectory = []
        self.interval_trajectory = []
        self.longitudinal_lpv, self.lateral_lpv = None, None

    @classmethod
    def create_from(cls, vehicle):
        v = cls(vehicle.road,
                vehicle.position,
                heading=vehicle.heading,
                velocity=vehicle.velocity,
                target_lane_index=getattr(vehicle, 'target_lane_index', None),
                target_velocity=getattr(vehicle, 'target_velocity', None),
                route=getattr(vehicle, 'route', None),
                timer=getattr(vehicle, 'timer', None),
                theta_a_i=getattr(vehicle, 'theta_a_i', None),
                theta_b_i=getattr(vehicle, 'theta_b_i', None))
        return v

    def step(self, dt):
        self.store_trajectories()
        if self.crashed:
            self.interval = VehicleInterval(self)
        else:
            # self.observer_step(dt)
            # self.partial_observer_step(dt)
            self.predictor_step(dt)
        super(IntervalVehicle, self).step(dt)

    def observer_step(self, dt):
        """
            Step the interval observer dynamics
        :param dt: timestep [s]
        """
        # Input state intervals
        position_i = self.interval.position
        v_i = self.interval.velocity
        psi_i = self.interval.heading

        # Features interval
        front_interval = self.get_front_interval()

        # Acceleration features
        phi_a_i = np.zeros((2, 3))
        phi_a_i[:, 0] = [0, 0]
        if front_interval:
            phi_a_i[:, 1] = interval_negative_part(
                intervals_diff(front_interval.velocity, v_i))
            # Lane distance interval
            lane_psi = self.lane.heading_at(self.lane.local_coordinates(self.position)[0])
            lane_direction = [np.cos(lane_psi), np.sin(lane_psi)]
            diff_i = intervals_diff(front_interval.position, position_i)
            d_i = vector_interval_section(diff_i, lane_direction)

            d_safe_i = self.DISTANCE_WANTED + self.LENGTH + self.TIME_WANTED * v_i
            phi_a_i[:, 2] = interval_negative_part(intervals_diff(d_i, d_safe_i))

        # Steering features
        phi_b_i = None
        lanes = self.get_followed_lanes()
        for lane_index in lanes:
            lane = self.road.network.get_lane(lane_index)
            longitudinal_pursuit = lane.local_coordinates(self.position)[0] + self.velocity * self.PURSUIT_TAU
            lane_psi = lane.heading_at(longitudinal_pursuit)
            position_corners = [[position_i[0, 0], position_i[0, 1]],
                                [position_i[0, 0], position_i[1, 1]],
                                [position_i[1, 0], position_i[0, 1]],
                                [position_i[1, 0], position_i[1, 1]]]
            corners_lateral = [-lane.local_coordinates(c)[1] for c in position_corners]
            lateral_i = np.array([min(corners_lateral), max(corners_lateral)])
            i_v_i = 1/np.flip(v_i, 0)
            phi_b_i_lane = np.transpose(np.array([
                [0, 0],
                intervals_product(lateral_i, i_v_i)]))
            # Union of candidate feature intervals
            if phi_b_i is None:
                phi_b_i = phi_b_i_lane
            else:
                phi_b_i[0] = np.minimum(phi_b_i[0], phi_b_i_lane[0])
                phi_b_i[1] = np.maximum(phi_b_i[1], phi_b_i_lane[1])

        # Commands interval
        a_i = intervals_product(self.theta_a_i, phi_a_i)
        b_i = intervals_product(self.theta_b_i, phi_b_i)

        # Velocities interval
        keep_stability = False
        if keep_stability:
            dv_i = integrator_interval(v_i - self.target_velocity, self.theta_a_i[:, 0])
        else:
            dv_i = intervals_product(self.theta_a_i[:, 0], self.target_velocity - np.flip(v_i, 0))
        dv_i += a_i
        dv_i = np.clip(dv_i, -self.ACC_MAX, self.ACC_MAX)
        if keep_stability:
            delta_psi = list(map(utils.wrap_to_pi, psi_i - lane_psi))
            d_psi_i = integrator_interval(delta_psi, self.theta_b_i[:, 0])
        else:
            d_psi_i = intervals_product(self.theta_b_i[:, 0], lane_psi - np.flip(psi_i, 0))
        d_psi_i += b_i

        # Position interval
        cos_i = [-1 if psi_i[0] <= np.pi <= psi_i[1] else min(map(np.cos, psi_i)),
                 1 if psi_i[0] <= 0 <= psi_i[1] else max(map(np.cos, psi_i))]
        sin_i = [-1 if psi_i[0] <= -np.pi/2 <= psi_i[1] else min(map(np.sin, psi_i)),
                 1 if psi_i[0] <= np.pi/2 <= psi_i[1] else max(map(np.sin, psi_i))]
        dx_i = intervals_product(v_i, cos_i)
        dy_i = intervals_product(v_i, sin_i)

        # Interval dynamics integration
        self.interval.velocity += dv_i * dt
        self.interval.heading += d_psi_i * dt
        self.interval.position[:, 0] += dx_i * dt
        self.interval.position[:, 1] += dy_i * dt

    def predictor_step(self, dt):
        """
            Step the interval predictor dynamics
        :param dt: timestep [s]
        """
        # Create longitudinal and lateral LPVs
        self.predictor_init()

        # Update lateral LPV center to track target lane
        for i, lane_index in enumerate(self.get_followed_lanes(squeeze=False)):
            lane = self.road.network.get_lane(lane_index)
            center = np.array([lane.position(0, 0)[1], 0])
            if (center != self.lateral_lpv[i].center).any():
                self.lateral_lpv[i].x_i_t[0, :] -= self.lateral_lpv[i].change_coordinates(center, offset=False) - \
                    self.lateral_lpv[i].change_coordinates(self.lateral_lpv[i].center, offset=False)
                self.lateral_lpv[i].x_i_t[1, :] -= self.lateral_lpv[i].change_coordinates(center, offset=False) - \
                    self.lateral_lpv[i].change_coordinates(self.lateral_lpv[i].center, offset=False)
                self.lateral_lpv[i].center = center

        # Step
        self.longitudinal_lpv.step(dt)
        for lpv in self.lateral_lpv:
            lpv.step(dt)

        # Backward coordinates change
        x_i_long = self.longitudinal_lpv.change_coordinates(self.longitudinal_lpv.x_i_t, back=True, interval=True)
        x_i_lat = self.lateral_lpv[0].change_coordinates(self.lateral_lpv[0].x_i_t, back=True, interval=True)
        for lpv in self.lateral_lpv[1:]:
            x_i_lat_lane = lpv.change_coordinates(lpv.x_i_t, back=True, interval=True)
            x_i_lat[0] = np.minimum(x_i_lat[0], x_i_lat_lane[0])
            x_i_lat[1] = np.maximum(x_i_lat[1], x_i_lat_lane[1])

        self.interval.position[:, 0] = x_i_long[:, 0]
        self.interval.position[:, 1] = x_i_lat[:, 0]
        self.interval.velocity = x_i_long[:, 2]
        self.interval.heading = x_i_lat[:, 1]

    def predictor_init(self):
        """
            Initialize the LPV models used for interval prediction
        """
        position_i = self.interval.position
        v_i = self.interval.velocity
        psi_i = self.interval.heading

        # Longitudinal predictor
        if not self.longitudinal_lpv:
            front_interval = self.get_front_interval()

            # Parameters interval
            params_i = self.theta_a_i.copy()

            # Disable velocity control
            if not front_interval or v_i.mean() < front_interval.velocity.mean():
                params_i[:, 1] = 0

            # TODO: for now, we remove Kx from parameters
            Kx = params_i.mean(axis=0)[2]
            params_i = params_i[:, :-1]
            # Disable position control
            if not front_interval or position_i[:, 0].mean() < front_interval.position[:, 0].mean():
                Kx = 0

            # Matrix polytope
            a_theta = lambda params: np.array([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-Kx, Kx, -params[0] - params[1] - Kx * self.TIME_WANTED, params[1]],
                [0, 0, 0, -params[0]]
            ])
            a0, da = polytope(a_theta, params_i)

            # LPV specification
            f_pos = front_interval.position[0, 0] if front_interval else 0
            f_vel = front_interval.velocity[0] if front_interval else 0
            x0 = [position_i[0, 0], f_pos, v_i[0], f_vel]
            center = [-self.DISTANCE_WANTED - self.target_velocity * self.TIME_WANTED,
                      0,
                      self.target_velocity,
                      self.target_velocity]
            d = [self.target_velocity, self.target_velocity, 0, 0]
            self.longitudinal_lpv = LPV(x0, a0, da, d, center)

            # Lateral predictor
            if not self.lateral_lpv:
                self.lateral_lpv = []
                for lane_index in self.get_followed_lanes(squeeze=False):
                    # Parameters interval
                    params_i = self.theta_b_i.copy()

                    # Matrix polytope
                    a_theta = lambda params: np.array([
                        [0, 1],
                        [-params[1], -params[0]]
                    ])
                    a0, da = polytope(a_theta, params_i)

                    # LPV specification
                    x0 = [position_i[0, 1], psi_i[0]]
                    lane = self.road.network.get_lane(lane_index)
                    center = [lane.position(0, 0)[1], 0]
                    d = [0, 0]
                    self.lateral_lpv.append(LPV(x0, a0, da, d, center))

    def get_front_interval(self):
        # TODO: For now, we assume the front vehicle follows the models' front vehicle
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if front_vehicle:
            if isinstance(front_vehicle, IntervalVehicle):
                # Use interval from the observer estimate of the front vehicle
                front_interval = front_vehicle.interval
            else:
                # The front vehicle trajectory interval is not being estimated, so it should be considered as certain.
                # We use a new observer created from that current vehicle state, which will have full certainty.
                front_interval = IntervalVehicle.create_from(front_vehicle).interval
        else:
            front_interval = None
        return front_interval

    def get_followed_lanes(self, lane_change_model="model", squeeze=True):
        """
            Get the list of lanes that could be followed by this vehicle.
        :param lane_change_model: - model: assume that the vehicle will follow the lane of its model behaviour.
                                  - all: assume that any lane change decision is possible at any timestep
                                  - right: assume that a right lane change decision is possible at any timestep
        :param squeeze: if True, remove duplicate lanes (at boundaries of the road)
        :return: the list of followed lane indexes
        """
        if lane_change_model == "model":
            lanes = [self.target_lane_index]
        elif lane_change_model == "all":
            lanes = self.road.network.side_lanes(self.target_lane_index) + [self.target_lane_index]
        elif lane_change_model == "right":
            lanes = [self.target_lane_index]
            _from, _to, _id = self.target_lane_index
            if _id < len(self.road.network.graph[_from][_to]) - 1 \
                    and self.road.network.get_lane((_from, _to, _id + 1)).is_reachable_from(self.position):
                lanes += [(_from, _to, _id + 1)]
            elif not squeeze:
                lanes += [self.target_lane_index]  # Right lane is also current lane
        return lanes

    def partial_observer_step(self, dt, alpha=0):
        """
            Step the boundary parts of the current state interval

            1. Split x_i(t) into two upper and lower intervals x_i_-(t) and x_i_+(t)
            2. Propagate their observer dynamics x_i_-(t+dt) and x_i_+(t+dt)
            3. Merge the resulting intervals together to x_i(t+dt).
        :param dt: timestep [s]
        :param alpha: ratio of the full interval that defines the boundaries
        """
        # 1. Split x_i(t) into two upper and lower intervals x_i_-(t) and x_i_+(t)
        o = self.interval
        v_minus = IntervalVehicle.create_from(self)
        v_minus.interval = copy.deepcopy(self.interval)
        v_minus.interval.position[1, :] = (1 - alpha) * o.position[0, :] + alpha * o.position[1, :]
        v_minus.interval.velocity[1] = (1 - alpha) * o.velocity[0] + alpha * o.velocity[1]
        v_minus.interval.heading[1] = (1 - alpha) * o.heading[0] + alpha * o.heading[1]
        v_plus = IntervalVehicle.create_from(self)
        v_plus.interval = copy.deepcopy(self.interval)
        v_plus.interval.position[0, :] = alpha * o.position[0, :] + (1 - alpha) * o.position[1, :]
        v_plus.interval.velocity[0] = alpha * o.velocity[0] + (1 - alpha) * o.velocity[1]
        v_plus.interval.heading[0] = alpha * o.heading[0] + (1 - alpha) * o.heading[1]
        # 2. Propagate their observer dynamics x_i_-(t+dt) and x_i_+(t+dt)
        v_minus.road = copy.copy(v_minus.road)
        v_minus.road.vehicles = [v if v is not self else v_minus for v in v_minus.road.vehicles]
        v_plus.road = copy.copy(v_plus.road)
        v_plus.road.vehicles = [v if v is not self else v_plus for v in v_plus.road.vehicles]
        v_minus.observer_step(dt)
        v_plus.observer_step(dt)
        # 3. Merge the resulting intervals together to x_i(t+dt).
        self.interval.position = np.array([v_minus.interval.position[0], v_plus.interval.position[1]])
        self.interval.velocity = np.array([v_minus.interval.velocity[0], v_plus.interval.velocity[1]])
        self.interval.heading = np.array([v_minus.interval.heading[0], v_plus.interval.heading[1]])

    def store_trajectories(self):
        """
            Store the current model, min and max states to a trajectory list
        """
        self.trajectory.append(LinearVehicle.create_from(self))
        self.interval_trajectory.append(copy.deepcopy(self.interval))

    def check_collision(self, other):
        """
            For robust planning, we assume that MDPVehicles collide with the uncertainty set of an IntervalVehicle,
            which corresponds to worst-case outcome.

        :param other: the other vehicle
        """
        if not isinstance(other, MDPVehicle):
            return super(IntervalVehicle, self).check_collision(other)

        if not self.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast rectangular pre-check
        if not utils.point_in_rectangle(other.position,
                                        self.interval.position[0] - self.LENGTH,
                                        self.interval.position[1] + self.LENGTH):
            return

        # Projection of other vehicle to uncertainty rectangle. This is the possible position of this vehicle which is
        # the most likely to collide with other vehicle
        projection = np.minimum(np.maximum(other.position, self.interval.position[0]),
                                self.interval.position[1])
        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((projection, self.LENGTH, self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min(self.velocity, other.velocity)
            self.crashed = other.crashed = True


class VehicleInterval(object):
    def __init__(self, vehicle):
        self.position = np.array([vehicle.position, vehicle.position], dtype=float)
        self.velocity = np.array([vehicle.velocity, vehicle.velocity], dtype=float)
        self.heading = np.array([vehicle.heading, vehicle.heading], dtype=float)
