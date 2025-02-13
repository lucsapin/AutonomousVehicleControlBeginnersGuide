"""
extended_kalman_filter_localization.py

Author: Shisato Yano
"""

# import path setting
import sys
from pathlib import Path
import numpy as np

abs_dir_path = str(Path(__file__).absolute().parent)
relative_path = "/../../../components/"

sys.path.append(abs_dir_path + relative_path + "visualization")
sys.path.append(abs_dir_path + relative_path + "state")
sys.path.append(abs_dir_path + relative_path + "vehicle")
sys.path.append(abs_dir_path + relative_path + "course/cubic_spline_course")
sys.path.append(abs_dir_path + relative_path + "control/pure_pursuit")
sys.path.append(abs_dir_path + relative_path + "sensors")
sys.path.append(abs_dir_path + relative_path + "sensors/gnss")
sys.path.append(abs_dir_path + relative_path + "sensors/lidar")
sys.path.append(abs_dir_path + relative_path + "localization/kalman_filter")
sys.path.append(abs_dir_path + relative_path + "obstacle")


# import component modules
from global_xy_visualizer import GlobalXYVisualizer
from min_max import MinMax
from time_parameters import TimeParameters
from vehicle_specification import VehicleSpecification
from state import State
from four_wheels_vehicle import FourWheelsVehicle
from cubic_spline_course import CubicSplineCourse
from pure_pursuit_controller import PurePursuitController
from sensors import Sensors
from sensor_parameters import SensorParameters
from gnss import Gnss
from omni_directional_lidar import OmniDirectionalLidar
from extended_kalman_filter_localizer import ExtendedKalmanFilterLocalizer
from obstacle import Obstacle
from obstacle_list import ObstacleList


# flag to show plot figure
# when executed as unit test, this flag is set as false
show_plot = True


def main():
    """
    Main process function
    """
    
    # set simulation parameters
    x_lim, y_lim = MinMax(-5, 55), MinMax(-20, 25)
    vis = GlobalXYVisualizer(x_lim, y_lim, TimeParameters(span_sec=30))

    # create course data instance
    course = CubicSplineCourse([0.0, 10.0, 25, 40, 50],
                               [0.0, 4, -12, 20, -13],
                               20)
    vis.add_object(course)

    # create vehicle's spec instance
    spec = VehicleSpecification(area_size=20.0)
    
    # create vehicle's state instance
    state = State(color='b')
    
    # create controller instance
    pure_pursuit = PurePursuitController(spec, course, color='m')

    # create obstacles
    obstacle_list = ObstacleList()
    obstacle_list.add_obstacle(Obstacle(State(x_m=10.0, y_m=15.0), length_m=1, width_m=8))
    obstacle_list.add_obstacle(Obstacle(State(x_m=40.0, y_m=0.0), length_m=2, width_m=10))
    obstacle_list.add_obstacle(Obstacle(State(x_m=10.0, y_m=-10.0, yaw_rad=np.rad2deg(45)), length_m=5, width_m=5))
    obstacle_list.add_obstacle(Obstacle(State(x_m=30.0, y_m=15.0, yaw_rad=np.rad2deg(10)), length_m=5, width_m=2))
    obstacle_list.add_obstacle(Obstacle(State(x_m=50.0, y_m=15.0, yaw_rad=np.rad2deg(15)), length_m=5, width_m=2))
    obstacle_list.add_obstacle(Obstacle(State(x_m=25.0, y_m=0.0), length_m=2, width_m=2))
    obstacle_list.add_obstacle(Obstacle(State(x_m=35.0, y_m=-15.0), length_m=7, width_m=2))
    vis.add_object(obstacle_list)

    # create vehicle instance
    # set state, spec, controller, sensors and localizer instances as arguments
    gnss = Gnss(x_noise_std=1.0, y_noise_std=1.0)
    # long range lidar
    lidar1_param = SensorParameters(lon_m=spec.wheel_base_m/2, max_m=15, dist_std_rate=0.05)
    lidar1 = OmniDirectionalLidar(obstacle_list, lidar1_param)
    # short range lidar
    lidar2_param = SensorParameters(lon_m=spec.wheel_base_m/2, max_m=15, dist_std_rate=0.05)
    lidar2 = OmniDirectionalLidar(obstacle_list, lidar2_param)
    
    sensors_list=Sensors(lidar1, gnss)
    ekf = ExtendedKalmanFilterLocalizer()
    # vehicle = FourWheelsVehicle(state, spec, controller=pure_pursuit, sensors=gnss, localizer=ekf)
    vehicle = FourWheelsVehicle(state, spec, controller=pure_pursuit, sensors=sensors_list, localizer=ekf)
    vis.add_object(vehicle)

    # plot figure is not shown when executed as unit test
    if not show_plot: vis.not_show_plot()

    # show plot figure
    vis.draw()


# execute main process
if __name__ == "__main__":
    main()
