---
sidebar_position: 6
---

# Chapter 6: Humanoid Locomotion and Interaction

## Understanding Humanoid Locomotion

Humanoid locomotion represents one of the most challenging problems in robotics. Unlike wheeled or tracked robots, humanoid robots must maintain balance while moving in dynamic, unpredictable environments. This requires sophisticated control systems that can handle the inherent instability of bipedal walking.

### The Challenge of Bipedal Walking

Humanoid locomotion involves several complex challenges:

- **Dynamic Balance**: Maintaining stability while moving
- **Terrain Adaptation**: Navigating various surfaces and obstacles
- **Energy Efficiency**: Minimizing power consumption during movement
- **Robustness**: Handling unexpected disturbances gracefully
- **Smooth Transitions**: Seamlessly switching between different gaits

### Gait Types for Humanoid Robots

Humanoid robots employ various gait patterns depending on their requirements:

#### Static Walking
- **Characteristics**: Maintains static stability at all times
- **Advantages**: High stability, simple control
- **Disadvantages**: Slow, energy inefficient
- **Use Cases**: Precise manipulation tasks, unstable terrain

#### Dynamic Walking
- **Characteristics**: Allows momentary loss of balance
- **Advantages**: Faster, more human-like
- **Disadvantages**: More complex control, higher risk
- **Use Cases**: Normal navigation, human-like movement

#### Running and Jumping
- **Characteristics**: Both feet leave ground simultaneously
- **Advantages**: Very fast movement
- **Disadvantages**: High control complexity, energy consumption
- **Use Cases**: Specialized applications, research

## Balance Control Systems

### Zero Moment Point (ZMP) Control

The Zero Moment Point is a fundamental concept in humanoid balance control:

```python
import numpy as np

class ZMPController:
    def __init__(self, robot_mass, gravity=9.81):
        self.mass = robot_mass
        self.gravity = gravity
        self.com_height = 0.8  # Center of mass height in meters

    def calculate_zmp(self, com_pos, com_acc):
        """
        Calculate ZMP based on center of mass position and acceleration
        ZMP = [x, y] where moments around these points are zero
        """
        zmp_x = com_pos[0] - (self.com_height / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (self.com_height / self.gravity) * com_acc[1]

        return np.array([zmp_x, zmp_y, 0.0])

    def check_stability(self, zmp_pos, support_polygon):
        """
        Check if ZMP is within the support polygon
        """
        # Simple check for rectangular support polygon
        min_x, max_x = support_polygon['x_range']
        min_y, max_y = support_polygon['y_range']

        return (min_x <= zmp_pos[0] <= max_x) and (min_y <= zmp_pos[1] <= max_y)
```

### Linear Inverted Pendulum Model (LIPM)

The LIPM simplifies humanoid dynamics for control purposes:

```python
class LinearInvertedPendulumModel:
    def __init__(self, com_height, gravity=9.81):
        self.omega = np.sqrt(gravity / com_height)

    def calculate_com_trajectory(self, zmp_trajectory, initial_com, initial_com_vel):
        """
        Calculate CoM trajectory from ZMP reference
        """
        com_trajectory = []
        dt = 0.01  # Time step

        com_pos = initial_com
        com_vel = initial_com_vel

        for zmp_point in zmp_trajectory:
            # LIPM dynamics: CoM'' = omega^2 * (CoM - ZMP)
            com_acc = self.omega**2 * (com_pos - zmp_point)

            # Integrate to get new position and velocity
            com_vel += com_acc * dt
            com_pos += com_vel * dt

            com_trajectory.append(com_pos.copy())

        return com_trajectory
```

## Walking Pattern Generation

### Preview Control for Walking

Preview control uses future ZMP references to generate stable walking patterns:

```python
class PreviewWalkingController:
    def __init__(self, com_height, preview_time=2.0, dt=0.01):
        self.com_height = com_height
        self.preview_steps = int(preview_time / dt)
        self.dt = dt

        # Pre-calculate preview control gains
        self.omega = np.sqrt(9.81 / com_height)

    def generate_walking_pattern(self, step_locations, step_times):
        """
        Generate CoM and ZMP trajectories for walking
        """
        total_steps = len(step_locations)

        # Initialize trajectories
        com_trajectory = []
        zmp_trajectory = []

        current_com = np.array([0.0, 0.0, self.com_height])
        current_com_vel = np.zeros(3)

        for i, (step_pos, step_time) in enumerate(zip(step_locations, step_times)):
            # Generate ZMP reference for this step
            zmp_ref = self.calculate_support_polygon_center(step_pos, i)

            # Calculate CoM trajectory using preview control
            for t in range(int(step_time / self.dt)):
                zmp_current = zmp_ref  # Simplified - in practice, interpolate

                # LIPM dynamics with preview
                com_acc = self.omega**2 * (current_com[:2] - zmp_current[:2])
                com_acc = np.append(com_acc, 0)  # No vertical acceleration

                # Integrate
                current_com_vel += com_acc * self.dt
                current_com += current_com_vel * self.dt

                com_trajectory.append(current_com.copy())
                zmp_trajectory.append(zmp_current.copy())

        return com_trajectory, zmp_trajectory

    def calculate_support_polygon_center(self, foot_pos, step_number):
        """
        Calculate ZMP reference based on foot placement
        """
        # Simplified: ZMP moves to foot center at appropriate time
        return foot_pos
```

### Footstep Planning

Intelligent footstep planning is crucial for stable locomotion:

```python
import numpy as np
from scipy.spatial import distance

class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2):
        self.step_length = step_length
        self.step_width = step_width

    def plan_footsteps(self, start_pos, goal_pos, terrain_map=None):
        """
        Plan a sequence of footsteps from start to goal
        """
        footsteps = []

        # Calculate direction vector
        direction = goal_pos[:2] - start_pos[:2]
        distance_to_goal = np.linalg.norm(direction)
        unit_direction = direction / distance_to_goal if distance_to_goal > 0 else np.array([1, 0])

        current_pos = start_pos.copy()
        step_count = 0

        while np.linalg.norm(current_pos[:2] - goal_pos[:2]) > self.step_length:
            # Calculate next foot position
            if step_count % 2 == 0:  # Right foot step
                lateral_offset = np.array([-self.step_width/2 * unit_direction[1],
                                          self.step_width/2 * unit_direction[0]])
            else:  # Left foot step
                lateral_offset = np.array([self.step_width/2 * unit_direction[1],
                                          -self.step_width/2 * unit_direction[0]])

            next_pos = current_pos[:2] + unit_direction * self.step_length + lateral_offset
            next_pos = np.append(next_pos, current_pos[2])  # Maintain height

            footsteps.append(next_pos.copy())
            current_pos[:2] = next_pos[:2]
            step_count += 1

            # Check for terrain validity if provided
            if terrain_map and not self.is_valid_foot_placement(next_pos, terrain_map):
                # Adjust placement or replan
                pass

        # Add final step to goal
        final_step = goal_pos.copy()
        final_step[2] = current_pos[2]  # Maintain consistent height
        footsteps.append(final_step)

        return footsteps

    def is_valid_foot_placement(self, position, terrain_map):
        """
        Check if foot placement is valid given terrain constraints
        """
        # Check for obstacles, slopes, unstable surfaces, etc.
        x, y = int(position[0]), int(position[1])
        if 0 <= x < terrain_map.shape[0] and 0 <= y < terrain_map.shape[1]:
            return terrain_map[x, y] == 0  # Assuming 0 means traversable
        return False
```

## Human-Robot Interaction

### Social Navigation

Humanoid robots must navigate in human environments while respecting social norms:

#### Personal Space Management
- **Intimate distance**: 0-45cm (reserved for close relationships)
- **Personal distance**: 45-120cm (for casual conversations)
- **Social distance**: 1.2-3.6m (for formal interactions)
- **Public distance**: 3.6m+ (for public speaking)

#### Socially-Aware Path Planning
```python
class SocialNavigationPlanner:
    def __init__(self):
        self.personal_space_radius = 1.0  # meters
        self.social_distance = 1.5  # meters

    def plan_socially_aware_path(self, start, goal, humans_in_environment):
        """
        Plan path that respects human personal space and social norms
        """
        # Modify traditional path planning to avoid human personal spaces
        modified_cost_map = self.create_social_cost_map(humans_in_environment)

        # Use A* or other path planning algorithm with modified costs
        path = self.a_star_pathfinding(start, goal, modified_cost_map)

        return path

    def create_social_cost_map(self, humans):
        """
        Create a cost map that penalizes proximity to humans
        """
        # Implementation would create a grid with higher costs near humans
        # Cost increases as distance to humans decreases
        pass
```

### Gesture and Expression Systems

Humanoid robots use gestures and expressions to communicate:

#### Upper Body Expressions
- **Arm gestures**: Pointing, waving, indicating directions
- **Head movements**: Nodding, shaking, tilting for attention
- **Facial expressions**: If equipped with expressive face
- **Posture**: Open vs closed body language

#### Example Gesture Controller
```python
class GestureController:
    def __init__(self, robot_interface):
        self.robot = robot_interface

    def execute_greeting_gesture(self):
        """
        Execute a friendly greeting gesture
        """
        # Raise right arm to wave position
        self.robot.move_joint("right_shoulder_pitch", 0.5)
        self.robot.move_joint("right_shoulder_roll", 0.2)
        self.robot.move_joint("right_elbow", -1.0)

        # Wave hand 3 times
        for i in range(3):
            self.robot.move_joint("right_wrist", 0.5 if i % 2 == 0 else -0.5)
            self.robot.sleep(0.5)

        # Return to neutral position
        self.robot.move_to_neutral()

    def execute_pointing_gesture(self, target_position):
        """
        Point to a specific location
        """
        # Calculate required joint angles to point to target
        joint_angles = self.calculate_pointing_angles(target_position)
        self.robot.move_joints(joint_angles)
```

## Manipulation and Grasping

### Grasp Planning for Humanoid Robots

Humanoid robots need sophisticated grasp planning for object manipulation:

#### Grasp Stability Metrics
- **Force closure**: Ability to resist external forces
- **Form closure**: Geometric constraints preventing object motion
- **Quality metrics**: Quantitative measures of grasp stability

#### Grasp Synthesis
```python
class GraspPlanner:
    def __init__(self):
        self.hand_model = self.load_hand_model()

    def plan_grasp(self, object_mesh, grasp_type="power"):
        """
        Plan a stable grasp for the given object
        """
        candidate_grasps = self.generate_candidate_grasps(object_mesh, grasp_type)

        best_grasp = None
        best_score = -float('inf')

        for grasp in candidate_grasps:
            score = self.evaluate_grasp_quality(grasp, object_mesh)
            if score > best_score:
                best_score = score
                best_grasp = grasp

        return best_grasp

    def evaluate_grasp_quality(self, grasp, object_mesh):
        """
        Evaluate the quality of a grasp using force closure analysis
        """
        # Calculate grasp quality metric
        # This would involve checking force closure conditions
        # and considering friction cones at contact points
        pass
```

### Whole-Body Manipulation

Humanoid robots can use their entire body for manipulation tasks:

#### Coordinated Arm-Body Motion
- **Base mobility**: Moving the robot base for better reach
- **Trunk motion**: Using torso for extended workspace
- **Balance maintenance**: Adjusting posture during manipulation
- **Multi-limb coordination**: Using both arms simultaneously

## Control Architecture for Locomotion

### Hierarchical Control Structure

Humanoid locomotion typically uses multiple control layers:

#### High-Level Planner
- **Trajectory generation**: Long-term path planning
- **Gait selection**: Choosing appropriate walking pattern
- **Obstacle avoidance**: High-level navigation planning

#### Mid-Level Controller
- **Balance control**: Maintaining stability during motion
- **Footstep adjustment**: Real-time foot placement corrections
- **Gait parameter modulation**: Adjusting walking parameters

#### Low-Level Controller
- **Joint control**: Direct motor command generation
- **Sensor feedback**: Processing proprioceptive data
- **Disturbance rejection**: Handling external forces

### Example Control Architecture
```python
class HumanoidController:
    def __init__(self):
        self.high_level_planner = HighLevelPlanner()
        self.mid_level_controller = MidLevelController()
        self.low_level_controller = LowLevelController()

        self.current_state = "standing"
        self.balance_active = True

    def execute_command(self, command):
        """
        Execute high-level command through hierarchical control
        """
        if command.type == "walk_to":
            self.initiate_walking(command.target)
        elif command.type == "grasp_object":
            self.initiate_manipulation(command.object_pose)
        elif command.type == "balance":
            self.activate_balance_control()

    def initiate_walking(self, target):
        """
        Coordinate all control levels for walking
        """
        # Plan high-level trajectory
        trajectory = self.high_level_planner.plan_trajectory(
            self.get_robot_position(), target
        )

        # Generate footstep sequence
        footsteps = self.high_level_planner.plan_footsteps(trajectory)

        # Activate mid-level balance controller
        self.mid_level_controller.set_walking_mode(footsteps)

        # Update low-level controllers
        self.low_level_controller.enable_walking_controllers()

        self.current_state = "walking"
```

## Practical Implementation Considerations

### Hardware-Specific Factors

Different humanoid platforms have unique characteristics:

#### Actuator Limitations
- **Torque limits**: Maximum forces that joints can exert
- **Speed constraints**: Maximum joint velocities
- **Power consumption**: Battery life considerations
- **Heat dissipation**: Managing actuator temperatures

#### Sensor Integration
- **IMU placement**: Optimal locations for balance sensing
- **Camera positioning**: Field of view for perception
- **Force sensors**: Placement for manipulation feedback
- **Tactile sensors**: Skin-like sensing for interaction

### Software Integration with ROS 2

Humanoid locomotion systems integrate with ROS 2 through various message types:

#### Common Message Types
- `sensor_msgs/JointState`: Joint position, velocity, effort
- `geometry_msgs/PoseStamped`: Robot pose information
- `nav_msgs/Path`: Planned navigation paths
- `trajectory_msgs/JointTrajectory`: Desired joint trajectories

## Summary

Humanoid locomotion and interaction represent the integration of complex control systems, mechanical engineering, and human factors. Successful humanoid robots must balance the challenges of bipedal walking with the demands of natural human interaction. The next chapter will explore conversational and multimodal robotics, building on the locomotion and interaction foundations established here.