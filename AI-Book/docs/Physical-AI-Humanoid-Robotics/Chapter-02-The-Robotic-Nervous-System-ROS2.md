---
sidebar_position: 2
---

# Chapter 2: The Robotic Nervous System - ROS 2

## Understanding ROS 2 Architecture

Robot Operating System 2 (ROS 2) serves as the **nervous system** of your humanoid robot. Just as our nervous system coordinates sensory input, processing, and motor output, ROS 2 provides the middleware infrastructure that connects all components of your robot system.

### Core Concepts of ROS 2

ROS 2 is built around several key architectural concepts:

#### Nodes
Nodes are individual processes that perform computation. Think of nodes as specialized organs in the robotic body:
- Each node performs a specific function (navigation, perception, control)
- Nodes can be written in different languages (C++, Python, etc.)
- Nodes communicate with each other through topics, services, and actions

#### Topics
Topics enable asynchronous, publisher-subscriber communication:
- Publishers send data streams to topics
- Subscribers receive data from topics
- Multiple publishers and subscribers can connect to the same topic
- Ideal for sensor data, robot state, and other continuous streams

#### Services
Services provide synchronous request-response communication:
- Client sends a request and waits for a response
- Useful for operations that need confirmation or return specific results
- Similar to REST API calls in web applications

#### Actions
Actions handle long-running tasks with feedback:
- Support goals, feedback, and results
- Perfect for navigation, manipulation, and other time-consuming operations
- Allow clients to monitor progress and cancel operations

### Installing ROS 2

For humanoid robotics, we recommend **Ubuntu 22.04 LTS** with **ROS 2 Humble Hawksbill**:

```bash
# Add ROS 2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
```

### Working with ROS 2 Workspaces

ROS 2 organizes code in workspaces:

```bash
# Create a workspace
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws

# Source ROS 2 installation
source /opt/ros/humble/setup.bash

# Build the workspace
colcon build --packages-select humanoid_bringup
```

### Python-Based ROS 2 Development with rclpy

Most humanoid robotics development uses Python for rapid prototyping:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Create subscriber for joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # Create publisher for velocity commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def joint_callback(self, msg):
        # Process joint state information
        self.get_logger().info(f'Received {len(msg.position)} joints')
```

### URDF Modeling for Humanoid Robots

Unified Robot Description Format (URDF) describes your robot's physical properties:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head link -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

### Best Practices for ROS 2 Development

1. **Modularity**: Keep nodes focused on single responsibilities
2. **Naming Conventions**: Use consistent naming for topics and services
3. **Logging**: Implement comprehensive logging for debugging
4. **Configuration**: Separate configuration from code using parameters
5. **Testing**: Develop unit and integration tests for each node

## Summary

ROS 2 provides the essential middleware infrastructure for humanoid robotics. Understanding nodes, topics, services, and actions is crucial for building coordinated robot behaviors. The next chapter will explore how to create digital twins of your robots using simulation environments.