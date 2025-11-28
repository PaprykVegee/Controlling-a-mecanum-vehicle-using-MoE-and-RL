# Controlling a Mecanum Vehicle Using Mixture of Experts (MoE) and Reinforcement Learning

This project introduces an adaptive control system for a mecanum-wheel robot using a **Mixture of Experts (MoE)** architecture combined with **Reinforcement Learning (RL)**. The goal is to enable reliable autonomous navigation in environments with diverse obstacle types by dynamically selecting the most suitable control policy.

The system operates in a simulated environment built in **Gazebo**, fully integrated with **ROS2** for communication, sensor streaming, and controller deployment.

---

## System Overview

### Reinforcement Learning Experts

The control layer includes **three independent RL expert models**, each trained to handle a specific category of obstacles or navigation challenges:

- **Expert A** – specialized for narrow spaces and corridor-like environments  
- **Expert B** – optimized for dynamic obstacles, such as moving objects  
- **Expert C** – trained to avoid large or irregular obstacles requiring wider maneuvers  

Each expert relies on sensor data provided through ROS2 and outputs velocity commands for the mecanum platform.

---

## Sensor Setup: RGB Camera + 2D LiDAR

The robot uses a minimal but effective sensor suite:

- **RGB camera** providing high-resolution visual input  
- **2D LiDAR** delivering accurate distance and obstacle geometry measurements  

Data from both sensors is streamed as ROS2 topics. This information is used by the experts and by the MoE gating network to select the appropriate controller in real time.

---

## Gating Network

A central **gating network** acts as the decision-making manager within the MoE framework. Its responsibilities include:

- Interpreting the current environment based on camera and LiDAR data  
- Classifying the type of encountered obstacle or scenario  
- Selecting the most suitable RL expert at each timestep  
- Routing the robot’s control to the chosen expert policy  

This modular structure allows for adaptive behavior that no single RL model could provide alone.
