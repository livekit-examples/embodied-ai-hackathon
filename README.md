# embodied-ai-hackathon

This repo is a submission to the SPC Embodied AI Hackathon. In this project, we demonstrate how a Voice AI Agent can be used to interact with a desktop robot arm on a slider.

The scope of this project includes building:

- An orchestrating voice agent to which an user can give general commands such as "Find me a tool which can help me open this box" and create code as policies to run for the robot.
- A set of policies for picking up and placing objects.
- A classical robotics policy for moving the robot on slider to various position on a desk.

Existing priors used in this project includes:

- LiveKit Agents which is a voice agents orchestration platform, our flagship product.
- LiveKit Portal which is our SDK for teleoperation and data collection on any robot. This is the backbone for orchestration of the entire project. This allows operators or policies to run from different computers in the same network and allow us to easily collect data for our more than complex system without running into configurations and hardware hassle.
- LeSlider which is a simple hardware extension to the SO101 we designed for fun and for desktop use cases like this.
