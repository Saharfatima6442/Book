import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Sidebar for the AI book
  aiBookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['Physical-AI-Humanoid-Robotics/Introduction'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'Physical-AI-Humanoid-Robotics/Chapter-01-Introduction-to-Physical-AI',
        'Physical-AI-Humanoid-Robotics/Chapter-02-The-Robotic-Nervous-System-ROS2',
        'Physical-AI-Humanoid-Robotics/Chapter-03-The-Digital-Twin-Gazebo-Unity',
        'Physical-AI-Humanoid-Robotics/Chapter-04-The-AI-Robot-Brain-NVIDIA-Isaac',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Perception and Interaction',
      items: [
        'Physical-AI-Humanoid-Robotics/Chapter-05-Vision-Language-Action-VLA',
        'Physical-AI-Humanoid-Robotics/Chapter-06-Humanoid-Locomotion-Interaction',
        'Physical-AI-Humanoid-Robotics/Chapter-07-Conversational-Multimodal-Robotics',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Implementation and Deployment',
      items: [
        'Physical-AI-Humanoid-Robotics/Chapter-08-Edge-AI-Deployment',
        'Physical-AI-Humanoid-Robotics/Chapter-09-Advanced-Perception-Navigation',
        'Physical-AI-Humanoid-Robotics/Chapter-10-AI-Planning-Decision-Making',
        'Physical-AI-Humanoid-Robotics/Chapter-13-Implementing-Physical-AI-Assistant',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Safety and Ethics',
      items: [
        'Physical-AI-Humanoid-Robotics/Chapter-11-Safety-Ethics',
        'Physical-AI-Humanoid-Robotics/Chapter-12-Capstone-Project',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Reference Materials',
      items: [
        'Physical-AI-Humanoid-Robotics/Conclusion',
        'Physical-AI-Humanoid-Robotics/Thanking-Notes',
        'Physical-AI-Humanoid-Robotics/Technical-Jargon-Definitions',
      ],
      collapsed: false,
    },
  ],
};

export default sidebars;
