# Triangular Number-Based Objective Function for Trajectory Planning of Wing-in-Ground Crafts' Fast Altitude Change and Convergence via Deep Reinforcement Learning


* **Description on directories:**  Directories named with "multiobj_w_" are cases of conventional multi-objective functions, whose magnitude of weight are labeled at the end of the name. Directories named with "proposed_obj_coeff_" represent cases of the proposed objective function, whose magnitude of coefficient is written at the end of the name. At each directory, code for each case is placed under the sub directory named 'jietijihua'. Each case are examined with eight different seeds, so there are eight sub-sub directories named with "s_", where main code files locate.

* **Main code file:** At the each sub-sub directory, there are six files for the training, which are as following:
1. *Main.py*. It's main function and the training environment and reward function are defined. 

2. *replay.py*. It's classes of replay buffer during training.

3. *net.py*. It's classes of neural network for the algorithm of Proximal Policy Optimization (PPO).

4. *agent.py*. It's classes of agents for PPO.

5. *aeropy_fix.py*. It's functions for aerodynamics and kinectics of the wing-in-ground craft (WIG) during training.

* **Prerequisites for running codes:**  Pytorch is needed to be installed in advance. In order to run the code at each sub-sub directory, the file 'datacollection.npy', 'guiyi_0.6to0.3_case1_0bu.pth', "actor_0.6to0.3_case1_0bu.pth", "critic_0.6to0.3_case1_0bu.pth" and "critic_target_0.6to0.3_case1_0bu.pth" should be downloaded and then placed in the root directory. Since the website of Github limits the size of files, the necessary file and main results are placed in the same directory and they are upload in onedrive. The link is [[[https://maildluteducn-my.sharepoint.com/:f:/g/personal/huhuan2019_mail_dlut_edu_cn/Eto68Yzc8KlKox0pZQeHn9wBiDSPEWZicshKlKtGgfqWRA?e=OUg4KN.](https://aluhiteducn-my.sharepoint.com/:f:/g/personal/huhuan2016_alu_hit_edu_cn/Em0qgCL3WpRAhRqomPTm--cBKIODyOAqEgUV-eib3FYbGQ?e=aHzlMD)](https://aluhiteducn-my.sharepoint.com/:f:/g/personal/huhuan2016_alu_hit_edu_cn/Em0qgCL3WpRAhRqomPTm--cBKIODyOAqEgUV-eib3FYbGQ?e=jRZuvQ)](https://aluhiteducn-my.sharepoint.com/:f:/g/personal/huhuan2016_alu_hit_edu_cn/Em0qgCL3WpRAhRqomPTm--cBKIODyOAqEgUV-eib3FYbGQ?e=qTvoay). The file 'datacollection.npy' is used for the calculation of aerodynamics for the WIG, and other files is needed for the code to implement deep reinforcement learning.

* **Main results after running codes:** At the each sub directory of 'jietijihua' (in the onedrive or be downloaded), there are eight picture which are the outputs after the training, the outcome of the whole eight sub cases are labled with "maximum.png". There are also other files for your reference.

1. *actor.pth*. It's neural network of the Actor in deep reinforcement learning.

2. *critic.pth*. It's neural network of the Critic in deep reinforcement learning.

3. *critic_target.pth*. It's neural network of the Target of Actor in deep reinforcement learning.

4. *guiyi.pth.npy*. It's memory buffer which is used for normalization during training.

6. *reward_curve.png*. It's reward curve during training.

7. *thebest.npy*. It's the data of trajectory which is rewarded mostly after training.

8. *theone.png*. It shows the trajectory which is rewarded mostly after training.

9. *bestlog.npy*. It records every breakthrough of the reward during training.
