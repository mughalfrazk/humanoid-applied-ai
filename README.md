# Initial Draft for Applied Artificial Intelligence Final Assessment

---

### **- What problem are you solving?**     
The problem I am trying to solve is making a Robot learn how to walk on its own and for that I am using Reinforcement Learning. 

<img src="https://www.gymlibrary.dev/_images/humanoid.gif" width=300>

The environment being used for this problem is Humanoid-v4 which is part the Mujoco environments project og Gymnasium. It's a 3D bipedal robot which is designed to simulate humans, it has a torso with a pair of legs and arms. The legs and arms both have two links each.   
The goal of the environment is to walk as fast as possible without falling over.    

### **- What techniques/algorithms from AI are you planning to implement?**

I am using Proximal Policy Optimization as my Reinforcement Learning algorithm. It is an on-policy model-free algorithm which does not require a lot of hyperparameter tuning for a better performance.        
Although PPO is a policy gradient method which are usually less sample efficient than the Q-Learning Methods, PPO was essentially introduced to eliminate the flaws of TRPO (Trust Region Policy Optimization). It provides features like ease of implementation, better sample-efficiency and ease of tuning.

<figure>
	<img width="501" alt="Screenshot 2024-04-16 at 10 46 17 AM" src="https://github.com/mughalfrazk/RL-gymasium/assets/70030525/9e51f8d9-472a-4de3-b945-da1d130b3de6">
	<br />
	<figcaption>PPO Objective</figcaption>
</figure>

Here E^t is the expectation operator which means we are computing this over batches of trajectories, A^t is the advantage function which is the difference between the Discounted sum of the reward and value function so negative advantage reduces the action’s future probability and positive advantage increases action’s future probability. An epsilon here ε positive here is a hyperparameter which is usually 0.2.   
Basically the main idea is that after an update, the new policy should not be far from the old policy and for that, PPO uses clipping to avoid too large an update.

<img width="1014" alt="Screenshot 2024-04-16 at 10 44 47 AM" src="https://github.com/mughalfrazk/RL-gymasium/assets/70030525/321c6b21-4447-4d00-80c9-a0eddc9330f8">

<img width="1237" alt="Screenshot 2024-04-16 at 10 45 26 AM" src="https://github.com/mughalfrazk/RL-gymasium/assets/70030525/7d8f834f-ba80-414c-873d-63f6113d45ac">


Here I have trained Proximal Policy Optimization for **more than 1 million timesteps** with different set of hyperparameters such as value for **_gamma_** (Discount Factor), **_ent_coef_** (Entropy Coefficient) and **_clip_range_** (Clipping parameter) in parallel and not so surprisingly the minimal tweaking in hyperparameters provides us with the best result which is after all one of the key benefits of PPO that is provides us the best result without too much hyperparameters tuning involved.

### **- Why that specific algorithm/technique?**

The environment I am working on is a model-free environment and for that I short listed a bunch of algorithms that work well with model-free environments. i.e.
- Advantage Actor Critic (A2C)
- Deep Deterministic Policy Gradient (DDPG)
- Proximal Policy Optimization (PPO)
- Soft Actor Critic (SAC)
- Twin Delayed DDPG (TD3)

And after training all of them in parallel for **more than 3.5 million timesteps** against the Humanoid environment I noticed huge differences in the performances and rewards of their output.

<figure>
	<img width="1670" alt="Screenshot 2024-04-15 at 7 15 44 PM" src="https://github.com/mughalfrazk/RL-gymasium/assets/70030525/0a2708d5-6f34-477c-ac79-1c6d8a75988a">
	<figcaption>Mean Episode Length</figcaption>
</figure>


<figure>
	<img width="1677" alt="Screenshot 2024-04-15 at 2 58 59 PM" src="https://github.com/mughalfrazk/RL-gymasium/assets/70030525/4092daf7-a1dc-439c-ab51-041ac6e143a7">
    	<figcaption>Mean Episode Reward</figcaption>
</figure>


Here we can see that SAC and PPO are the only 2 with some positive result. Although PPO has decreased the number of rewards significantly at a certain point, it is still learning slowly and it is the fastest of the algorithms compared. So I chose PPO to move forward for my problem and dropped others.

### **- Describe your dataset, when it was made, who is the author, how trustable the data is, and what preprocessing you will need to do.**

I am using the [Humanoid-v4](https://www.gymlibrary.dev/environments/mujoco/humanoid/) environment which is part of the Mujoco project of OpenAI gym toolkit which was introduced around 2016.
In 2021 DeepMinds acquired Gym and all the development was moved from Gym to [Gymnasium](https://www.gymlibrary.dev/) which is maintained by Farama Foundation.

Mujoco is a very renowned and open-source dataset for Reinforcement Learning tasks, it is very trustable, efficient and doesn't need a lot of preprocessing it can pretty much be used straight out of the box.

### **- What language/ libraries are you planning to use for the implementation and the planned set-up.**

	Language: Python 
	Libraries:
    - Stable-basline3: for RL algorithm implementation
    - Tensorboaord: For graph visualization

## How to run

