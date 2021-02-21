---
layout: post
title:  "Playing Flappy Bird with Reinforcement Learning"
tag:
- Reinforcement Learning
- Deep Learning
- Tensorflow
- Games
category: blog
author: franciscosalgado
---

DeepMind's DQN paper released in 2015 is still one of the most fascinating papers I ever read. In this paper an agent learns how to beat Human performance across 49 different Atari games using the same architecture and hyperparameters.

This weekend, I decided to try my hands at Deep Q-Learning by coding a learning agent to play Flappy Bird and see how close it comes to the performance of a Human player (or even outperforms him!). I already had experience with Q-Learning from some University courses so it was quick to get an initial agent playing the game. Nonetheless, Sutton and Barto's RL book [2] as well as the DQN Tutorial from the Pytorch Docs [3] proved super useful to brush up on some Reinforcement Learning details. This was also a chance to try Tensorflow 2.0 

Instead of using a screenshot of the game to encode the state and a CNN for the policy network, as done on the original DQN paper [1], I used the player and pipes positions for the state and a Multilayer Perceptron as the policy Network. This was due to being limited to simulate the game and train the nework on my Macbook (between 30mins and 1h for each combination of hyperparameters).

In the end, I was achieving a consistent score of around 23, which I would say is on par with a Human player. Head over to [my github](https://github.com/d3rezz/flappybird-RL) for the full code and a more detailed description of how it is implemented.

<table align="center" style="border: none; border-collapse: collapse; background-color: #ffffff;">
    <tr style="border: none; background-color: #ffffff;">
        <td align="center" style="border: none;">
            <img src="/assets/post_images/2020-03-22-flappy-bird-RL/flappy.gif" />
        </td>
    </tr>
</table>

I encoded each state (game frame) using 5 values:
- Horizontal distance between the player and the next pipe
- Vertical distance between the player and the lower pipe
- Vertical distance between the player and the upper pipe 
- Distance from the player to the top of the map
- Distance from the player to the base of the map

Initially, I was using simply the first two, together with the player's vertical speed, however it struggled to propagate back the negative rewards from hitting the top tube.

<table align="center" style="border: none; border-collapse: collapse; background-color: #ffffff;">
    <tr style="border: none; background-color: #ffffff;">
        <td align="center" style="border: none;">
            <img src="/assets/post_images/2020-03-22-flappy-bird-RL/state_encoding.png" height="450"/>
        </td>
    </tr>
    <tr style="border: none; background-color: #ffffff;">
        <td style="border: none;" align="center">State encoding</td> 
    </tr>
</table>


I found the algorithm very sensitive to the hyperparameters chosen and tricky to tweak, even though I was logging the values of loss, epsilon (probability of a taking random action), gradient magnitude and episode count to Tensorboard. Using a Replay Memory buffer and a Target Network that is only updated every few episodes definitely helped the stability of the agent.

Overall, this was a fun quick experiment and I now feel more prepared to tackle SOTA Deep RL papers. Next time, I will be upgrading the agent to use raw screenshots of the games and use a CNN for the policy network. Looking back, I wish I had coded in the ability to take a random seed to ensure reproducibility of the experiments, as sometimes the same hyperparameters would converge to very different performances.
I should also investigate which additional metrics I should be keeping track of, in order get more insight on how each parameters affects the learning process of the agent.
Lastly, I was very impressed with how painless it is to use TF 2.0 vs the previous versions of the framework. It integrates beautifully with Pycharm's debugger, which made implementing the Q-Learning algorithm a breeze.


## References

[1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

[2] Sutton, Richard S., and Andrew G. Barto. "Reinforcement learning: An introduction." MIT press, 2018.

[3] Reinforcement Learning (DQN) Tutorial. pytorch.org/tutorials/intermediate/reinforcement_q_learning.html. Accessed 22 Mar. 2020. 

