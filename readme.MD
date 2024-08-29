# Enhancing Decision-Making in Energy Management Systems through Action-Independent Dynamics Learning

Incorporating auxiliary objectives into Reinforcement Learning allows agents to acquire additional knowledge, thereby increasing their search for the optimal policy. This article presents the Model-Predictor Proximal Policy Optimization (MP-PPO) algorithm, which merges the concepts of various PPO variants with a Transformer probabilistic prediction module. This model capitalizes on the time dependence inherent in energy management systems, predicting future state transitions by learning to predict certain state characteristics. Notably, our algorithm seamlessly integrates this predictive capability into the Actor-Critic architecture, avoiding the need for an external model. Through experiments on real data, we demonstrate that integrating predictive capabilities for partial state prediction improves both the sample effectiveness and efficiency of the original PPO approach without requiring exterior prior information.

Repository still being update and code being cleaned.

Off-policy version: transformer already trained beforehand