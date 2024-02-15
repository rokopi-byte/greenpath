# Greenpath

Dataset and Codebase for "GREEN PATH: an expert system for space planning
and design by the generation of human trajectories". [Link to the paper](https://doi.org/10.1007/s11042-024-18228-6)

## Abstract:
Public space is usually conceived as where people live, perceive, and interact with other people. The environment affects people in several different ways as well. The impact of environmental problems on humans is significant, affecting all human activities, including health and socio-economic development. Thus, there is a need to rethink how space is used. Dealing with the important needs raised by climate emergency, pandemic and digitization, the contributions of this paper consist in the creation of opportunities for developing generative approaches to space design and utilization. It is proposed GREEN PATH, an intelligent expert system for space planning. GREEN PATH uses human trajectories and deep learning methods to analyse and understand human behaviour for offering insights to layout designers. In particular, a Generative Adversarial Imitation Learning (GAIL) framework hybridised with classical reinforcement learning methods is proposed. An example of the classical reinforcement learning method used is continuous penalties, which allow us to model the shape of the trajectories and insert a bias, which is necessary for the generation, into the training. The structure of the framework and the formalisation of the problem to be solved allow for the evaluation of the results in terms of generation and prediction. The use case is a chosen retail domain that will serve as a demonstrator for optimising the layout environment and improving the shopping experience. Experiments were assessed on shoppers’ trajectories obtained from four different stores, considering two years.


Codebase by [Doch88](https://github.com/Doch88), based on [this](https://github.com/cvlab-stonybrook/Scanpath_Prediction). See it for details about data preparation and usage.

Dataset is provided already pre-processed.

Then install requirements using:

    pip install -r requirements.txt

### Training:

Tensorboard callback is used to monitor the training process. Tensorboard events in the asset folder that is generated.

    python train.py retail_dataset.json ..\preprocessed_dataset

### Test

    python test.py retail_dataset.json trained_models ..\preprocessed_dataset
