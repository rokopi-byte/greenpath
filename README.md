# Greenpath

Dataset and Codebase for "GREEN PATH: an expert system for space planning
and design by the generation of human trajectories". [Link to be inserted here]

Codebase is based on [this](https://github.com/cvlab-stonybrook/Scanpath_Prediction). See it for details about data preparation and usage.

Dataset is provided already pre-processed.

Then install requirements using:

    pip install -r requirements.txt

### Training:

Tensorboard callback is used to monitor the training process. Tensorboard events in the asset folder that is generated.

    python train.py retail_dataset.json ..\preprocessed_dataset

### Test

    python test.py retail_dataset.json trained_models ..\preprocessed_dataset