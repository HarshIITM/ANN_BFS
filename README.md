# ANN_BFS
A machine learning model to learn the numerical solution of 2 dimensional Backward Facing Step.

An Artificial Neural Network (ANN) architecture is developed which can learn the spatial and temporal information of the flow field variables ( u-velocity, v-velocity, stream function, vorticity function) 

The ANN is trained with flow field at different time so that it can be used later for predicting these above mentioned physical quantities across the domain at any time-step.

The model has shown the training accuracy of 97.5 percent with just 100 epochs of training.

ANNbfs.py is the main code of algorithm.

.json file is the saved model architecture which can be called inside the main code.
