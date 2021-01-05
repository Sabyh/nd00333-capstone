# MNIST Handwritten Digit Classification using Microsoft Azure

In this capstone project I have took MNIST and iris open source dataset from kaggle and trains it with 2 distinct methods. 
First utilizing AutoML and afterward utilizing HyperDrive with tuned hyperparameters. The best model will later be conveyed and tried.

## Dataset

### Overview
The data is taken from Kaggle repository. 

**Citation:** The data files contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "digit", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

### Task

In this project, Azure AutoML and Hyperdrive will be used to make prediction on MNIST dataset. We will pass our model the image pixels in columns and our model will predict which digit the image pixel represent.

**Clinical features:**

The features are basically pixel values.


### Access
After downloading the MNIST Dataset from kaggle as a csv file, it is registered as a Dataset in the Azure Workspace in a Tabular form uploading from local system. 
I have used **Dataset.get_by_name(ws, dataset_name)** to access our registered dataset.

## Automated ML
The AutoML settings and configuration used are as follows:
```
from azureml.train.automl import AutoMLConfig
import time
import logging

automl_settings = {
    "name": "AutoML_Demo_Experiment_{0}".format(time.time()),
    "experiment_timeout_minutes" : 20,
    "enable_early_stopping" : True,
    "iteration_timeout_minutes": 10,
    "n_cross_validations": 5,
    "primary_metric": 'AUC_weighted',
    "max_concurrent_iterations": 10,
     "primary_metric" : 'accuracy'
}

automl_config = AutoMLConfig(task='classification',
                             debug_log='automl_errors.log',
                             path=project_folder,
                             compute_target=compute_target,
                             training_data=training_data,
                             label_column_name=label,
                             featurization= 'auto',
                             **automl_settings,
                             )
```
Experiment timeout is set to control the use of resources. Maximum 10 iterations can be run simultaneously to maximize usage. Classification task is performed as the target column digit has values 0-9. Primary metric was Accuracy to test how well our model performed in automl. Featurization is likewise done which naturally scales and standardizes the dataset.

<img src="https://github.com/Sabyh/nd00333-capstone/blob/master/Screenshots/mnist/automl-run-status.PNG">
<img src="Screenshots/mnist/best-run-models.PNG">

### Results
The best performing model after training using AutoML is VotingEnsemble with the Accuracy of 0.994110 accuracy.

The other models that are used with VotingEnsemble and there weights are:
```
'ensembled_algorithms': "['MinMaxScalar SVM', 'MaxAbsScaler SVM', 'stackEnsemble',  'StandardScalarWrapper']"

'ensemble_weights': '[0.99993, 0.99992, 0.99985, 0.99956]'
```

To improve the model we can utilize distinctive objective measurement like AUC_weighted or Recall. We can likewise have a go at hypertuning the boundaries to check whether there is any improvement.

<img src="Screenshots/mnist/automl-best-model.PNG">
<img src="Screenshots/mnist/automl3.PNG">

**Best Run Id**

<img src="Screenshots/mnist/best-run-id.PNG">

**Best AutoML Model Registered**

<img src="Screenshots/mnist/register-model.PNG">

## Hyperparameter Tuning
As its a image classification task so we have used convolutional neural network to train it on the image dataset. I have used softmax as a activation function and for loss I have used categorical_crossentropy. 


The parameters used for hyperparameter tuning are:
- epochs (One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.)
- batch_size (the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.)
- dropout (The term “dropout” refers to dropping out units (both hidden and visible) in a neural network. Simply put, dropout refers to ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random.)
- hidden (Number of hidden layers in neural network)

<img src="Screenshots/mnist/hyperdrive-1.PNG">

### Results
The best Accuracy for the HyperDrive model is 81%
The best hyperparameters for this accuracy are:
- 'hidden:': 200
- 'batch-size': 64
- 'drop-out': 1
- 'epoch': 50

To improve the model we can utilize diverse objective measurement to get more extensive point of view. We can likewise take a stab at expanding the scope of the hyperparameters to check whether there is any improvement.
<img src="Screenshots/mnist/hyperdrive-best-model.PNG">

**Best HyperDrive Model Registered**

<img src="Screenshots/mnist/hyperdrive-register-model.PNG">

## Model Deployment
The AutoMl model is conveyed utilizing Azure Container Instance as a WebService. Best run environment and score.py document is given to the InferenceConfig. 

Cpu_cores and memory_gb are instated as 1 for the arrangement design. The aci administration is then made utilizing workspace, aci administration name, model, derivation config and sending arrangement. 

The model is effectively sent as a web administration and a REST endpoint is made with status Healthy. A scoring uri is likewise produced to test the endpoint.

<img src="Screenshots/mnist/model-deployment.PNG">

<img src="Screenshots/mnist/deployed-endpoint.PNG">

The endpoint is tested by following way: 
- using endpoint.py file which passes 1 data points as json to the scoring uri which return a label indicating which digit it is.


<img src="Screenshots/mnist/model-test2.PNG">

## Screen Recording
Link to screencast: [Link](https://drive.google.com/file/d/1cjiBMPU2ahl8U4eu-XX28zSwV44XnCA6/view?usp=sharing)

## Future Improvements
- Larger dataset can be used to increase data quality
- Different parameters can also be used with hyperparameter tuning
- Feature engineering can be performed using PCA 
- We can use max pooling to increase accuracy 
