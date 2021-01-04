# MNIST classification using Microsoft Azure

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

**12 clinical features:**

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

<img src="Screenshots/autoML_run.png">
<img src="Screenshots/automl_experiment.png">

### Results
The best performing model after training using AutoML is VotingEnsemble with the Accuracy of 88.49350649350649 %

The models that VotingEnsemble used with it's weight are:
```
'ensembled_algorithms': "['LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'LightGBM', 'RandomForest', 'ExtremeRandomTrees', 'GradientBoosting', 'RandomForest', 'XGBoostClassifier', 'LogisticRegression']"

'ensemble_weights': '[0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091]'
```

To improve the model we can use different target metric like AUC_weighted or Recall. We can also try hypertuning the parameters to see if there is any improvement.

<img src="Screenshots/automl_best_model.png">
<img src="Screenshots/automl_best_accuracy.png">

**Best Run Id**

<img src="Screenshots/automl_bestrunid.PNG">

**Best AutoML Model Registered**

<img src="Screenshots/automl_registered.PNG">

## Hyperparameter Tuning
As it is a binary classification problem, the model used for HyperDrive is Logistic Regression. 
- It is easy to understand
- Trains very easily and faster compared to complex models


The parameters used for hyperparameter tuning are:
- Regularization Strength (C) with range 0.1 to 1.0
    -- Inverse of regularization strength. Smaller values cause stronger regularization
- Max Iterations (max_iter) with values 50, 100, 150 and 200
    -- Maximum number of iterations to converge

<img src="Screenshots/hyperdrive_run.png">

### Results
The best Accuracy for the HyperDrive model is 7666666666666667 %
The best hyperparameters for this accuracy are:
- 'Regularization Strength:': 0.5077980350098886
- 'Max iterations:': 50

To improve the model we can use different target metric to get broader perspective. We can also try increasing the range of the hyperparameters to see if there is any improvement.
<img src="Screenshots/hyperdrive_bestmodel.png">

**Best HyperDrive Model Registered**

<img src="Screenshots/hyperdrive_registered.PNG">

## Model Deployment
The AutoMl model is deployed using Azure Container Instance as a WebService. Best run environment and score.py file is provided to the InferenceConfig.
Cpu_cores and memory_gb are initialized as 1 for the deployment configuration. The aci service is then created using workspace, aci service name, model, inference config and deployment configuration.

The model is successfully deployed as a web service and a REST endpoint is created with status Healthy. A scoring uri is also generated to test the endpoint.

<img src="Screenshots/model_deployment.png">

<img src="Screenshots/deployed_endpoint.png">

The endpoint is tested in 2 ways: 
- using endpoint.py file which passes 2 data points as json 
- using 3 random sample data points and to see the actual value and predicted value 

<img src="Screenshots/model_test.png">

## Screen Recording
Link to screencast: [Link](https://youtu.be/fj7Av9YiuiY)

## Future Improvements
- Larger dataset can be used to increase data quality
- Different models can also be used with hyperparameter tuning
- Feature engineering can be performed using PCA 
