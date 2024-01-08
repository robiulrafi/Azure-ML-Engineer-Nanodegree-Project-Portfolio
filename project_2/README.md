# MLOps: Orchestrating Machine Learning Operations
## Overview
In this project's context, an AutoML run was executed on the [Bank Marketing](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) Dataset. The top-performing model obtained from the AutoML process was deployed as a RESTful API Webservice. Endpoints were utilized to interact with the deployed model in Microsoft Azure Machine Learning Studio. Lastly, pipeline automation was employed to enhance machine learning operations. 

## Architectural Diagram
This diagram visually summarizes the workflow of this project.
![Image Alt Text](Artifacts/workflow.PNG)
**Image Courtesy: Nanodegree course for machine learning operation using MS Azure**

**Authentication**: This step aims to create a Service Principal for workspace access. However, due to restricted privileges in the Udacity-provisioned AML lab environment, it couldn't be executed in this project.

**Auto ML Model**: The objective is to leverage AutoML for training and ranking classification models based on AUC-weighted accuracy, aiding in identifying the most effective model for deployment.

**Deploying the Best Model**: This step involves selecting and deploying the top-performing model and establishing an endpoint for interaction to enable its utilization.

**Enabling Logging**: The goal here is to activate Application Insights to gather operational health logs from the deployed model's endpoint, enabling ongoing monitoring.

**Consuming Model Endpoints**: This step executes a script to interact with the deployed model's endpoint, testing its functionality, and saving data for benchmarking purposes.

**Pipeline Creation and Publishing**: This step creates an automated pipeline endpoint for AutoML training, streamlining and improving the efficiency of model training processes.

**Documentation**: This step provides a comprehensive screencast illustrating the ML application's operation and generates a detailed README.md to describe and document the project's essential steps.


## Key Steps
The workflow can be decomposed into the following pivotal steps:

**Dataset Registration**: 
The following code snippet and screenshot show how the workflow registers the  [Bank Marketing](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) dataset in the Azure ML studio.
```python
found = False
key = "BankMarketing Dataset"
description_text = "Bank Marketing DataSet for Udacity Course 2"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)
```
![Image Alt Text](Artifacts/Data_Registration.PNG)

**Automated ML Experiment**:  
In this project, the following configuration of AutoML has been used:
```python
from azureml.pipeline.core import PipelineData, TrainingOutput

ds = ws.get_default_datastore()
metrics_output_name = 'metrics_output'
best_model_output_name = 'best_model_output'

metrics_data = PipelineData(name='metrics_data',
                           datastore=ds,
                           pipeline_output_name=metrics_output_name,
                           training_output=TrainingOutput(type='Metrics'))
model_data = PipelineData(name='model_data',
                           datastore=ds,
                           pipeline_output_name=best_model_output_name,
                           training_output=TrainingOutput(type='Model'))
```
Once the AutoML steps are executed, the experiment undergoes training and testing using the provided data within the AutoML environment to derive the best model. Below are the accompanying screenshots:

![Image Alt Text](Artifacts/Job_end_details.PNG)
**Screenshot of job completion** 

![Image Alt Text](Artifacts/Best_model_N.PNG)

**Screenshot of the Best Model** 

**Deployment of the Best Model and Enable Logging** 
The deployment involves utilizing the Best Model to interact with the HTTP API service. This interaction is established by transmitting data through POST requests to the model. Once the Best Model is deployed, the process involves using Application Insights to enable log retrieval. These logs are showcased in the following snapshots:

![Image Alt Text](Artifacts/Application_Insights_true.PNG)

**Enabling the Application Insights**

![Image Alt Text](Artifacts/Application_Insights_dashboard.PNG)

**Status of the Application Insights Dashboard**

![Image Alt Text](Artifacts/Log_shots.PNG)

**Enabling and Capturing the logs**

**Consume model using Swagger**

During this step, the deployed model was accessed using Swagger. Utilizing a swagger-based JSON file provided by Azure for deployed models, interactions were made. The snapshot below demonstrates the engagement with Swagger instances running for documenting the HTTP API associated with the model.

![Image Alt Text](Artifacts/Swagger_api_data.PNG)

**Swagger instances running for the Documentation of the HTTP API associated with the best model**



## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

