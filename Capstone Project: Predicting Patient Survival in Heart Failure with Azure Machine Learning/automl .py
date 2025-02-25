#!/usr/bin/env python
# coding: utf-8

# # Auto ML on Heart Failure Prediction Dataset
# 
# Importing required dependencies

# In[1]:


import os
import joblib
import azureml.core
from azureml.core import Workspace, Experiment, Dataset, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.widgets import RunDetails
from azureml.train.automl import AutoMLConfig
from pprint import pprint # Used in printing automl model parameters
from azureml.core import Model # Used to get model information

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


# ## Initializing the Workspace
# 
# Workspace initialization from the given configuration. 

# In[2]:


ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')


# ## Creating the Capstone experiment on Azure ML
# 
# The experiment to track all the submitted and completed runs in the given workspace.

# In[3]:


# Choose a name for the run history container in the workspace
experiment_name = 'capstone-automl'
project_folder = './cpautoml-project'
experiment = Experiment(ws, experiment_name)

run = experiment.start_logging()


# ## Adding or Creating the Compute Cluster  
# 
# Creating the compute cluster for the AutoML run

# In[4]:


# Choosing a name for the cluster



compute_cluster_name = "notebook252373"

# Use vm_size = "Standard_D2_V2" in the provisioning configuration. with maximum nodes are being 4



try:
    compute_target = ComputeTarget(workspace=ws, name=compute_cluster_name)
    print("Found existing cluster, use this cluster that was found.")
except:
    print("Creating new cluster...")
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS3_V2', max_nodes=4)
    compute_target = ComputeTarget.create(ws, compute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
print(compute_target.get_status().serialize())



# In[5]:


from azureml.core.dataset import Dataset
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')


found = False
key = "heart-failure-prediction"
description_text = "Prediction of patients’ survival with heart failure - Capstone project"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Creating AML Dataset and register it into Workspace
        my_dataset = 'https://raw.githubusercontent.com/robiulrafi/Azure-ML-Engineer-Nanodegree-Project-Portfolio/main/Optimizing%20an%20ML%20Pipeline%20in%20Azure/heart_failure_clinical_records_dataset.csv'
        dataset = Dataset.Tabular.from_delimited_files(my_dataset)        
        # Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)

df = dataset.to_pandas_dataframe()
df.describe()


# In[6]:


# Reviewing the dataset result
dataset.take(5).to_pandas_dataframe()


# ## AutoML Configuration
# 
# Following configuration is chosen:
# 
# 1. `n_cross_validations`: 2
#    - This parameter specifies the number of cross-validation folds to be used during training. Cross-validation is a technique used to assess the model's performance and generalization ability. A smaller number of folds (like 2 in this case) might be chosen for faster training, especially when computational resources or time are limited.
# 
# 2. `primary_metric`: 'accuracy'
#    - The primary metric determines what aspect of the model's performance to optimize during training. 'Accuracy' is a common metric for classification tasks, as it represents the proportion of correctly predicted instances out of the total instances. Maximizing accuracy ensures that the model makes accurate predictions overall.
# 
# 3. `enable_early_stopping`: True
#    - Enabling early stopping allows the training process to terminate if the performance metric stops improving. This helps prevent overfitting and saves computational resources by stopping training early if the model's performance has plateaued.
# 
# 4. `max_concurrent_iterations`: 4
#    - This parameter specifies the maximum number of parallel iterations or concurrent training jobs to execute. Setting it to 4 allows for multiple training iterations to be executed simultaneously, leveraging available computational resources efficiently and potentially reducing the overall training time.
# 
# 5. `experiment_timeout_minutes`: 20
#    - Sets the maximum amount of time, in minutes, allowed for the experiment to run. A value of 20 minutes ensures that the experiment doesn't run indefinitely and provides a time constraint for model training. This helps prevent excessive resource consumption and ensures timely completion of the experiment.
# 
# 6. `verbosity`: logging.INFO
#    - Controls the amount of logging information displayed during the training process. Setting it to `logging.INFO` ensures that informational messages are logged, providing insights into the training progress without overwhelming the output with excessive details.
#    
# The AutoMLConfig parameters are as follows:
# 
# 1. `compute_target`:
#    - Specifies the compute target where the training will be executed.
# 
# 2. `task`:
#    - Defines the type of task, in this case, 'classification' since we're dealing with a binary classification problem.
# 
# 3. `training_data`:
#    - Indicates the dataset to be used for training the model.
# 
# 4. `label_column_name`:
#    - Specifies the name of the target column ('DEATH_EVENT') that the model will predict.
# 
# 5. `featurization`:
#    - Controls the featurization process. Setting it to 'auto' allows AutoML to automatically handle feature engineering.
# 
# 6. `debug_log`:
#    - Specifies the file path to store debug logs generated during the training process.
# 
# 7. `**automl_settings`:
#    - Additional settings for the AutoML experiment, such as the number of cross-validations, primary metric, early stopping, maximum concurrent iterations, experiment timeout, and verbosity level. These settings are passed as keyword arguments to the AutoMLConfig.

# In[7]:


# Putting automl settings here
automl_settings = {"n_cross_validations": 2,
                    "primary_metric": 'accuracy',
                    "enable_early_stopping": True,
                    "max_concurrent_iterations": 4,
                    "experiment_timeout_minutes": 20,
                    "verbosity": logging.INFO
                    }

# Parameters for AutoMLConfig

automl_config = AutoMLConfig(compute_target = compute_target,
                            task='classification',
                            training_data=dataset,
                            label_column_name='DEATH_EVENT',
                            featurization= 'auto',
                            debug_log = "automl_errors.log",
                            **automl_settings
                            )


# In[8]:


# Submitting the experiment
automl_run = experiment.submit(automl_config)


# ## Run Details

# In[9]:


RunDetails(automl_run).show()
automl_run.wait_for_completion(show_output=True)


# ## Finding the Best Model with the best run

# In[10]:


# Get the best run and model
best_run_automl, best_model_automl = automl_run.get_output()


# In[11]:


##checking the best model 
best_run_automl


# In[12]:


print('Best Run Id: ' + best_run_automl.id,
     'Best Model Name: ' + best_run_automl.properties['model_name'])


# In[13]:


# get_metrics()
# Returns the metrics
print("Best run metrics :",best_run_automl.get_metrics())


# In[14]:


# get_details()
# Returns a dictionary with the details for the run
print("Best run details :",best_run_automl.get_details())


# In[15]:


# get_properties()
# Fetch the latest properties of the run from the service
print("Best run properties :",best_run_automl.get_properties())


# In[16]:


best_run_automl.get_tags()


# ## Model Deployment

# In[18]:


# Register the model
best_autoML_Model=best_run_automl.register_model(model_path='outputs/', model_name='automl-best-model',
                   tags={'Training context':'AutoML', 'type': 'Classification'},
                  
                   description = 'AutoML based Heart Failure Predictor')


# In[19]:


# List registered models to verify if model has been saved
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')


# In[49]:


best_run_automl.get_file_names()

# Downloading the yaml file that includes the environment dependencies
best_run_automl.download_file('outputs/conda_env_v_1_0_0.yml', 'env.yml')


# In[51]:


# Downloading the model file

best_run_automl.download_file('outputs/model.pkl', 'Automl_model.pkl')


# In[53]:


best_run_automl.register_model(model_name = "best_run_automl.pkl", model_path = './outputs/')

print(best_run_automl)


# In[27]:


best_run_automl.download_file('outputs/model.pkl', './model.pkl')

# Downloading the scoring file
best_run_automl.download_file('outputs/scoring_file_v_1_0_0.py', './score.py')

best_run_automl.download_file('outputs/conda_env_v_1_0_0.yml', 'envFile.yml')


# In[58]:


model = automl_run.register_model(model_name = 'best_run_automl.pkl')
print(automl_run.model_id)


environment = best_run_automl.get_environment()
entry_script='inference/scoring.py'
best_run_automl.download_file('outputs/scoring_file_v_1_0_0.py', entry_script)


inference_config = InferenceConfig(entry_script = entry_script, environment = environment)

# Deploying the model via ACI WebService
# https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-deploy-azure-container-instance.md

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                                    memory_gb = 1, 
                                                    auth_enabled= True, 
                                                    enable_app_insights= True)

service = Model.deploy(ws, "deployservice", [model], inference_config, deployment_config)
service.wait_for_deployment(show_output = True)


# In[60]:


import json
import requests

# importing test data
test_df = df.sample(5) # data is the pandas dataframe of the original data
label_df = test_df.pop('DEATH_EVENT')

test_sample = json.dumps({'data': test_df.to_dict(orient='records')})
print(test_sample)


# In[61]:


# Authentication is enabled, so I use the get_keys method to retrieve the primary and secondary authentication keys:
primary, secondary = service.get_keys()

print('Service state: ' + service.state)
print('Service scoring URI: ' + service.scoring_uri)
print('Service Swagger URI: ' + service.swagger_uri)
print('Service primary authentication key: ' + primary)


# In[47]:


test_sample


# In[66]:


# predict using the deployed model
result = service.run(test_sample)
print(result)


# In[71]:


## Endpoint Consumption
import requests
import json

# URL for the web service

scoring_uri = service.scoring_uri
# If the service is authenticated, set the key or token
key = primary

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
           "age": 60, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 59, 
           "diabetes": 0, 
           "ejection_fraction": 25, 
           "high_blood_pressure": 1, 
           "platelets": 212000, 
           "serum_creatinine": 3.5, 
           "serum_sodium": 136, 
           "sex": 1, 
           "smoking": 1,
           "time": 187
          },
          {
           "age": 51, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 1380, 
           "diabetes": 0, 
           "ejection_fraction": 25, 
           "high_blood_pressure": 1, 
           "platelets": 271000, 
           "serum_creatinine": 0.9, 
           "serum_sodium": 130, 
           "sex": 1, 
           "smoking": 0,
           "time": 38
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
print("++++++++++++++++++++++++++++++")
print("Expected result: [false, true], where 'true' means '1' as result in the 'DEATH_EVENT' column")


# In[69]:


## Printing The Service Logs
print(service.get_logs())


# In[ ]:


service.delete()

