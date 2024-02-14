# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.


## Summary
The Bank Marketing dataset, sourced from the [UCI ML](https://archive.ics.uci.edu/dataset/222/bank+marketing) Repository, contains demographic data and responses (Yes or No) from bank clients to direct phone marketing campaigns promoting term deposit products. Its primary objective involves predicting whether a client will choose to subscribe to a term deposit. This dataset includes various columns depicting client demographics as input variables, with the 'y' output variable representing the binary status of whether the client has opted for a term deposit, denoted as Yes or No. Essentially, this dataset forms the foundation for modeling efforts aimed at forecasting clients' decisions regarding term deposit subscriptions based on their demographic profiles.

The VotingEnsemble, an ensemble model generated through AutomML, stands out as the top-performing model. It boasts an impressive accuracy rate of **91.49%**. In comparison, the LogicRegression model, supported by HyperDrive in Scikit-learn, achieves an accuracy of **91.08%**. This signifies that the VotingEnsemble model outperforms the LogicRegression model by a margin of **0.41%** in terms of accuracy, showcasing its superiority in predictive capabilities in this scenario.

## Scikit-learn Pipeline

**Pipeline architecture**

The pipeline architecture involves several components orchestrated to train a machine-learning model. Here's an overview:

**1. Python Training Script ([train.py](https://github.com/robiulrafi/AZURE_ML_ND_PORTFOLIO/blob/main/project_1/train.py))**: This script contains the code responsible for training the model. It likely includes data preprocessing, model fitting, and evaluation steps.

**2. Tabular Dataset from UCI ML Repository**: This dataset sourced from the UCI ML Repository serves as the input data for training the machine learning model. It contains the information required for model learning and prediction.

**3. Scikit-learn Logistic Regression Algorithm**: This algorithm, provided by Scikit-learn, is utilized for model training. It's a linear classification algorithm used to predict categorical outcomes based on input features.

**4. Azure HyperDrive**: The Scikit-learn Logistic Regression model is connected to Azure HyperDrive, which functions as a hyperparameter tuning engine. HyperDrive explores different hyperparameter configurations to optimize the model's performance, enhancing its accuracy or other specified metrics.

**5. Jupyter Notebook on Compute Instance**: The training run is managed and executed through a [Jupyter Notebook](https://github.com/robiulrafi/AZURE_ML_ND_PORTFOLIO/blob/main/project_1/udacity-project.ipynb) hosted on a compute instance. This notebook likely coordinates data loading, model training, hyperparameter tuning setup, and monitoring of the training process.

The architecture diagram as demonstrated below, credited to Udacity's **MLEMA Nanodegree**, provides a visual representation of the logical flow of these components, showcasing how they interact and contribute to the model training process.

![Image Alt Text](Artifacts/Pipeline_Sklearn.PNG)


**The benefits of the chosen parameter sampling**

In this project, the following random parameter sampling approach has been adopted: 
+ ps = RandomParameterSampling({'--C': choice(0.01, 0.1, 0.2, 0.5, 0.7, 1.0), '--max_iter': choice(range(10,110,10))})
  
The advantages of employing the parameter sampling method, specifically the RandomParameterSampler, for HyperDrive tuning are noteworthy. In this case, the sampler has been configured to optimize parameters for the [Sckit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model, focusing on 'C' (Regularization Strength) and 'max_iter' (Max iterations).

This sampler brings several benefits to the tuning process:

**1. Support for Discrete and Continuous Parameters**: The sampler accommodates both discrete and continuous hyperparameters, enabling a wide range of potential values for 'C' and 'max_iter' in the LogisticRegression model.

**2. Early Termination of Low-Performance Runs**: It allows for early termination of underperforming runs, optimizing computational resources by stopping runs that are not showing promising results based on the chosen metric.

**3. Simplicity and Reduced Bias**: Its straightforward implementation simplifies the parameter search process, reducing bias in hyperparameter selection. This unbiased approach contributes to enhancing the model's accuracy.

**4. Versatility for Hyperparameter Combinations**: Random parameter sampling explores various hyperparameter combinations, fostering an exploratory learning process that can uncover effective configurations for the model.

However, it's important to note that while random parameter sampling offers these benefits, it may require more execution time due to its exploration of diverse parameter spaces. Overall, its ability to support both discrete and continuous parameters, terminate low-performance runs, and facilitate an unbiased search process makes it a valuable choice for optimizing models via HyperDrive tuning.


**The benefits of the chosen early stopping policy**

In this experiment, the following stopping policy is specified:

+ policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=2, delay_evaluation=5) 

The BanditPolicy, employed as an early termination policy within HyperDrive, functions autonomously to halt runs that exhibit poor performance, significantly enhancing computational efficiency. This policy relies on parameters such as slack factor/slack amount and evaluation interval. It dynamically cancels runs wherein the primary metric deviates beyond the specified slack factor/slack amount in comparison to the best-performing run. This proactive approach ensures that only runs showcasing promising performance aligned with the defined metric parameters continue, optimizing computational resources and expediting the model optimization process.

## AutoML

The AutoML run utilized the following configuration settings within the **AutoMLConfig**:

```python
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    max_concurrent_iterations=4,
    max_cores_per_iteration=-1,
    featurization='auto',
    iterations=30,
    enable_early_stopping=True,
    debug_log = 'automl_errors.log',
    training_data=df_train,
    label_column_name='y',
    n_cross_validations=2)
```
These configurations were used to define the behavior and parameters for the AutoML run. Key settings include the task type (classification), the primary metric for evaluation (accuracy), iteration and timeout settings, early stopping, featurization, and logging specifications. The df_train dataset was employed as the training data, with 'y' as the label column, and 2 cross-validations were performed during the training process. Due to constraints linked to the Udacity Virtual Machine for Azure, this experiment is limited to a duration of 30 minutes before it times out. This constraint dictates the maximum duration allowed for model training within this environment, influencing the experimentation strategy and resource utilization.

## Pipeline comparison

The Scikit-learn LogicRegression model, empowered by HyperDrive, achieved an impressive top accuracy of **91.08%**, as evidenced below:

![Image Alt Text](Artifacts/Capture_4_HD.PNG)

![Image Alt Text](Artifacts/Capture_5_HD.PNG)

In contrast, the AutoML-driven **VotingEnsemble** model exhibited an exceptional top accuracy of **91.49%**, surpassing the aforementioned result:

![Image Alt Text](Artifacts/Capture_8_AML.PNG)

![Image Alt Text](Artifacts/Capture_2_AML.PNG)

While the disparity in accuracy appears slight, there exists a notable architectural contrast between the two approaches. While HyperDrive was limited to evaluating Logistic Regression from Sci-KitLearn, AutoML boasted an extensive repertoire of approximately 20 models for evaluation. This vast selection puts HyperDrive at a distinct disadvantage compared to AutoML, as AutoML offers a diverse pool of over 20 models available for selection during experiments.

Moreover, AutoML extends its utility beyond model creation by offering model interpretation capabilities. This functionality proves beneficial in comprehending the rationale behind a model's predictions and gaining insight into the significance of individual features for specific tasks. Here are some illustrations below:

![Image Alt Text](Artifacts/Capture_3_AML.PNG)

The above figure demonstrates the feature importance of the given classification task. 

The below figures show us the different performance curves and confusion matrix of the given classification problem via **VotingEnsemble** model. 
![Image Alt Text](Artifacts/Capture_5_AML.PNG)

![Image Alt Text](Artifacts/Capture_7_AML.PNG)


In addition, the AutoML also detects the class imbalance in the dataset automatically as shown in the evidence from the [Jupyter Notebook](https://github.com/robiulrafi/AZURE_ML_ND_PORTFOLIO/blob/main/project_1/udacity-project.ipynb) run below: 

![Image Alt Text](Artifacts/Capture_9_AML.PNG)


## Future work

Potential Areas for Enhancements in Future Experiments:

**1. Leveraging AutoML's Model Interpretability**: Extend the application of AutoML's model interpretability on larger and more intricate datasets. This approach promises accelerated insights into feature engineering, enabling the refinement of complex model accuracy.

**2. Exploration of Hyperparameter Sampling Methods**: Experiment with diverse hyperparameter sampling methods such as Grid sampling or Bayesian sampling. This exploration could encompass the Scikit-learn LogicRegression model or other custom-coded machine learning models. Consider adopting Bayesian Parameter Sampling, employing Markov Chain Monte Carlo methods for probability distribution sampling, as part of future enhancements for HyperDrive.

**3. Refinements in AutoML Settings**: Consider adjusting the experiment timeout in AutoML to facilitate additional model experimentation. While addressing class imbalance is crucial, identifying the most pertinent performance metric for a specific task stands as an additional focal point. Determining the optimal performance metric tailored to the task's requirements could provide further insights and refine the evaluation process.


## Proof of cluster clean up
The proof cluster cleaning is shown in the image below:

![Image Alt Text](Artifacts/Capture_cluster.PNG)
