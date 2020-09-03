<p align="center">
  <img width="1600" height="200" src="images/bo_titlev3.png">
</p>


# Introduction

Effective supply chain management is an integral part of many businesses and essential to business success, customer satisfaction, and operational costs management. 
The goal of a supply chain is to have an item in stock when a customer is ready to purchase the item, in the cheapest and fastest means possible.

One of the common challenges in supply chain management is the complexity associated with predicting backorder. Luckily with the current data growth and advances in predictive analytics and artificial intelligence, machine learning could rescue supply chain managers from this daunting task and therefore save your business from unnecessary costs associated with backorder.

In this project, I demonstrate using powerful python open source libraries and packages (pandas, scikit-learn, TensorFlow, and Keras) to build a model that predicts items that are more likely to backorder given some input features.

Once satisfied with the model results, I’ll deploy the predictive solution as a web service through Heroku, a Platform as a Service (PaaS) application, where any user will be able to input required item features and receive a prediction whether that item is or is not likely to backorder within the next eight weeks.  

# Data Collection

The dataset is retrieved from Kaggle and contains **1.687 million records of inventory data** for eight weeks prior to the week we want to predict backorder for.

The dataset contains the following features:

- sku - A unique identifier for each product
- national_inv - Current inventory level for the product    
- lead_time - Transit time for product (if available)
- in_transit_qty - Amount of product in transit from source
- forecast_3_month - Forecast sales for the next 3 months
- forecast_6_month - Forecast sales for the next 6 months
- forecast_9_month - Forecast sales for the next 9 months
- sales_1_month - Sales quantity for the prior 1 month time period
- sales_3_month - Sales quantity for the prior 3 month time period
- sales_6_month - Sales quantity for the prior 6 month time period
- sales_9_month - Sales quantity for the prior 9 month time period
- min_bank - Minimum recommend amount to stock
- potential_issue - Source issue for part identified
- pieces_past_due - Parts overdue from source
- perf_6_month_avg - Source performance for prior 6 month period
- perf_12_month_avg - Source performance for prior 12 month period
- local_bo_qty - Amount of stock orders overdue
- deck_risk - Part risk flag
- oe_constraint - Part risk flag
- ppap_risk - Part risk flag
- stop_auto_buy - Part risk flag
- rev_stop - Part risk flag
- went_on_backorder – backorder indicator (Yes-backordered, No - Not backorder)

For detailed code and dataset structure information see: **[01. File_Exploration.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/01.%20File_Exploration.ipynb)**


# Exploratory Data Analysis
- I explored the data in two stages, to:
	1. Understand the data, and
	2. Understand the typical profile of an item that backordered in the past eight weeks.

My goal here was to extract meaningful insights from our feature set that our model will use as signal for learning the general pattern of  typical backordered items, a process known as generalization. Generalization is what will help our final model perform well on unseen data. 

- To understand the data, I took the following steps and made the following observations:
	1. Ensured no duplicate samples, by verifying the dataset has only one record per “sku.”
	2. Explored missing data, outliers, and important feature correlations and handled them accordingly.
	3. Explored data distributions and noted a highly imbalanced dataset (only 6% instances of backorder).
	
<p align="center">
  <img width="630" height="600" src="images/1. data_imbalance.png">
</p>
			iv. Performed a univariate outlier analysis and noted a high presence of possible data outliers in the dataset

For detailed code and charts on exploratory data analysis (please see: **[02(a). Data_Exploration](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/02.%20Data_Exploration/02%28a%29.%20Data_Exploration.ipynb)**)

- To understand the typical profile of a backordered item, I
	1. Evaluated linear correlations in the dataset and removed multicollinear features to reduce our problem’s feature space. 
	2. Performed statistical tests of significance to uncover important data signals for typical backordered items.
		- I noted the following:
			- Items with low inventory, low lead times, low quantities in transit, and low supplier performance are significantly more likely to backorder.
			- Items with a high sales forecast are significantly more likely to backorder.

For detailed code on **identifying hidden data relationships** please see: **[02(b). Data_Exploration_Numericals](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/d4e194e7a204364e010481568f647cf02681c3cc/notebooks/02.%20Data_Exploration/02(b).%20Data_Exploration-Numericals.ipynb)** and **[02(c) Data_Exploration_Categoricals](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/734cbcfccf23ee94ed5f96fec403315bc62bd945/notebooks/02.%20Data_Exploration/02(c)%20Data_Exploration-Categoricals.ipynb)**
		

# Data Preprocessing

To prepare for data for modeling, I performed the following actions:

- Tested various methods for handling the data imbalance identified during data exploration and selected to use a combination of under-sampling the majority sample and over-sampling the minority sample using the Synthetic Minority  Oversampling Technique (SMOTE).

<p align="center">
  <img width="630" height="600" src="images/balanced.png">
</p>

For detailed code and charts on handling the dataset imbalance please see: **[03(a). Data_Imbalance.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/03.%20Data_Preprocessing/03%28a%29.%20Data_Imbalance.ipynb)**

- Performed feature engineering to add important data relationships (signals) uncovered during EDA. 
The following features were added: 
	* ‘‘low_inventory”
	* “low_lead_time”
	* “low_qty_intransit”
	* “high_forecast”
	* “low_perfomance”

For detailed code on feature engineering steps please see: **[3(b). Feature_Engineering.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/03.%20Data_Preprocessing/03%28b%29.%20Feature_Engineering.ipynb)**

- Evaluated learning curves to see whether the limit on model complexity and training dataset size is reached.
The results showed that increasing the training dataset size further will not improve performance.

<p align="center">
  <img width="550" height="400" src="images/3. learning_curves.png">
</p>

For detailed code and charts on learning curves please see: **[3(c). learning_Curves.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/03.%20Data_Preprocessing/03%28c%29.%20Learning_Curves.ipynb)**

- Identified features that are more likely to impact backorder using Step Forward Feature Selection (SFFS) and SelectKBest feature selection python scikit learn methods.

I noted that our model will only be able to use at least 5 but not more than 11 features for best performance.

The peak  performance of 82.07%  was reached when eight features were selected. 

<p align="center">
  <img width="630" height="300" src="images/4. sffs.png">
</p>

For detailed code and charts on feature selection please see: **[03(d). Feature_Selection_Wrapper_Methods.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/03.%20Data_Preprocessing/03%28d%29.%20Feature_Selection_Wrapper_Methods.ipynb)** and **[03(e). Feature_Selection_BestK.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/03.%20Data_Preprocessing/03%28e%29.%20Feature_Selection_BestK.ipynb)**

# Data Modeling

To model the data, I implemented the following steps:

- Evaluated **baseline performance** by preparing different baseline models using different criteria. 
For example, I prepared a model that predicts randomly and observed that given the data we have, such a model could only correctly identify backorders less than 1% of the time.

<p align="center">
  <img width="630" height="600" src="images/5. cm1.png">
</p>

Our objective was therefore to improve from this baseline performance. (please see: **[4(a). Baseline_Model.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/04.%20Data_Modeling/04%28a%29.%20Baseline_Model.ipynb)**)

- Performed **model selection** from a candidate pool of promising models by considering model complexity and training time compromise constraints.
The analysis showed that the simpler logistic regression and linear discriminant analysis (LDA) models will suit our purposes well.
(please see: **[4(b).Model_Selection.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/04.%20Data_Modeling/04%28b%29.%20Model_Selection.ipynb)** )

- Performed **hyperparameter optimization** on our best candidate models and achieved a recall of 83.15% using a logistic regression model with the following hyperparameters:
	- solver = “liblinear”
	- penalty = “l2”
	- c = 0.1

For complete code and charts on hyperparameter optimization please see: **[04(c). model_hyperparameters.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/6b80773dccca3f67fd67bd21aced90f7053d8bd0/notebooks/04.%20Data_Modeling/04(c).%20model_hyperparameters.ipynb)**

- Handled data imbalance on our final model test set using same prior methods applied on the training set. (please see: **[4(d). balance_validation_set.ipynb](https://nbviewer.jupyter.org/github/mfalila/supply_chain_analysis/blob/master/notebooks/04.%20Data_Modeling/04%28d%29.%20balance_validation_set.ipynb)**)

- Wrapped all steps above to a data preprocessing pipeline for data modeling.

<p align="center">
  <img width="630" height="650" src="images/6. preprocessing_pipeline.png">
</p>

# Results

After preprocessing the data and selecting candidate hyperparameters on selected promising models obtained from analyses on prior steps, I fitted the selected models on the data and obtained the following results:

- Improved recall from less than 1% to 75.0% at 79.5% total accuracy with a simple logistic regression model.

<p align="center">
  <img width="630" height="600" src="images/7. cm2.png">
</p>

For detailed code on final modeling please see: **[05. Final_Model.ipynb](https://github.com/mfalila/supply_chain_analysis/blob/master/notebooks/05.%20Final_Model.ipynb)**

- Trained and fitted a deep learning  convolutional neural network and obtained even higher performance (improved recall to 80.9% at 86.8% accuracy).

<p align="center">
  <img width="630" height="600" src="images/8. cm3.png">
</p>

For detailed code and charts on the CNN model please see: **[06. cnn.ipynb](https://github.com/mfalila/supply_chain_analysis/blob/master/notebooks/06.%20cnn.ipynb)**

# Model Deployment

Model simplicity s an important consideration in applied machine learning. In practice, compromising model accuracy for model simplicity is common. For example, while we achieved better performing model using a deep learning framework, a more simpler model with equally acceptable performance seems to be a better option for our purposes, as simpler models are easier to train and maintain during the model’s life cycle.

For these reasons, I selected the logistic regression as a final deployment model.

To deploy our model, I took the following steps:
- Built a model’s web API using the Flask web development framework
- Created a GitHub repository to store our model and required deployment code
- Deployed our model using Heroku, a cloud platform as a service supporting python web applications. 

Below is the model’s web API where a user can enter item information, hit predict, and the model returns a message stating whether the entered item will or will not backorder.

<p align="center">
  <img width="630" height="200" src="images/9. deployment.png">
</p>

The deployed model can be accessed from **[Backorder Predictor api](https://ml-backorder-predictor.herokuapp.com/)**

For detailed deployment code please see: **[deployment](https://github.com/mfalila/supply_chain_analysis/tree/master/deployment)**

# Conclusion

My project objective was to use python libraries to build a machine learning predictive solution that will identify items that are more likely to back order.

I met this objective by taking the following steps:

- Explored the dataset to understand the data file structure and the data structure.
- Perform different tests of statistical significance to uncover hidden data relationships that our predictive model could learn from and leverage when predicting unseen instances.
- Employed statistical analysis using various python libraries to identify the number and names of features that could more likely help in identifying backordered items.
- Built and deployed a final logistic regression model as a web application on Heroku PaaS.

# Further Improvements and Applications

While I achieved impressive results, there is still room for improvement. For example:
 - Performing log transformations on features that were not gaussian like and observe if we might get performance boost.
 - Building model ensembles.

All these decisions are detected by the need to meet business objectives of the data science project.

In addition, the following applications could be augmented to the project:	
-  Modifying the model’s API with a few lines of code to accept a JSON or csv inputs for batch predictions
-  Output probabilities associated with predictions to aid supply chain managers  take specific actions if an item has a probability to backorder above specified user defined thresholds.
- Integrating the API as part of a full supply management system where an order is automatically triggered if the probability of an item backordering exceeds some user defined threshold.
	
# References
Corrales, D. C., Lasso, E., Ledezma, A., Corrales, J. C., Pinto, Singh, Villavicencio, Mayr-Schlegel, & Stamatatos. (2018). Feature selection for classification tasks: Expert knowledge or traditional methods? Journal of Intelligent & Fuzzy Systems, 34(5), 2825–2835. https://doi.org/10.3233/JIFS-169470

Huan Liu, & Setiono, R. (1995). Chi2: feature selection and discretization of numeric attributes. Proceedings of 7th IEEE International Conference on Tools with Artificial Intelligence, 388.

Nadim Nachar. (2008). The Mann-Whitney U: A Test for Assessing Whether Two Independent Samples Come from the Same Distribution. Tutorials in Quantitative Methods for Psychology, 4(1), 13–20.

sklearn.discriminant_analysis.LinearDiscriminantAnalysis Retrieved August 30, 2020 from https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
sklearn.feature_selection.SelectKBest. Retrieved August 30, 2020 from https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

sklearn.linear_model.LogisticRegression. Retrieved August 30, 2020 from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Yap, W. B., Rani, A. K., Rahman, A. A. H., Fong, S., Khairudin, Z., & Abdullah, N. N. (2014). An Application of Oversampling, Undersampling, Bagging and Boosting in Handling Imbalanced Datasets

Zou, X., Feng, Y., Li, H., & Jiang, S. (2018). Improved over-sampling techniques based on sparse representation for imbalance problem. Intelligent Data Analysis, 22(5), 939–958. https://doi.org/10.3233/IDA-173534



















