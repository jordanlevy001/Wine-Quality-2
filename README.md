# Wine-Quality-2
A second machine learning wine quality analysis performed as a follow up to the ML-Wine-Quality repository.

## Project Overview
The goal of this project was to use machine learning techniques: logistic regression, SVM, Random Forest Classifier to predict the overall quality score (likeability) of various wines. The data for this project was obtained with free use from [Kaggle](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) through UCI Machine Learning.

## Technology/Software Utilized
- Google Colab
- TensorFlow, Keras
- Scikit-learn
- Pandas
- Plotly
- Seaborn
- NumPy
- Matplotlib
- hvPlot

## Data
Free Data Source: [Kaggle](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

Columns:
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfure Dioxide
7. Total Sulfur Dioxide
8. Density -- this column can also be dropped
9. pH
10. Sulphates
11. Alcohol
12. **Quality --> OUTPUT: this is the overall quality score**
13. Reviews -- this column was added to categorize the quality score into 1 of 3 categories: 1. Bad 2. Average 3. Excellent

<img width="1117" alt="Wine Stats" src="https://user-images.githubusercontent.com/88804543/154178721-63f21163-eb1d-46bb-aaa1-35f497011f22.png">


## Exploratory Data Analysis, Processing, Visualization

Looking at the quality score counts:

<img width="434" alt="Quality Score Counts" src="https://user-images.githubusercontent.com/88804543/154179935-911cc753-3884-4529-9c64-03723b0a8262.png">

- Only (6/1143) 0.5% of wines had a quality score of 3
- Only (16/1143) ~1% of wines had a quality score of 8
- Only (33/1143) ~3% of wines had a quality score of 4


Examining the trends when grouped by quality score:

<img width="1092" alt="Group by QScore" src="https://user-images.githubusercontent.com/88804543/154178843-34e1d2f3-8491-411a-99d1-e3ea16b052e1.png">

The trends observed from this table:
- Volatile Acidity decreases with increasing quality score 
- Citric Acid increases with increasing quality score
- Sulphates increases with increasing quality score
- Alcohol increases with increasing quality score

Taking a closer look at: volatile acidity, citric acid, sulphates

<img width="973" alt="Quality1" src="https://user-images.githubusercontent.com/88804543/154179178-5713213c-0599-4c52-81d8-a5e9218a0160.png">


Taking a closer look at: chlorides, alcohol, pH

<img width="948" alt="Quality2" src="https://user-images.githubusercontent.com/88804543/154179208-a2780459-c9db-4aa1-8a34-a73ccd99a0d6.png">


#### Principal Component Analysis (PCA)
PCA confirms the Volatile Acidity, Citric Acid, Sulphates, Alcohol metrics together account for 99% of the variance

<img width="448" alt="Screen Shot 2022-02-15 at 11 24 54 AM" src="https://user-images.githubusercontent.com/88804543/154134246-ac22e71b-dca0-4788-b7db-29d03f25ad93.png">

Here is the updated DataFrame, with the highest weighted metrics: Volatile Acidity, Citric Acid, Sulphates, Alcohol and quality (which is needed as the output).

<img width="446" alt="Reduced DF" src="https://user-images.githubusercontent.com/88804543/154179638-456ccb74-26dc-4c93-a857-7784cb9d89e4.png">


## Analysis: Logistic Regression, Support-Vector Machines (SVM), Random Forest Classifier

### Logistic Regression
Supervised learning can be divided into regression and classification. Regression is used to predict continuous variables. Whereas classification is used to predict discrete outcomes. For this wine analysis, the target variable (what we're trying to predict), is a quality score -- which is a continuous variable. The wine quality score is a numerical value within a given range, making it a continuous variable. In both classification and regression issues, the data is divided into features and targets. Features are the variable used to inform the prediction. The target/output is the predicted outcome.



### Random Forest Classifier
Random forest classifiers are a type of ensemble learning model that combines multiple smaller models into a more robust and accurate model. Random forest models use a number of weak learner algorithms (decision trees) and combine their output to make a final classification (or regression) decision. Structurally, random forest models are very similar to neural networks. Random forest models have been popular in machine learning algorithms for many years because of their scalability and robustness. Both output and feature selection of random forest models are easy to interpret. Random forest models can also easily handle outliers and nonlinear data.

The n_estimators will allow us to set the number of trees that will be created by the algorithm. Generally, the higher number makes the predictions stronger and more stable, but can slow down the output because of the higher training time allocated. The best practice is to use between 64 and 128 random forests. This analysis used 128.


### Random Forest Classifer vs Neural Networks
Random forest models will only handle tabular data, so data such as images or natural language data cannot be used in a random forest without heavy modifications to the data. Neural networks can handle all sorts of data types and structures in raw format or with general transformations (such as converting categorical data).

In addition, each model handles input data differently. Random forest models are dependent on each weak learner being trained on a subset of the input data. Once each weak learner is trained, the random forest model predicts the classification based on a consensus of the weak learners. In contrast, deep learning models evaluate input data within a single neuron, as well as across multiple neurons and layers.

As a result, the deep learning model might be able to identify variability in a dataset that a random forest model could miss. However, a random forest model with a sufficient number of estimators and tree depth should be able to perform at a similar capacity to most deep learning models.

## Results

The principal component analysis (PCA) confirmed the four most important metrics for wine likeability are: volatile acidity, citric acid, sulphates, alcohol quantities. The most important of those four is alcohol quantity.

### Random Forest Classifer

The Random Forest Classifier Accuracy Score was 98.5%


















