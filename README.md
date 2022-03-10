# Wine-Quality-2
A second machine learning wine quality analysis performed as a follow up to the ML-Wine-Quality repository. This is a fresh attempt to improve optimization of the machine learning techniques. A dataset with identical features but slightly different values was used. 

## Project Overview
The goal of this project was to use machine learning techniques: logistic regression, SVM, Random Forest Classifier to predict the quality score (likeability) of various wines. Binning was used to decrease the number of classes of the output variable. A column was added to bin the quality scores into 1 of 3 bins: 1. Bad (Quality Score: 1-3), 2. Average (Quality Score: 4-7), 3. Excellent (Quality Score: 8-10). The data for this project was obtained with free use from [Kaggle](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) through UCI Machine Learning.

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

Columns/Features:
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
13. Reviews -- this column was added to bin the quality score into 1 of 3 categories: 1. Bad (Quality Score: 1-3) 2. Average (Quality Score: 4-7) 3. Excellent (Quality Score: 8-10)

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
PCA analysis shows 8 features together account for 99% of the variance

<img width="615" alt="PCA Plot" src="https://user-images.githubusercontent.com/88804543/157563911-baa1fbab-5772-4904-98e3-529f0641cf18.png">

<img width="537" alt="PCA Explained Variance Ratio" src="https://user-images.githubusercontent.com/88804543/157562537-6245f638-9128-423c-af50-ea55a72e5967.png">

Here are the features sorted in order of greatest importance/weight:

<img width="549" alt="Feature Importance Ranking" src="https://user-images.githubusercontent.com/88804543/157562547-7edfa1a7-efdc-4600-917e-4a49fb2d427f.png">


## Analysis: Logistic Regression, Support-Vector Machines (SVM), Random Forest Classifier

### Logistic Regression
Supervised learning can be divided into regression and classification. Regression is used to predict continuous variables. Whereas classification is used to predict discrete outcomes. For this wine analysis, the target variable (what we're trying to predict), is a quality score -- which is a continuous variable. The wine quality score is a numerical value within a given range, making it a continuous variable. In both classification and regression issues, the data is divided into features and targets. Features are the variable used to inform the prediction. The target/output is the predicted outcome.

<img width="507" alt="Logistic Regression" src="https://user-images.githubusercontent.com/88804543/157564291-6d43ec03-d717-49a4-a8bb-8bc3f0959c51.png">


## Support-Vector Machines (SVM)




<img width="523" alt="SVM" src="https://user-images.githubusercontent.com/88804543/157564307-94aa6087-53e0-4979-9ff5-07cc13ff7f2a.png">


### Random Forest Classifier
Random forest classifiers are a type of ensemble learning model that combines multiple smaller models into a more robust and accurate model. Random forest models use a number of weak learner algorithms (decision trees) and combine their output to make a final classification (or regression) decision. Structurally, random forest models are very similar to neural networks. Random forest models have been popular in machine learning algorithms for many years because of their scalability and robustness. Both output and feature selection of random forest models are easy to interpret. Random forest models can also easily handle outliers and nonlinear data.

The n_estimators will allow us to set the number of trees that will be created by the algorithm. Generally, the higher number makes the predictions stronger and more stable, but can slow down the output because of the higher training time allocated. The best practice is to use between 64 and 128 random forests. This analysis used 128.


### Random Forest Classifer vs Neural Networks
Random forest models will only handle tabular data, so data such as images or natural language data cannot be used in a random forest without heavy modifications to the data. Neural networks can handle all sorts of data types and structures in raw format or with general transformations (such as converting categorical data).

In addition, each model handles input data differently. Random forest models are dependent on each weak learner being trained on a subset of the input data. Once each weak learner is trained, the random forest model predicts the classification based on a consensus of the weak learners. In contrast, deep learning models evaluate input data within a single neuron, as well as across multiple neurons and layers.

As a result, the deep learning model might be able to identify variability in a dataset that a random forest model could miss. However, a random forest model with a sufficient number of estimators and tree depth should be able to perform at a similar capacity to most deep learning models.

<img width="606" alt="RFC Acc Score" src="https://user-images.githubusercontent.com/88804543/157564326-5e9fe299-60b6-48cb-8274-8116f44f2646.png">



## Results

The principal component analysis (PCA) confirmed the eight most important metrics for wine likeability are: volatile acidity, citric acid, sulphates, alcohol quantities. The most important of those eight is alcohol quantity.

The Logistic Regression Accuracy Score was 98.54%

The SVM Accuracy Score was 98.54%

The Random Forest Classifier Accuracy Score was 98.54%


















