# Welcome to BotDetector3000
Machine Learning project for RUG-ml-2018

## Data
The complete **cresci-2017** data set can be downloaded from the [Bot Repository](https://botometer.iuni.iu.edu/bot-repository/datasets.html). A filtered version used in the NLP approach has been uploaded to the repository.

The Stanford GloVe embeddings can be downloaded from [here](https://nlp.stanford.edu/projects/glove/). 

The scripts will look for data files in specific directories.
* The English-filtered tweets, which are used for the NLP approaches (both the models and the embeddings' training), should be placed in the path "data/preprocessedTweets/".
* The GloVe embeddings should be placed in the path "data/gloveEmbeds/".
* The complete spambot and genuine tweets.csv files, used in the decision trees, should be placed in the path "data/datasets_full.csv/traditional_spambots_1.csv/" and "data/datasets_full.csv/genuine_accounts.csv/".

## You need to have anaconda or miniconda installed
[Miniconda](https://conda.io/en/latest/miniconda.html)

[Anaconda](https://www.anaconda.com/distribution/)

## To install the GPU environemnt for the NLP approach, run:
```bat
conda env create -f gpu-environment.yml
```
Prerequisites for using the GPU environment can be found [here](https://www.tensorflow.org/install/gpu), along with a list with compatible GPUs. 

**Note: this is an environment for Windows! Using it in another OS can lead to compatibility issues.**

## The environment for running the decision tree model:
```bat
conda env create -f cpu-environment.yml
```

## Running the code
It is advised to run the code while in the proper directories, as indicated in the parentheses.
* tf-idf: 
  * To test the Naive Bayes (Models) model using tf-idf run: 
    ```bat
    python sklearnNB.py 
    ```
  * To test the Support Vector Model (Models) model using tf-idf run: 
    ```bat
    python svm.py 
    ```
* word embeddings:
  * Custom embeddings 
    * first run the training (Preprocess): 
      ```bat
      python word2vec.py 
      ```
    * then run the model/-s (Models): 
      ```bat
      python sklearnNBEmbeded.py 
      ```
      or
      ```bat
      python svmEmbeded.py 
      ```
  * GloVe embeddings:
    * run the model/-s (Models):
      ```bat
      python sklearnNBEmbededGlove.py 
      ```
      or
      ```bat
      python svmEmbededGlove.py 
      ```
* Decision trees:
   * run the logistic regression model (FeatureSelection) to show the feature importance and two performances with two encodings:
    ```bat
    python Logistic_Regression_for_weights.py 
    ```
    * run SkDecision Tree model (FeatureSelection) to show the accuracy with OneHot and Binary encoding:
    ```bat
    python Decision_tree_under_two_encoding.py
    ```
  
  * run the decision tree model (FeatureSelection) which is implemented step by step:
    ```bat
    python decision_tree.py 
    ```
    or the model (FeatureSelection) by using scikit learn library:
    ```bat
    python skDecisionTree.py 
    ```
