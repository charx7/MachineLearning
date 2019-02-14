# Welcome to BotDetector3000
Machine Learning project for RUG-ml-2018

## Data
The scripts will look for data files in specific directories. 
The path "Data/preprocessedTweets" contains the English-filtered tweets which are used for the NLP approaches (both the models and the embeddings' training). 
The path "Data/...." should contain the complete spambot and genuine tweet .csv files for the decision trees to run.

## You need to have anaconda or miniconda installed
[Miniconda](https://conda.io/en/latest/miniconda.html)

[Anaconda](https://www.anaconda.com/distribution/)

## To install the gpu environemnt run:
```bat
conda env create -f gpu-environment.yml
```
## Or for the CPU one:
```bat
conda env create -f environment.yml
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
  * run the model:
    ```bat
    python decision_tree.py 
    ```
    or the model by using scikit learn library:
    ```bat
    python skDecisionTree.py 
    ```
