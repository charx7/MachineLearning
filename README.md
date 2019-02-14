# Welcome to BotDetector3000
Machine Learning project for RUG-ml-2018

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
