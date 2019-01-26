import numpy as np

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def embed_Dataframe_with_Glove(dataframeToEmbed, gloveModel):
    # Create a numpy empty array
    embededData = []
    print('Embedding the Dataframe...')

    for tweet in tqdm(dataframeToEmbed):
        # Define an empty np 0s array
        ohv_test_sent = np.zeros((1,200)) # TODO restore embedding size

        for word in tweet:
            # If we find it on the vocab (exists) then we use its embedding
            if word in gloveModel:
                # Set it to be the embed
                word_vector = gloveModel[word]
            else:
                # Set the unk token
                word_vector = gloveModel['unk']

            # define the embed
            embed = word_vector

            # Sum the embeded vectors to form the embeded sentence
            ohv_test_sent = ohv_test_sent + embed

        # Append to the data we are going to return
        embededData.append(ohv_test_sent)


    npArrayEmbededData = np.array(embededData)
    reshapedNpArray = np.squeeze(npArrayEmbededData, axis=1)

    return reshapedNpArray


if __name__ == '__main__':
        # Load the glove vectors
        gloveModel = loadGloveModel('glove.twitter.27B.50d.txt')
        # test print 'unk' is the UNKNOWN_TOKEN
        print(gloveModel['unk'])
