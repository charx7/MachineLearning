# To import the twitter API env variables
import os

def getKeys():
    # Get the key from os.getenv
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET_KEY = os.getenv('TWITTER_SECRET_API_KEY')

    #print('The twitter api key is: ', TWITTER_API_KEY)
    #print('The twitter secret api key is: ', TWITTER_SECRET_API_KEY)

    return TWITTER_API_KEY, TWITTER_SECRET_API_KEY
