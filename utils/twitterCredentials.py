# To import the twitter API env variables
import os

def getKeys():
    # Get the key from os.getenv
	TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
	TWITTER_API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
	TWITTER_API_ACCESS_TOKEN = os.getenv('TWITTER_API_ACCESS_TOKEN')
	TWITTER_API_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_API_ACCESS_TOKEN_SECRET')
	return TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_API_ACCESS_TOKEN, TWITTER_API_ACCESS_TOKEN_SECRET
