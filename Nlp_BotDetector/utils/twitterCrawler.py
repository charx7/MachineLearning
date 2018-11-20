# imports
import json
import re
import twitterCredentials
import tweepy # pip install tweepy

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
	def __init__(self, api=None):
		super(MyStreamListener, self).__init__()
		self.num_tweets = 0

	def on_status(self, status):
		print("Taking data")
		# don't get retweets
		if not status.retweeted and 'RT @' not in status.text:
			self.num_tweets += 1
			# stop at 20 tweets
			if self.num_tweets < 20:
				myjson = json.loads(json.dumps(status._json))
				print("======== TWEET ========")
				# get tweet text
				if 'extended_tweet' not in myjson:
					tweet = myjson['text']
				else:
					tweet = myjson['extended_tweet']['full_text']
				# clean tweet from urls
				tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
				print(tweet)
				return True
			else:
				print(self.num_tweets)
				return False

	def on_error(self, status_code):
		if status_code == 420:
			#returning False in on_error disconnects the stream
			return False
			# returning non-False reconnects the stream, with backoff.

	
TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_API_ACCESS_TOKEN, TWITTER_API_ACCESS_TOKEN_SECRET = twitterCredentials.getKeys()
auth = tweepy.OAuthHandler('TWITTER_API_KEY', 'TWITTER_API_SECRET_KEY')
auth.set_access_token('TWITTER_API_ACCESS_TOKEN', 'TWITTER_API_ACCESS_TOKEN_SECRET')
api = tweepy.API(auth)
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(track=['trump'], languages=["en"], async=True)
