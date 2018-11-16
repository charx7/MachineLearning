print("Working")

# imports
import json
import re
# pip install tweepy
import tweepy

auth = tweepy.OAuthHandler('DcjYgQqiJSf0QlkkjGEbM1Kvi', 'eVmLjkwwE6EnCrx5RLCqgNgHorn1idPH8PJ7rYYe0mbmT5NmxS')
auth.set_access_token('2402501288-sRjPSidTnDH5VTgXEDVwCy7SahqI1yM12IDhcKZ', 'GimLtZY68gQh5eAnsvg53ba1uzxk7dg33foxmaoEjJMlo')

api = tweepy.API(auth)

def clear_tweet(tweet, toRemove):
    return tweet.replace(toRemove, "")

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0

    def on_status(self, status):
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

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(track=['trump'], languages=["en"], async=True)
