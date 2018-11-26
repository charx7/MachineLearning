class BotClassifier(object):
    # Class constructor
    def __init__(self, trainData, method = 'tf-idf'):
        #self.tweets, self.bot = trainData['text'], trainData['bot']
        self.method = method

    def train(self):
        self.tokenize()
        print('im doing training!')

    def tokenize(self):
        print('Im doing tokenize')

# Test the implementation of the BotClassifier Class
bot3000 = BotClassifier('train', 'tf-idf')
bot3000.train()
