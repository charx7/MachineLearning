import pandas as pd

def joinData(botData, genuineData):
    # assumes equiality on our columns
    print('##############')
    print('Start Join')
    print('##############')
    # Label them as bots or non-bots
    botData['bot'] = 1
    genuineData['bot'] = 0

    # Check the head of our data
    print('The columns are: ', list(genuineData.columns.values))
    print('The columns are: ', list(botData.columns.values))

    # Data Join
    joinData = [botData, genuineData]
    joinedDataframe = pd.concat(joinData)
    print('Head of joined dataz \n', joinedDataframe.head())
    totalRowsBots = botData.shape
    print('Lenght of the bot Data is: ', totalRowsBots[0])

    totalRowsGenuine = genuineData.shape
    print('Lenght of the genuine Data is: ', totalRowsGenuine[0])

    totalRowsJoined = joinedDataframe.shape
    print('Lenght of the genuine Data is: ', totalRowsJoined[0])

    counts = joinedDataframe['bot'].value_counts()
    print('The number of tweets is: \n 1 for Bot \n 0 for Human. \n', counts)

    # Clear some memory
    del botData
    del genuineData

    print('##############')
    print('End Join')
    print('##############')

    return joinedDataframe

# botData = pd.read_csv("../data/tweetsBots.csv")
# genuineData = pd.read_csv("../data/tweetsGenuine.csv")
#
# df = JoinData(botData, genuineData)
