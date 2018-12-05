import pandas as pd

def joinData(botData, genuineData):
    # assumes equality on our columns
    print('-------Start Join-------')
    # Label them as bots or non-bots
    botData['bot'] = 1
    genuineData['bot'] = 0

    # Check the head of our data uncomment for debugz
    #print('The columns are: ', list(genuineData.columns.values))
    #print('The columns are: ', list(botData.columns.values))

    # Data Join
    joinData = [botData, genuineData]
    joinedDataframe = pd.concat(joinData)
    #print('Head of joined dataz \n', joinedDataframe.head())
    totalRowsBots = botData.shape
    #print('Lenght of the bot Data is: ', totalRowsBots[0])

    totalRowsGenuine = genuineData.shape
    #print('Lenght of the genuine Data is: ', totalRowsGenuine[0])

    totalRowsJoined = joinedDataframe.shape
    #print('Lenght of the Final Data is: ', totalRowsJoined[0])

    counts = joinedDataframe['bot'].value_counts()
    #print('The number of tweets is: \n 1 for Bot \n 0 for Human. \n', counts)

    # Clear some memory
    del botData
    del genuineData

    # Re index to avoid duplicates
    joinedDataframe = joinedDataframe.reset_index(drop=True)
    print('-------End Join-------\n')

    return joinedDataframe

# botData = pd.read_csv("../data/tweetsBots.csv")
# genuineData = pd.read_csv("../data/tweetsGenuine.csv")
#
# df = JoinData(botData, genuineData)
