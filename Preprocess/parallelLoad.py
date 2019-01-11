import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import os

# Helper function that we are going to paralelize
def read_csv(filename):
    'convert a filename into a pandas df'
    #file_route = '../data/traditionalSpamBotsChunks1/' + str(filename)
    print('Loading the file: ', filename)

    data = pd.read_csv(filename)
    return data#pd.read_csv(file_route)

def parallelLoad(filesRoute):
    #data = pd.read_csv('../data/tweetsBotsChunks/tweetsBots_chunk3.csv')
    # Set up the pool
    pool = Pool(processes=8)

    # get a list of file names
    files = os.listdir(filesRoute)
    file_list = [filename for filename in files if filename.split('.')[1]=='csv']
    # Modify to add the correct route to the files
    file_list = [str(filesRoute) +  filename for filename in files]
    print('Starting the pooling...')
    # Have the pool map the file names to dfs
    df_list = pool.map(read_csv, file_list)
    # Combine the DF
    combined_df = pd.concat(df_list, ignore_index=True)
    # Close the pool of created processes to we clean up resources
    pool.close()
    return combined_df

# Run the main process
if __name__ == '__main__':
    result = parallelLoad('../data/traditionalSpamBotsChunks1')
    print('Done!')
    print(result.head(5))
