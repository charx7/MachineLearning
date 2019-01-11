import pandas as pd
import sys

sys.path.append('../Preprocess')
from parallelLoad import parallelLoad

if __name__ == '__main__':
    result = parallelLoad('../data/traditionalSpamBotsChunks1/')
    print('Done!')
    print(result.head(20))
