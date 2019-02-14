#!/user/bin/env python3
# author : Haibin
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#

import sys
sys.path.append('../Preprocess')
from dataJoin import joinData
from parallelLoad import parallelLoad
from preprocess import CustomAnalyzer, doFreq
from Label_Onehot_Encoding_module import LabelEncoder_OneHotEncoder
from Binary_encoding_module import BiEncoder


from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt
from sklearn.preprocessing import StandardScaler

# Join Data

tqdm.pandas()
print('Loading data...')

botData = pd.read_csv('../data/datasets_full.csv/traditional_spambots_1.csv/tweets.csv', index_col=0)
genuineData = pd.read_csv('../data/datasets_full.csv/genuine_accounts.csv/tweets.csv', index_col=0)

# Joining data

print('Joining data...')

seed = 42
df = joinData(botData.sample(20000, random_state = seed), genuineData.sample(20000, random_state = seed))

# See how many tweets we read

print("Read {0:d} tweets".format(len(df)))

print('Origin data shape:', df.shape)

print('Origin data columns:\n', list(df.columns))

#

X_OneHot=LabelEncoder_OneHotEncoder(df)
X_Binary=BiEncoder(df)
y = df['bot']

# The process for OneHot

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_OneHot,y,test_size=0.33,random_state=42)

# Processing with Normalization

# normalize the features based on the mean and standard deviation of each column.

Scaler = StandardScaler()

Scaler.fit(X_train1) # find scaling for each column that make this zero mean and unit std
                    # the line of code only looks at training data to get mean and std and we can use it to transform new feature data.
Scaled_X_train1 = Scaler.transform(X_train1)

Scaled_X_test1 = Scaler.transform(X_test1) #apply those means and std to the test set.

LR_clf_for_ScaledData = LogisticRegression(penalty='l2',C=1.0,class_weight=None,solver='warn')

LR_clf_for_ScaledData.fit(Scaled_X_train1,y_train1)

y_pred_Scaled = LR_clf_for_ScaledData.predict(Scaled_X_test1)

acc_Scaled = mt.accuracy_score(y_test1,y_pred_Scaled)
print('Accuracy_Scaled_OneHot', acc_Scaled)

weights_Scaled = LR_clf_for_ScaledData.coef_.T # take transpose to make a column vector

variable_names = X_OneHot.columns

#for coef, name in sorted(zip(weights_Scaled,variable_names)):

    #print(name,'has weight of', coef[0])

# The process for BinaryEncoder

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_Binary,y,test_size=0.33,random_state=42)

# Processing with Normalization

Scaler2 = StandardScaler()

Scaler2.fit(X_train2)

Scaled_X_train2 = Scaler2.transform(X_train2)

Scaled_X_test2 = Scaler2.transform(X_test2) #apply those means and std to the test set.

LR_clf_for_ScaledData_Bi = LogisticRegression(penalty='l2',C=1.0,class_weight=None,solver='warn')

LR_clf_for_ScaledData_Bi.fit(Scaled_X_train2,y_train2)

y_pred_Scaled_Bi = LR_clf_for_ScaledData_Bi.predict(Scaled_X_test2)

acc_Scaled_Bi = mt.accuracy_score(y_test2,y_pred_Scaled_Bi)
print('Accuracy_Scaled_Binary', acc_Scaled_Bi)

# Drawing the accuracy graph

n_groups=2
accuracy_values=(0.9816,0.9801)
accuracy_values_2=(0.8669,0.8403)
fig,ax = plt.subplots()

index=np.arange(n_groups)
bar_width = 0.3

opacity = 0.7

error_config = {'ecolor':'0.8'}

rects1 = ax.bar(index,accuracy_values,bar_width,alpha=opacity,color='b',error_kw = error_config)
#rects1 = ax.bar(index,accuracy_values,bar_width,alpha=opacity,color='b',error_kw = error_config)
rects2 = ax.bar(index+bar_width,accuracy_values_2,bar_width,alpha=opacity,color='r',error_kw = error_config)
#rects2 = ax.bar(index+bar_width,accuracy_values_2,bar_width,alpha=opacity,color='r',error_kw = error_config)
#ax.set_xlabel('The results of different experiments',fontsize=18)
ax.set_ylabel('Accuracy Scores',fontsize=18)
ax.set_xlim(left=-0.5, right=2)
ax.set_title('Accuracy w.r.t. encoding method',fontsize=24, fontweight='bold')
ax.set_xticks(index + bar_width/2)
ax.set_xticklabels(('Decission Tree', 'Logistic Regression'),fontsize=18)

ax.legend((rects1[0], rects2[0]),('Binary','One-Hot'),fontsize=18,prop={'size':18})

#ax.legend((rects1[0], rects2[0]))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,'%f' % height,ha='center', va='top')

autolabel(rects1)
autolabel(rects2)


fig.tight_layout()
plt.show()




Drawing the weight graph

plt.style.use('ggplot')
Drawing_Weights = pd.Series(LR_clf_for_ScaledData.coef_[0],index=X_OneHot.columns)
Drawing_Weights.plot(kind='bar')
plt.show()

