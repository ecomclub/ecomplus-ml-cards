import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas.plotting
from surprise import SVD, Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import cross_validate
import csv
import sys
import time


begin = time.time() #run time calculation

# -- data analysis
df= pd.read_csv(sys.argv[1],encoding='iso-8859-1')
df.head()

# -- Let's inspect the pattern of how to use the user cards.

df.groupby('authentication_id').agg({'position':[np.size,np.mean]}).describe()

# -- Let's inspect the pattern of how the movies are rated.

df.card_id.value_counts()[:40]
df.groupby('card_id').agg({'position':[np.size,np.mean]}).describe()
df.groupby('card_id').agg({'position':[np.size,np.mean]}).sort_values([('position','mean')],ascending=False)[:5]

# -- Let's only consider cards with at least 100 number of ratings.

at_least = df.groupby('card_id').agg({'position':[np.size,np.mean]})['position']['size'] >= 100
df.groupby('card_id').agg({'position':[np.size,np.mean]})[at_least].sort_values([('position','mean')],ascending=False)[:5]

# -- end data analysis

# -- collaborative filtering (CF)

reader = Reader(rating_scale=(0,40)) # number of cards
data = Dataset.load_from_df(df[['authentication_id','card_id','position']], reader=reader) #read data frame

# -- Let's start with kNN using cosine similarity.

algo_kNN = KNNBasic(sim_options = {'name':'cosine', 'user_based': False})
cross_validate(algo_kNN, data, measures=['RMSE','MAE'], cv = 5, verbose = True)

algo_SVD = SVD()
cross_validate(algo_SVD, data, measures=['RMSE','MAE'], cv = 5, verbose = True)

# -- Let's switch to matrix factorization. One model is the single value decomposition (SVD).

trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
algo = SVD()

# -- We can use SVD to predict the ratings for the other cards that user has not seen,
#    and output the few cards with highest position, hence the birth of our system recommendation.


algo.fit(trainset)
prediction = algo.test(testset)
prediction[:3]
accuracy.rmse(prediction, verbose=True)

n = int(sys.argv[2]) # parameter of how many recommendations we want.

top_n = defaultdict(list) # Create a pseudo list
for uid, iid, r_ui, est, _ in prediction:
    top_n[uid].append((iid,est))
for uid, user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1], reverse = True)
    top_n[uid] = user_ratings[:n]

# -- print the data in csv file

weight =40 # card size initial
with open(sys.argv[3], 'w') as csvfile: 
    writecsv = csv.writer(csvfile)
    writecsv.writerow(["authentication_id","card_id","position"])

    for uid, user_ratings in top_n.items():
            list_prediction=[iid for (iid, _) in user_ratings]
            for row in list_prediction:
                writecsv.writerow([uid,row,weight])
                weight = weight -1
            del list_prediction[:]
            weight=40

end = time.time()  #end run time calculation and print
print ("Time Execution: ", end-begin)
