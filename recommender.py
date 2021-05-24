
import pandas as pd
import numpy as np
import turicreate as tc


# load
raw= pd.read_csv("data_comlement.csv")
data = raw[["user_id","name","rating_x"]]
sfd = tc.SFrame(data)

# data split
high_rated_data = sfd[sfd["rating_x"] >= 7]
low_rated_data = sfd[sfd["rating_x"] < 7]
train_data_1, test_data = tc.recommender.util.random_split_by_user(
                                    high_rated_data, user_id='user_id', item_id='name')
train_data = train_data_1.append(low_rated_data)


# learning
rankmf = tc.ranking_factorization_recommender.create(train_data, "user_id", "name", target='rating_x',)
mf = tc.factorization_recommender.create(train_data, "user_id", "name", target='rating_x',)


# compare model
tc.recommender.util.compare_models(test_data, [mf,rankmf],model_names = ["mf","rankmf"])


id2anime = {}
for id in np.unique(raw["anime_id"]):
    id2anime[id] = raw[raw["anime_id"] == id]["name"].values[0]

# ガルパン
similar_items = mf.get_similar_items(['Girls und Panzer'],k =10) 
similar_items_ = rankmf.get_similar_items(['Girls und Panzer']) 
print(similar_items, similar_items_)

# このすば
similar_items = mf.get_similar_items(['Kono Subarashii Sekai ni Shukufuku wo!']) 
similar_items_ = rankmf.get_similar_items(['Kono Subarashii Sekai ni Shukufuku wo!']) 
print(similar_items, similar_items_)


# model save
mf.save("MF.model")
rankmf.save("rankMF.model")