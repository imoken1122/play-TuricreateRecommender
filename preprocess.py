import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import turicreate as tc
def load():
    rait = pd.read_csv("rating.csv")
    anime = pd.read_csv("anime.csv")
    user_ids_count = Counter(rait.user_id)
    anime_ids_count = Counter(rait.anime_id)

    # 20作品以上を評価したuserを残す
    n = sum(np.array(list(user_ids_count.values()) ) > 100) # 30000
    user_ids = [u for u,c in user_ids_count.most_common(n)]
    # 50回以上を評価されたanimeを残す
    m = sum(np.array(list(anime_ids_count.values()) ) > 200) # 4625
    anime_ids = [u for u,c in anime_ids_count.most_common(m)]

    rait_sm = rait[rait.user_id.isin(user_ids) & rait.anime_id.isin(anime_ids)]
    merge_rait = rait_sm.merge(anime, left_on="anime_id",right_on="anime_id", how = "left")


    map_user_id = {}
    for i, u_id in enumerate(user_ids):
        map_user_id[u_id] = i
    map_anime_id = {}
    for i, a_id in enumerate(anime_ids):
        map_anime_id[a_id] = i


    merge_rait.loc[:, 'user_id'] = merge_rait.progress_apply(lambda x: map_user_id[x.user_id], axis=1)
    merge_rait.loc[:, 'anime_id'] = merge_rait.progress_apply(lambda x: map_anime_id[x.anime_id], axis=1)

    return merge_rait

def complement(merge_rait):

    def func(x):
        return 10/(1+ np.exp(-0.76*x + 5))
    merge_rait_ = merge_rait[merge_rait.rating_x != -1]
    lack_data = merge_rait[merge_rait.rating_x == -1][["user_id","anime_id"]]

    sfd = tc.SFrame(merge_rait_[["user_id","anime_id","rating_x"]])
    m = tc.factorization_recommender.create(sfd, "user_id","anime_id",target = "rating_x")


    pred= lack_data.progress_apply(lambda x :m.predict({"user_id":x.user_id,"anime_id":x.anime_id})[0],axis=1)
    merge_rait.loc[lack_data.index, "rating_x"] = func(pred.values).astype(int)

    merge_rait.to_csv("data.csv",index = False)

if __name__ == "__main__":
    complement(load())