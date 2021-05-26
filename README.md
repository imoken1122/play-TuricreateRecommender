# __turicreate とは__
***

<br>

turicreate は Apple の機械学習のライブラリで, 古典的手法から深層学習を用いたタスクまでいくつかカバーされており, sklearnのように手軽に学習させることができます. 


[https://apple.github.io/turicreate/docs/api/index.html:embed:cite]


<br>

最初にturicreateでレコメンドを行う方法を書いておきます.   
ちなみにこの[チュートリアル](https://apple.github.io/turicreate/docs/userguide/recommender/) がわかりやすいです

__turicreateのインストール__

```
pip install turicreate
```

<br>

__レコメンダーを作成__

hoge.csv には [user_id , item_id, rating] のカラムがあるとすると, 以下で学習できます.

```python
import pandas as pd
import turicreate as tc
data = pd.read_csv("hoge.csv") 
sfd = tc.SFrame(data)
model = tc.factorization_recommender.create(sfd, "user_id","item_id",target = "rating")
```

<br>

(1) あるユーザにレコメンドしたい時と,  (2) あるアイテムと類似するアイテムをレコメンドする時は以下のように書きます


```python

rec_for_user = model.recommend([<任意のuser_id>]) # (1)

sim_item = model.get_similarity_item([<任意のitem_id>]) #  (2)

```

<br>


# レコメンドに用いるデータセット
***

<br>

Kaggleに置いてある約700万件のユーザレビューのデータを用います.

[https://www.kaggle.com/CooperUnion/anime-recommendations-database:embed:cite]

<br>

以下は各ファイルのカラム情報です. 

__rating..csv__

```
user_id, anime_id, rating (-1 , 0 ~ 10)
```
[f:id:kenzo1122:20210525215240p:plain:w200:h250]

<br>

__anime.csv__
```
anime_id, name, genre , type, episodes, rating, members 
```
[f:id:kenzo1122:20210525215246p:plain]

<br>

今回学習に用いるデータは,  rating.csv と, anime.csv[ anime_id, name]  とします. 

<br>

# データ前処理
***

<br>

最初にこの章のコードを置いておきます


[https://github.com/imoken1122/turicreate-AnimeRecommender/blob/main/preprocess.py]



#### 読み込み

```python
import pandas as pd
import numpy as np
from tqdm._tqdm_notebook import tqdm_notebook
from collections import Counter
tqdm_notebook.pandas(desc=" progress: ")
rait = pd.read_csv("rating.csv")
anime = pd.read_csv("anime.csv")
```
<br>

#### 評価に参加した数が少ないuser と 評価された数が少ないアニメを削除
```python
# 各userが出現する回数, 各animeが出現する回数
user_ids_count = Counter(rait.user_id)
anime_ids_count = Counter(rait.anime_id)

# 20作品以上を評価したuserを残す
n = sum(np.array(list(user_ids_count.values()) ) > 80) # 30000
user_ids = [u for u,c in user_ids_count.most_common(n)]
# 50回以上を評価されたanimeを残す
m = sum(np.array(list(anime_ids_count.values()) ) > 100) # 4625
anime_ids = [u for u,c in anime_ids_count.most_common(m)]

rait_sm = rait[rait.user_id.isin(user_ids) & rait.anime_id.isin(anime_ids)]
```

<br>

#### anime_id で rating.csv と anime.csv をマージして, indexふりなおし
```python

# マージ
merge_rait = rait_sm.merge(anime, left_on="anime_id". right_on ="anime_id", how = "left")

# index ふりなおしの辞書作成
map_user_id = {u_id:i for i, u_id in enumerate(user_ids)}
map_anime_id = {a_id:i for i, a_id in enumerate(anime_ids)}

# 各々indexふりなおし
merge_rait.loc[:, 'user_id'] = merge_rait.progress_apply(lambda x: map_user_id[x.user_id], axis=1)
merge_rait.loc[:, 'anime_id'] = merge_rait.progress_apply(lambda x: map_anime_id[x.anime_id], axis=1)

```

<br>

#### rating = -1  の評価値を matrix facrization で予測して埋める

rating が -1 の対処法ですが,  その行を削除する, そのuserの評価の平均値, そのアニメの評価平均値で埋めるなどあると思います.   
ここでは, rating が -1 以外のデータを用いて turicreate の [matrix factorization](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.factorization_recommender.create.html#turicreate.recommender.factorization_recommender.create) で学習します.  そして, 評価値が-1である行のユーザがあるアニメにどんな点数をつけるか予測させます.   
 
また評価予測値が 0~ 10 の範囲に必ずにも予測されないかつ多少評価値が高く予測されていたため, 以下の`func(x)`によって, 調整しました. 

```python

import pandas as pd
import turicreate as tc

def func(x):
    return 10/(1+ np.exp(-0.76*x + 5))

# rating が -1 以外のデータ
merge_rait_ = merge_rait[merge_rait.rating_x != -1] 
sfd = tc.SFrame(merge_rait_[["user_id","anime_id","rating_x"]])

#学習
m = tc.factorization_recommender.create(sfd, "user_id","anime_id",target = "rating_x") # matrix factorization

# -1 のみのデータ
lack_data = merge_rait[merge_rait.rating_x == -1][["user_id","anime_id"]]  rating が -1 のデータ

# ratingが-1であるuser_id, anime_id の組み合わせで rating を予測
pred= lack_data.progress_apply(lambda x :m.predict({"user_id":x.user_id,"anime_id":x.anime_id})[0],axis=1) 

 # 埋める
merge_rait.loc[lack_data.index, "rating_x"] = func(pred.values).astype(int)

# 保存
merge_rait.to_csv("data_comlement.csv",index = False) 
```


<br>


このデータを保存し, 前処理は終わりです.

<br>


# レコメンダーを作成
***


<br>

最初にこの章のコードを置いておきます


[https://github.com/imoken1122/turicreate-AnimeRecommender/blob/main/recommender.py]




<br>

レコメンドエンジンを作成にあたって, [ranking_factorization_recommender](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.ranking_factorization_recommender.create.html#turicreate.recommender.ranking_factorization_recommender.create) と  [factorization_recommender](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.factorization_recommender.create.html#turicreate.recommender.factorization_recommender.create) で学習させてみて, RMSEやレコメンドされるアニメを比較したいと思います. 

#### データ読み込み

ここで, turicreate は学習データがstring型であっても学習可能で,  この後レコメンドする際の入力が少し楽なので anime_id ではなく name をカラムに入れて学習させます.

```python

import pandas as pd

raw= pd.read_csv("data_comlement.csv")
data = raw[["user_id","name","rating_x"]]
sfd = tc.SFrame(data)


```

<br>

#### データ分割

rating が 7 以上と7未満のデータをわけ, 7以上のデータを学習データ(train)とテストデータ(test)に分割し, 学習データ(train)に 7未満のデータを追加しました. 

```python
high_rated_data = sfd[sfd["rating_x"] >= 7]
low_rated_data = sfd[sfd["rating_x"] < 7]
train_data_1, test_data = tc.recommender.util.random_split_by_user(
                                    high_rated_data, user_id='user_id', item_id='name')
train_data = train_data_1.append(low_rated_data)

```

#### 学習

2つのアルゴリズムで学習します. 

```python
rankmf = tc.ranking_factorization_recommender.create(train_data, "user_id", "name", target='rating_x',)
mf = tc.factorization_recommender.create(train_data, "user_id", "name", target='rating_x',)
```
<br>

それぞれの学習データでの RMSE は以下の通り. 

```
- ranking_factorization_recommender  =>  training RMSE: 1.0974
- factorization_recommender => training RMSE: 0.97999
```


#### 評価


テストデータによる RMSE を見てみます.   
複数のモデルを評価したいときは `compare_models` という関数を使うのが便利です.


```
tc.recommender.util.compare_models(test_data, [mf,rankmf],model_names = ["mf","rankmf"])
```

出力 MF
```
+--------+----------------------+-----------------------+
| cutoff |    mean_precision    |      mean_recall      |
+--------+----------------------+-----------------------+
|   1    | 0.04300000000000002  | 0.0015764884263909638 |
|   2    |        0.0435        | 0.0027709084607027987 |
|   3    | 0.045333333333333344 |  0.004170689820898463 |
|   4    |        0.047         |  0.005753904113269214 |
|   5    | 0.050400000000000014 |  0.007749015247566327 |
|   6    | 0.05216666666666667  |  0.009662804940866148 |
|   7    | 0.05157142857142859  |  0.011009947706248266 |
|   8    |       0.050375       |  0.012236354681983605 |
|   9    | 0.050222222222222224 |  0.013832997081852426 |
|   10   | 0.05010000000000001  |  0.015225935421589516 |
+--------+----------------------+-----------------------+

RMSE: 0.9354762940915811
```

出力 rankMF
```
+--------+---------------------+----------------------+
| cutoff |    mean_precision   |     mean_recall      |
+--------+---------------------+----------------------+
|   1    |        0.486        | 0.01853521300202965  |
|   2    |        0.4425       | 0.03324023400512996  |
|   3    |  0.4143333333333334 | 0.04601403899744517  |
|   4    |       0.38875       | 0.056825576739042816 |
|   5    | 0.37179999999999985 | 0.06670896395476202  |
|   6    |  0.3601666666666669 | 0.07716049263294347  |
|   7    |  0.3497142857142858 | 0.08786837246193445  |
|   8    |       0.33925       | 0.09696057462761283  |
|   9    |  0.3337777777777778 | 0.10690062267355312  |
|   10   |        0.3262       | 0.11585739549254023  |
+--------+---------------------+----------------------+
RMSE: 1.319034169311567
```

<br>

テストデータに対する RMSE は MF の方が良いようです.     

表として出力された評価は何に対しての評価なのかよくわかりませんが, precision や recall から単語からは, おそらくテストデータに含まれるユーザにもっともらしいアニメをレコメンドできてるか, という評価だと思われます.   

表の precision の数値を見る限り, MFよりrankMFは高評価でレコメンドしたアニメが実際に高評価だった数が多いということなので, 無難なアニメがレコメンドされると予想できます.


<br>

# 類似アニメをレコメンドさせてみる


ranking_factorization_recommender と factorization_recommender によるレコメンドされるアニメを見てみます. そのためには, [get_similar_items](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.get_similar_items.html#turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.get_similar_items)関数を用います. (matrix factorizationされた後のアイテム潜在ベクトル間のコサイン類似度によってレコメンドされるアイテムが選ばれます. )

<br>

ガールズパンツァーに類似してるアニメをそれぞれのレコメンダーで検索してみます. 

```python
similar_items = mf.get_similar_items(['Girls und Panzer']) 
similar_items_ = rankmf.get_similar_items(['Girls und Panzer']) 
```
出力
```
- MF 
+-----------------------------+--------------------+------+
|           similar           |       score        | rank |
+-----------------------------+--------------------+------+
|         Yuru Yuri♪♪         | 0.9631929993629456 |  1   |
|   Yuru Yuri Nachuyachumi!   | 0.9609564542770386 |  2   |
|      Minami-ke Tadaima      | 0.9535280466079712 |  3   |
|          Yuru Yuri          | 0.9530896544456482 |  4   |
|           Love Lab          | 0.9445452094078064 |  5   |
|      Yuru Yuri San☆Hai!     | 0.9436423182487488 |  6   |
|        The iDOLM@STER       | 0.9122692346572876 |  7   |
|    Shinryaku!? Ika Musume   | 0.9116688966751099 |  8   |
|        Non Non Biyori       | 0.9070506691932678 |  9   |
| Gochuumon wa Usagi Desu ka? | 0.9044833779335022 |  10  |
+-----------------------------+--------------------+------+

- rankMF
+-------------------------------+--------------------+------+
|            similar            |       score        | rank |
+-------------------------------+--------------------+------+
|   Girls und Panzer Specials   | 0.8605985045433044 |  1   |
| Girls und Panzer: Kore ga ... | 0.8559162020683289 |  2   |
|      Strike Witches Movie     | 0.8132143020629883 |  3   |
| Girls und Panzer: Shoukai ... | 0.765860378742218  |  4   |
|         Kiniro Mosaic         | 0.7395626306533813 |  5   |
|          Yuru Yuri♪♪          | 0.7339246273040771 |  6   |
|   Girls und Panzer der Film   | 0.720527172088623  |  7   |
| Stella Jogakuin Koutou-ka ... | 0.703089714050293  |  8   |
|           Yuyushiki           | 0.7017302513122559 |  9   |
| Strike Witches: Operation ... | 0.6946920156478882 |  10  |
+-------------------------------+--------------------+------+
[10 rows x 4 columns]

```

<br>

結果は, rankMF によってレコメンドされたアニメは, 大半がガルパンの映画やシリーズで, 一方のMFは, ゆるゆりやアイドルマスター,のんのんびよりなど複数人の女の子主役のアニメがレコメンドされているように思えます. (ほとんどみたことがないので調べた)


<br>

次は この素晴らしい世界に祝福を! に類似したアニメを検索します
```python
similar_items = mf.get_similar_items(['Kono Subarashii Sekai ni Shukufuku wo!']) 
similar_items_ = rankmf.get_similar_items(['Kono Subarashii Sekai ni Shukufuku wo!']) 
```

出力
```
- MF

'Shimoneta to Iu Gainen ga Sonzai Shinai Taikutsu na Sekai',
'D-Frag!', 'Danna ga Nani wo Itteiru ka Wakaranai Ken',
'Yahari Ore no Seishun Love Comedy wa Machigatteiru. Zoku OVA',
'Danna ga Nani wo Itteiru ka Wakaranai Ken 2 Sure-me',
'Yahari Ore no Seishun Love Comedy wa Machigatteiru.',
'Sakamoto desu ga?', 
'Hataraku Maou-sama!',
 'Seitokai Yakuindomo*',
'Amagi Brilliant Park'

-rankMF

'Kono Subarashii Sekai ni Shukufuku wo! OVA',
'Netoge no Yome wa Onnanoko ja Nai to Omotta?',
'Hai to Gensou no Grimgar',
'Ore ga Ojousama Gakkou ni &quot;Shomin Sample&quot; Toshite Gets♥Sareta Ken',
'Gate: Jieitai Kanochi nite, Kaku Tatakaeri 2nd Season',
'Dagashi Kashi', 
'Re:Zero kara Hajimeru Isekai Seikatsu',
'Rakudai Kishi no Cavalry', 
'Jitsu wa Watashi wa',
'Shimoneta to Iu Gainen ga Sonzai Shinai Taikutsu na Sekai',
'Gate: Jieitai Kanochi nite, Kaku Tatakaeri'
```

MFによってレコメンドされたアニメは見たことがないためなんとも言えませんが, rankMFの方は ,灰と幻想のグリムガルやリゼロ, Gateといくつか異世界転生系のアニメをレコメンドしています.
  
<br>

全体のコード

[https://github.com/imoken1122/turicreate-AnimeRecommender]



##  デモ

最後に,  React と FastAPI を用いて[デモ](https://arncmd.herokuapp.com/)を作成しましたので遊んでみてください 
(ただ Heroku は30分間アクセスがないとスリープしてしまうため, 最初はフロントからリクエストを送信しても何も返ってこない可能性がありますので何回か送信してみてください.) 



# おまけ
***

<br>

ここで用いたデータセットのアニメタイトルは全て日本語のローマ字表記と英語なので, [2]を参考にスクレイピングでそれらに対応する日本語表記を取得するスクリプトを置いておきます. ( ※ 完璧ではありませんが有名タイトルは大抵日本語表記になります)


[https://github.com/imoken1122/everyones-scraper/blob/main/anime_title_en2ja.py]


<br>

# 参考

[1] [https://apple.github.io/turicreate/docs/api/turicreate.toolkits.recommender]

[2] [https://apple.github.io/turicreate/docs/userguide/recommender]

[3] [https://note.com/npaka/n/naf3b46f598ab?magazine_key=m4c8ee8cad783]







