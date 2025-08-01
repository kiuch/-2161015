from gensim.models import Word2Vec
import MeCab


# MeCabのTokenizer初期化
m = MeCab.Tagger("-Owakati")

# サンプルの文章データ
raw_sentences = [
    "猫は外を歩いている。",
    "夏の夜は暑くて寝苦しい。",
    "彼はりんごとオレンジを食べた。",
    "電車は時間通りに到着した。",
    "子供たちは公園で遊んでいる。"
]

# 文章を単語に分割
sentences = [m.parse(sentence).strip().split() for sentence in raw_sentences]

# Word2Vecモデルの初期化とトレーニング
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2)

# モデルの保存
model.save("word2vec_example.model")

# 保存したモデルのロード
model = Word2Vec.load("word2vec_example.model")

# 似ている単語を取得
similar_words = model.wv.most_similar("猫", topn=5)
print(similar_words)