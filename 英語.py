from gensim.models import Word2Vec
import re

# 例として使うテキストデータ
text_data = [
    "I love to eat apples and bananas.",
    "He likes to play soccer and basketball.",
    "She enjoys reading books and watching movies.",
    "Apples are red and sweet.",
    "Bananas are yellow and long."
]

# テキストデータを単語のリストに前処理する関数
def preprocess_text(text):
    # 小文字に変換し、句読点を除去して単語に分割
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # 英数字とスペース以外を除去
    return text.split()

# 訓練用のデータを作成
sentences = [preprocess_text(sentence) for sentence in text_data]

print("訓練データ（前処理後）:")
for s in sentences:
    print(s)
    # Word2Vecモデルを訓練
model = Word2Vec(
    sentences,
    vector_size=100,  # ベクトルの次元数
    window=5,         # 周囲の単語の範囲
    min_count=1,      # 出現回数が1未満の単語は無視
    sg=0,             # CBOW (Continuous Bag of Words) を使用
    epochs=100        # エポック数
)

print("\nWord2Vec モデルが訓練されました。")
# 単語のベクトルを取得
vector_apple = model.wv['apples']
print(f"\n'apples' のベクトル（最初の5次元）: {vector_apple[:5]}...")

# 類似度の高い単語を検索
print("\n'apples' に最も類似する単語:")
similar_words_apple = model.wv.most_similar('apples', topn=3)
for word, similarity in similar_words_apple:
    print(f"  {word}: {similarity:.4f}")

print("\n'soccer' に最も類似する単語:")
similar_words_soccer = model.wv.most_similar('soccer', topn=3)
for word, similarity in similar_words_soccer:
    print(f"  {word}: {similarity:.4f}")

# 2つの単語間の類似度を計算
similarity_apple_banana = model.wv.similarity('apples', 'bananas')
print(f"\n'apples' と 'bananas' の類似度: {similarity_apple_banana:.4f}")

similarity_apple_soccer = model.wv.similarity('apples', 'soccer')
print(f"'apples' と 'soccer' の類似度: {similarity_apple_soccer:.4f}")

# 特定の単語がモデルの語彙に含まれているか確認
print(f"\n'books' がモデルの語彙にあるか: {'books' in model.wv.key_to_index}")
print(f"'train' がモデルの語彙にあるか: {'train' in model.wv.key_to_index}")

# 特定の単語から別の単語を引いて、新しい単語を足す（例: 'king' - 'man' + 'woman' = 'queen'）
# 今回のデータセットでは単語数が少ないため、一般的な例の再現は難しいですが、概念を示すために記載
# if 'king' in model.wv and 'man' in model.wv and 'woman' in model.wv:
#     result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
#     print(f"\n'king' - 'man' + 'woman' の結果: {result}")