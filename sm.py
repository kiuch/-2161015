import MeCab # 形態素解析器
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# --- 1. データ準備 ---

# MeCabの辞書パスを明示的に指定（Janomeを使用する場合は不要）
# ★ここをあなたのMeCabインストールパスに合わせてください★
# 例: MeCab本体を C:\MeCab にインストールした場合
#    mecabrcは通常 C:\MeCab\etc\mecabrc または C:\MeCab\mecabrc にあります
# MECAB_DIC_PATH = r"C:\MeCab\mecabrc" # MeCabを使用する場合にコメント解除

def preprocess_text(text):
    """
    テキストのクリーンアップと形態素解析を行い、単語のリストを返す関数
    """
    # MeCabのTaggerの初期化（Janomeを使う場合はTokenizer()に置き換え）
    # MeCabを使用する場合:
    # m = MeCab.Tagger(f"-Owakati -r {MECAB_DIC_PATH}")
    # Janomeを使用する場合:
    from janome.tokenizer import Tokenizer
    t = Tokenizer()


    # 1. 大まかなクリーンアップ (正規表現の追加)
    # 連続するチルダ、アンダーバー、アスタリスクなどの記号列を削除
    clean_text = re.sub(r'~+|_+|\*+', '', text)
    # ファイル冒頭のヘッダーっぽい行を削除 (例: ~~~~~~~, Text File : 0 など)
    clean_text = re.sub(r'^~+\s*Text\s*File\s*:\s*\d+\s*~+\s*', '', clean_text, flags=re.MULTILINE)
    
    # 以前のクリーンアップ処理
    clean_text = re.sub(r'\[.*?\]', '', clean_text) # [ ]で囲まれたテキストを削除
    clean_text = re.sub(r'[「」『』（）”’！？、。]', '', clean_text) # 日本語の記号を削除
    clean_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', clean_text) # 半角記号を削除
    
    # 改行コード \n, \r, \c, バックスラッシュ \ を直接除去
    clean_text = clean_text.replace(r'\n', ' ').replace(r'\r', ' ').replace(r'\c', ' ')
    clean_text = clean_text.replace('\\', '') # 残ったバックスラッシュも除去
    clean_text = clean_text.replace('\u3000', ' ') # 全角スペースを半角スペースに統一
    clean_text = re.sub(r'\s+', ' ', clean_text) # 連続するスペースを1つに
    clean_text = clean_text.strip() # 前後の空白を削除

    sentences_tokens = []
    # テキストを文ごとに分割 (句読点や改行で区切るなど、より高度な方法も検討)
    # ここでは、簡略化のためピリオド、感嘆符、疑問符で区切る。
    # 実際には、日本語の文分割ライブラリ (e.g., Kensho, Juman) の利用も検討
    sentence_delimiters = r'[。！？！？\n]' # 日本語の句読点と改行で分割
    raw_sentences = re.split(sentence_delimiters, clean_text)

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence: # 空の文はスキップ
            continue

        tokens = []
        try:
            # Janomeを使用する場合
            tokens = [token.surface for token in t.tokenize(sentence)]
            
            # MeCabを使用する場合 (コメント解除して使用)
            # tokens = m.parse(sentence).split()

            # 短すぎる単語や不要な空白文字列を除去するフィルタリング
            # (形態素解析後に出る「 」のような不要な空白トークンを排除)
            tokens = [token for token in tokens if token and len(token) > 1 and token.strip() != '']
            # len(token) > 1 は一文字のひらがな・カタカナ・漢字などを除外してしまう可能性があるので注意
            # あまりに頻出する意味のない記号的な単語（例：'-'など）もここで除外検討

            if tokens: # トークンが残っている場合のみ追加
                sentences_tokens.append(tokens)

        except Exception as e:
            # MeCab/Janomeの解析エラーを捕捉
            print(f"形態素解析エラー（文スキップ）: {sentence[:50]}... : {e}")
            continue

    return sentences_tokens

# --- データ収集の実行 ---
# ここに、GitHubからクローンした「pokemon_text_dumps」フォルダのパスを設定してください。
# 例: data_directory = r"C:\Users\masat\OneDrive\デスクトップ\ポケモンテキスト\pokemon_text_dumps"
data_directory = "C:\\Users\\masat\\OneDrive\\デスクトップ\\ポケモンテキスト\\pokemon_text_dumps\\sm\\storytext" # 例としてxyの日本語テキストフォルダを直接指定

all_sentences_tokens = []
if os.path.exists(data_directory):
    for filename in os.listdir(data_directory):
        # 日本語のファイルのみを対象にする
        if filename.endswith("japanese.txt") or filename.endswith("japanese_kanji.txt"):
            filepath = os.path.join(data_directory, filename)
            print(f"Loading and processing: {filename}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                processed_sentences = preprocess_text(raw_text)
                all_sentences_tokens.extend(processed_sentences)
            except UnicodeDecodeError:
                print(f"警告: {filename} のUTF-8デコードに失敗しました。別のエンコーディングを試すか、ファイルを修正してください。")
            except Exception as e:
                print(f"警告: {filename} の読み込みまたは処理中にエラーが発生しました: {e}")
else:
    print(f"エラー: データディレクトリ '{data_directory}' が見つかりません。パスを確認してください。")
    exit()

if not all_sentences_tokens:
    print("エラー: 処理する日本語テキストデータがありません。パスを確認するか、ファイルの内容を確認してください。")
    exit()

print(f"総文数（トークンリストの数）: {len(all_sentences_tokens)}")
print(f"最初の5つの文のトークン化例: {all_sentences_tokens[:5]}")


# --- 2. Word2Vecモデルの学習 ---

print("\n--- Word2Vecモデルの学習を開始します ---")
word2vec_model = Word2Vec(
    sentences=all_sentences_tokens,
    vector_size=200,       # ベクトルの次元数（例として200次元）
    window=5,              # 周囲5単語をコンテキストとして考慮
    min_count=5,           # 5回以上出現する単語のみ学習対象（データ量に応じて調整）
    sg=0,                  # CBOWモデルを使用 (0=CBOW, 1=Skip-gram)
    epochs=20,             # 学習エポック数
    workers=os.cpu_count() - 1 if os.cpu_count() else 1 # CPUコア数-1を利用、最低1
)

# 学習済みモデルの保存
model_save_path = "pokemon_word2vec_model.model"
word2vec_model.save(model_save_path)
print(f"\nWord2Vecモデルを '{model_save_path}' に保存しました。")

# --- 3. 学習済みモデルの確認と利用 ---

print("\n--- 学習済みモデルの確認 ---")
try:
    if 'ポケモン' in word2vec_model.wv:
        print("「ポケモン」のベクトル形状:", word2vec_model.wv['ポケモン'].shape)
        print("「ポケモン」に似た単語:")
        for word, similarity in word2vec_model.wv.most_similar('ポケモン'):
            print(f"  {word}: {similarity:.4f}")
    else:
        print("警告: モデルに「ポケモン」という単語が含まれていません。min_countや学習データ量を確認してください。")

    if '人間' in word2vec_model.wv and '共存' in word2vec_model.wv:
        print("\n「人間」と「共存」の類似度:", word2vec_model.wv.similarity('人間', '共存'))
    else:
        print("警告: モデルに「人間」または「共存」という単語が含まれていません。min_countや学習データ量を確認してください。")

except KeyError as e:
    print(f"エラー: モデルに単語 '{e}' が含まれていません。学習データが少ないか、min_countが高すぎる可能性があります。")

# --- セマンティクスコア算出の準備 ---
# この後、この学習済みword2vec_modelを使ってセマンティクスコアを算出します。

# テーマキーワードの定義（例：「人間とポケモンの共存」）
theme_keywords = ['共存', '協力', '絆', '仲間', '信頼', '助け合い', '理解', '共生', '平和', '多様性', '自然', '環境', '対立', '支配']

def get_theme_reference_vector(keywords, model_wv):
    """
    テーマを表す単語群の平均ベクトルを計算する
    """
    theme_vectors = []
    for k in keywords:
        if k in model_wv: # モデルに単語が存在するか確認
            theme_vectors.append(model_wv[k])
    
    if not theme_vectors:
        print(f"警告: テーマキーワードのいずれもモデルに存在しません。学習データが不足している可能性があります。")
        return np.zeros(model_wv.vector_size) # ゼロベクトルを返すか、エラーを出す
    
    return np.mean(theme_vectors, axis=0)

# 論文で定義した「人間とポケモンの共存」テーマの参照ベクトル
theme_coexistence_vector = get_theme_reference_vector(theme_keywords, word2vec_model.wv)

if np.all(theme_coexistence_vector == 0):
    print("エラー：テーマ参照ベクトルが生成できませんでした。キーワードやモデルを確認してください。")
else:
    print("\nテーマ参照ベクトルが生成されました。これでセマンティクスコア算出に進めます。")