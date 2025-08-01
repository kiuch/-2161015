import MeCab # 形態素解析器
from gensim.models import Word2Vec # Word2Vecモデル
import numpy as np # 数値計算用
from sklearn.metrics.pairwise import cosine_similarity # コサイン類似度計算用
import os # ファイル操作用
import re # 正規表現用

# --- 1. データ準備（前回のコードから再掲、パスは適切に設定済みとする） ---

# MeCabのTaggerの初期化（Janomeを使う場合はTokenizer()に置き換え）
# Janomeを使用する場合:
from janome.tokenizer import Tokenizer
t = Tokenizer()

def preprocess_text(text):
    """
    テキストのクリーンアップと形態素解析を行い、単語のリストを返す関数
    """
    # Janomeを使用する場合
    t = Tokenizer() # Janomeのインスタンスを作成

    # 1. 大まかなクリーンアップ
    clean_text = re.sub(r'~+|_+|\*+', '', text)
    clean_text = re.sub(r'^~+\s*Text\s*File\s*:\s*\d+\s*~+\s*', '', clean_text, flags=re.MULTILINE)
    
    # 以前のクリーンアップ処理
    clean_text = re.sub(r'\[.*?\]', '', clean_text)
    clean_text = re.sub(r'[「」『』（）”’！？、。]', '', clean_text)
    clean_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', clean_text)
    
    clean_text = clean_text.replace(r'\n', ' ').replace(r'\r', ' ').replace(r'\c', ' ')
    clean_text = clean_text.replace('\\', '')
    clean_text = clean_text.replace('\u3000', ' ')
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.strip()

    sentences_tokens = []
    sentence_delimiters = r'[。！？！？\n]'
    raw_sentences = re.split(sentence_delimiters, clean_text)

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        tokens = []
        try:
            tokens = [token.surface for token in t.tokenize(sentence)]
            tokens = [token for token in tokens if token and len(token) > 1 and token.strip() != '']
            if tokens:
                sentences_tokens.append(tokens)
        except Exception as e:
            print(f"形態素解析エラー（文スキップ）: {sentence[:50]}... : {e}")
            continue
    return sentences_tokens

# --- データ収集の実行（Word2Vec学習の前回コードから再掲） ---
# ここに、GitHubからクローンした「pokemon_text_dumps」フォルダのパスを設定してください。
data_directory = "C:\\Users\\masat\\OneDrive\\デスクトップ\\ポケモンテキスト\\pokemon_text_dumps\\xy\\storytext" # ★ここを実際のパスに設定してください★

all_sentences_tokens = []
if os.path.exists(data_directory):
    for filename in os.listdir(data_directory):
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

# --- 2. Word2Vecモデルの学習（前回のコードから再掲） ---
# 学習済みモデルをロードする場合は、この部分をコメントアウト
# model_save_path = "pokemon_word2vec_model.model"
# if os.path.exists(model_save_path):
#     print(f"\n学習済みWord2Vecモデル '{model_save_path}' をロードします。")
#     word2vec_model = Word2Vec.load(model_save_path)
# else:
print("\n--- Word2Vecモデルの学習を開始します ---")
word2vec_model = Word2Vec(
    sentences=all_sentences_tokens,
    vector_size=200,       # ベクトルの次元数
    window=5,              # 周囲5単語をコンテキストとして考慮
    min_count=5,           # 5回以上出現する単語のみ学習対象
    sg=0,                  # CBOWモデル
    epochs=20,             # 学習エポック数
    workers=os.cpu_count() - 1 if os.cpu_count() else 1
)
model_save_path = "pokemon_word2vec_model.model"
word2vec_model.save(model_save_path)
print(f"\nWord2Vecモデルを '{model_save_path}' に保存しました。")


# --- 3. セマンティクスコア算出 ---

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
        else:
            print(f"警告: テーマキーワード '{k}' が学習済みモデルに存在しません。") # 警告を追加
    
    if not theme_vectors:
        raise ValueError("テーマキーワードのいずれも学習済みモデルに存在しません。テーマの定義または学習データを確認してください。")
    
    return np.mean(theme_vectors, axis=0)

def get_text_vector(text_tokens, model_wv):
    """
    単語リストからテキストブロック全体の平均ベクトルを計算する
    """
    vectors = [model_wv[token] for token in text_tokens if token in model_wv]
    if not vectors:
        # テキストブロックにモデルにない単語しかない場合、ゼロベクトルを返す
        return np.zeros(model_wv.vector_size) 
    return np.mean(vectors, axis=0)

def calculate_coexistence_score(scenario_tokens_list, theme_ref_vector, model_wv):
    """
    各シナリオのセマンティクスコア（テーマとの類似度）を計算する
    scenario_tokens_list: シナリオ全体を構成する単語リストのリスト（文ごとなど）
    theme_ref_vector: テーマの参照ベクトル
    model_wv: Word2Vecの単語ベクトルモデル (model.wv)
    """
    if not scenario_tokens_list:
        return 0.0

    scores = []
    # シナリオ全体を一つのブロックとして扱う場合
    # 全てのトークンを平坦化
    flat_tokens = [token for sentence_tokens in scenario_tokens_list for token in sentence_tokens]
    
    if not flat_tokens:
        return 0.0

    scenario_vector = get_text_vector(flat_tokens, model_wv)

    if np.all(scenario_vector == 0): # 全てゼロベクトルになった場合（関連単語なし）
        return 0.0

    # コサイン類似度を計算
    # reshape(1, -1)は、1つのサンプル（行）と任意の列数を持つ2D配列に変換するため
    similarity = cosine_similarity(scenario_vector.reshape(1, -1), theme_ref_vector.reshape(1, -1))[0][0]
    
    # スコアを-1から1の範囲から0から1の範囲に正規化
    normalized_score = (similarity + 1) / 2
    
    return normalized_score

# --- メイン処理 ---

# 1. テーマ参照ベクトルの生成
try:
    theme_coexistence_vector = get_theme_reference_vector(theme_keywords, word2vec_model.wv)
    print("\nテーマ参照ベクトルが生成されました。")
except ValueError as e:
    print(f"\nエラー: {e}")
    print("テーマキーワードが学習済みモデルに十分含まれていません。min_countの調整や学習データ量を確認してください。")
    exit()

# 2. サンプルシナリオ（全日本語テキスト）のセマンティクスコアを計算
# この例では、読み込んだ全日本語テキストを一つの大きなシナリオとしてスコアを計算
print("\n--- 「人間とポケモンの共存」テーマのスコア計算 ---")
overall_coexistence_score = calculate_coexistence_score(
    all_sentences_tokens,
    theme_coexistence_vector,
    word2vec_model.wv
)

print(f"読み込んだ全日本語テキストにおける「人間とポケモンの共存」セマンティクスコア: {overall_coexistence_score:.4f}")

# --- 今後、作品ごとにスコアを計算する部分 ---
# ここではxy_storytext_japanese.txtとxy_storytext_japanese_kanji.txtのみを読み込んでいるため、
# これは「ポケモンX・Y」の日本語テキスト全体に対するスコアの概算とみなせます。
# 論文では、各作品のテキストを個別に読み込み、個別のスコアを計算する必要があります。

# 例：作品ごとのスコアを計算するフレームワーク
# pokemon_games_data = {} # { 'Red/Green': {'text_tokens': [...], 'sales': X}, 'Gold/Silver': {...} }

# for game_name, game_data in pokemon_games_data.items():
#     game_score = calculate_coexistence_score(
#         game_data['text_tokens'],
#         theme_coexistence_vector,
#         word2vec_model.wv
#     )
#     print(f"'{game_name}' の「人間とポケモンの共存」スコア: {game_score:.4f}")
#     game_data['coexistence_score'] = game_score