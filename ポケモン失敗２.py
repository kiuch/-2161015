import MeCab 
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# MeCabのTaggerの初期化（Janomeを使う場合はTokenizer()に置き換え）
# Janomeを使用する場合:
from janome.tokenizer import Tokenizer
t = Tokenizer()


def preprocess_text(text):
    """
    テキストのクリーンアップと形態素解析を行い、単語のリストを返す関数
    """
    t_tokenizer = Tokenizer() # Janomeのインスタンスを作成

    # 1. 大まかなクリーンアップ (文分割前に記号を削除しすぎないように注意)
    # [ ]で囲まれたテキストを削除 (例: [主人公] [VAR_1001] など)
    clean_text = re.sub(r'\[.*?\]', '', text) 
    
    # ポケモンゲーム特有の制御文字や不要なバックスラッシュを削除/置換
    clean_text = clean_text.replace(r'\n', ' ').replace(r'\r', ' ').replace(r'\c', ' ') # 改行コードをスペースに
    clean_text = clean_text.replace('\\', '') # 残ったバックスラッシュ（エスケープでないもの）を削除
    clean_text = clean_text.replace('\u3000', ' ') # 全角スペースを半角スペースに統一

    # 数字の除去 (テーマに不要な場合)
    clean_text = re.sub(r'\d+', '', clean_text)

    # 連続するスペースを1つにまとめ、前後の空白を削除
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    sentences_tokens = []
    # 簡易的な文分割 (句読点や改行で区切る)
    # ここで「。」「！」「？」は文分割のために残しておく
    sentence_delimiters = r'[。！？\n]' 
    raw_sentences = re.split(sentence_delimiters, clean_text)

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence: # 空の文はスキップ
            continue
        
        # 文ごとの記号除去はここでやる (文分割には影響しない)
        sentence = re.sub(r'[、。（）『』“”’／\-_~!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip() # 再度スペース整理

        if not sentence: # 記号除去で空になった場合もスキップ
            continue

        filtered_tokens = []
        try:
            for token in t_tokenizer.tokenize(sentence): # Janomeの場合
                surface = token.surface 
                # 品詞によるフィルタリングを検討 (Janomeのtoken.part_of_speechを活用)
                # 例: if not (token.part_of_speech.startswith('助詞') or token.part_of_speech.startswith('助動詞')):
                
                # 簡易的なストップワード/ノイズ単語除去（一旦最小限に留める）
                if surface.strip() == '':
                    continue
                if surface.lower() in ['text', 'file', 'id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'こと', 'もの', 'よう', 'そう']: # よくあるノイズ
                    continue
                
                # １文字のひらがな/カタカナで意味のないものを除外（ただし重要な漢字一文字は残す）
                if len(surface) == 1 and re.match(r'[あ-んア-ン]', surface) and surface not in ['力', '愛', '心', '夢', '光', '闇', '星', '技']: # 例
                    continue

                filtered_tokens.append(surface)
            
            if filtered_tokens: 
                sentences_tokens.append(filtered_tokens)

        except Exception as e:
            print(f"形態素解析エラー（文スキップ）: {sentence[:50]}... : {e}")
            continue
    return sentences_tokens

# --- データ収集の実行（Word2Vec学習の前回コードから再掲） ---
# ★ここをあなたのデータディレクトリに合わせてください★
data_directory = "C:\\Users\\masat\\OneDrive\\デスクトップ\\ポケモンテキスト\\pokemon_text_dumps\\oras\\storytext" 

all_sentences_tokens = []
if os.path.exists(data_directory):
    for filename in os.listdir(data_directory):
        # 日本語のファイルのみを対象にする
        # ファイル名がxy_storytext_japanese.txtやxy_storytext_japanese_kanji.txtであることを想定
        if "japanese.txt" in filename or "japanese_kanji.txt" in filename:
            filepath = os.path.join(data_directory, filename)
            print(f"Loading and processing: {filename}")
            try:
                # ファイルエンコーディングがUTF-8であることを確認
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                processed_sentences = preprocess_text(raw_text)
                all_sentences_tokens.extend(processed_sentences)
            except UnicodeDecodeError:
                print(f"警告: {filename} のUTF-8デコードに失敗しました。エンコーディングを確認してください。")
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
    vector_size=200,       # ベクトルの次元数
    window=5,              # 周囲5単語をコンテキストとして考慮
    min_count=5,           # 5回以上出現する単語のみ学習対象（データ量に応じて調整）
    sg=0,                  # CBOWモデルを使用 (0=CBOW, 1=Skip-gram)
    epochs=20,             # 学習エポック数
    workers=os.cpu_count() - 1 if os.cpu_count() else 1 # CPUコア数-1を利用、最低1
)
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
# テーマキーワードの定義（例：「人間とポケモンの共存」）
theme_keywords = ['一緒', '協力', '絆', '仲間', '信頼', '助け合い', '理解', 'なつく', '平和', '多様性', '自然', '争い', '対立', '支配']

def get_theme_reference_vector(keywords, model_wv):
    """
    テーマを表す単語群の平均ベクトルを計算する
    """
    theme_vectors = []
    for k in keywords:
        if k in model_wv:
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

    flat_tokens = [token for sentence_tokens in scenario_tokens_list for token in sentence_tokens]
    
    if not flat_tokens:
        return 0.0

    scenario_vector = get_text_vector(flat_tokens, model_wv)

    if np.all(scenario_vector == 0): # 全てゼロベクトルになった場合（関連単語なし）
        return 0.0

    similarity = cosine_similarity(scenario_vector.reshape(1, -1), theme_ref_vector.reshape(1, -1))[0][0]
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
print("\n--- 「人間とポケモンの共存」テーマのスコア計算 ---")
overall_coexistence_score = calculate_coexistence_score(
    all_sentences_tokens,
    theme_coexistence_vector,
    word2vec_model.wv
)

print(f"読み込んだ全日本語テキストにおける「人間とポケモンの共存」セマンティクスコア: {overall_coexistence_score:.4f}")