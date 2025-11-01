import pandas as pd
import MeCab
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# --- 1. 準備と設定 ---

# 日本語表示の設定 (Windows環境で利用可能なフォントを優先)
# もしMeiryoでエラーが出る場合は 'Yu Gothic' や 'MS Gothic' を試してください。
plt.rcParams['font.family'] = 'Meiryo' 
plt.rcParams['font.size'] = 12

# データの読み込みパス
file_path = r'C:\Users\masat\OneDrive\デスクトップ\deep learning\ポケモンsvシナリオ.csv'

# 感情極性辞書（シナリオ評価に特化して強化）
positive_words = {"素晴らしい", "感動的", "最高", "名作", "面白い", "良い", "良かった", "好き", "泣く", "神", "楽しい", "期待", "カワイイ", "ツナガル", "タノシイ", "アツい", "テンカイ"}
negative_words = {"弱い", "平凡", "残念", "陳腐", "最悪", "ストレス", "評価できない", "微妙", "つまらない", "不満", "悪い", "オクレ", "モンダイ", "ムリョウ", "ソガイ", "メンドウ", "サイアク", "コンナン", "ワルイ", "ナンイ"}

# ストップワード (頻出するが意味の薄い単語を大幅に追加・強化)
stop_words = {
    "この", "の", "は", "が", "に", "を", "と", "て", "た", "だ", "し", "もっと", "も", "です", "ます", "けど", "だろ", "それ", 
    "いう", "ある", "なる", "する", "いる", "こと", "ない", "できる", "もの", "ため", "そノ", "られる", "れる", "これ", 
    # カタカナの補助動詞・動詞・助動詞
    "スル", "イル", "イウ", "アル", "ナル", "ナイ", "コト", "デキル", "シレル", "カンズル", 
    # データで頻出した汎用的な単語
    "ゲーム", "シリーズ", "ポケモン", "ワールド", "オープン", "ホンサク", "プレーヤー", "ルート", "ブブン", 
    "レベル", "タメ", "ソノ", "セイリツ", "トオク", "ミエル", "ハツ", "イク", "クル", "オク", "ブタイ", "カンケイ",
    "ホカク", "シュルイ", "タチバ", "マチ", "イチ", "アタリ", "バアイ"
}

# データの読み込み関数 (前回の回答でエンコーディングの問題は解決済みと仮定)
def force_read_csv(file_path):
    encodings_to_try = ['utf-8', 'shift_jis', 'cp932', 'euc-jp']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except Exception:
            continue
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        return df
    except Exception:
        return None

df = force_read_csv(file_path)

# レビュー本文が含まれる列名を明示的に指定
TEXT_COLUMN = 'レビュー' 

if TEXT_COLUMN in df.columns:
    game_reviews = df[TEXT_COLUMN].astype(str).tolist()
    game_reviews = [r.replace('nan', '').strip() for r in game_reviews if r != 'nan' and r != '']
    print(f"✅ レビュー列: '{TEXT_COLUMN}' を分析対象とします。")
else:
    print(f"🚨 エラー: データフレームに '{TEXT_COLUMN}' という列が見つかりません。")
    raise ValueError(f"列 '{TEXT_COLUMN}' が見つかりません。")

# --- MeCab Taggerの初期化 ---
mecab = MeCab.Tagger() 

# --- 2. 前処理 (形態素解析とフィルタリング) ---

def preprocess_text(text):
    """テキストを形態素解析し、名詞・動詞・形容詞・感動詞の原形を抽出"""
    words = []
    if not isinstance(text, str) or len(text) < 2:
        return []
        
    try:
        node = mecab.parseToNode(text)
    except Exception:
        return []

    target_hinshi = ('名詞', '動詞', '形容詞', '感動詞')
    
    while node:
        features = node.feature.split(',')
        hinshi = features[0]
        original_form = node.surface # デフォルトは単語そのもの

        # 原形 (7番目のフィールド) を取得し、なければ単語そのもの (surface)
        if len(features) >= 7 and features[6] != '*':
            original_form = features[6]
        
        # 抽出条件: 対象品詞、ストップワードでない、1文字より長い
        # ここで原形がストップワードに含まれていれば除外する
        if hinshi in target_hinshi and original_form not in stop_words and len(original_form) > 1:
            words.append(original_form)
        
        node = node.next
    return words

processed_reviews = [preprocess_text(review) for review in game_reviews]
tokenized_reviews_str = [" ".join(words) for words in processed_reviews] # 共起行列/TF-IDF用

print("✅ 前処理結果 (形態素解析とフィルタリング) の最初の5件:")
for i in range(min(5, len(processed_reviews))):
    print(f"  レビュー {i+1}: {processed_reviews[i]}")
print("-" * 50)


# --- 3. 単語頻出度分析 (Word Frequency) ---

all_words = [word for sublist in processed_reviews for word in sublist]
word_counts = Counter(all_words)
most_common = word_counts.most_common(20) # 頻出上位20単語
print("✅ 単語頻出度分析 (上位20単語):")
for word, count in most_common:
    print(f"  {word}: {count}回")

# 棒グラフで可視化
if most_common:
    words, counts = zip(*most_common)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.title('単語頻出度 (Word Frequency)')
    plt.xlabel('単語')
    plt.ylabel('出現回数')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('word_frequency_bar_chart.png')
    plt.close()
    print("棒グラフを 'word_frequency_bar_chart.png' として保存しました。")
else:
    print("分析対象の単語がありませんでした。")

print("-" * 50)


# --- 4. 共起行列 (Co-occurrence Matrix) ---

co_occurrence_counts = defaultdict(int)
for words in processed_reviews:
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            pair = tuple(sorted((words[i], words[j])))
            co_occurrence_counts[pair] += 1

top_n = 10
sorted_co_occurrence = sorted(co_occurrence_counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
print(f"✅ 共起分析 (共起頻度の高い上位{top_n}ペア):")
for (word1, word2), count in sorted_co_occurrence:
    print(f"  {word1} - {word2}: {count}回")

# TF-IDF行列の作成
vectorizer = TfidfVectorizer(use_idf=True)
tfidf_matrix = vectorizer.fit_transform(tokenized_reviews_str)
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print("\n✅ TF-IDF行列の最初の5行と5列 (データの一部):")
print(tfidf_df.iloc[:5, :5])
print("-" * 50)


# --- 5. 感情分析 (Sentiment Analysis) ---

def analyze_sentiment(words):
    """辞書ベースの感情分析"""
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    
    if positive_score > negative_score:
        sentiment = 'Positive'
    elif negative_score > positive_score:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return sentiment, positive_score, negative_score

results = []
for i, words in enumerate(processed_reviews):
    sentiment, pos_score, neg_score = analyze_sentiment(words)
    results.append({
        'Review_ID': i + 1,
        'Original_Review': game_reviews[i],
        'Sentiment': sentiment,
        'Positive_Score': pos_score,
        'Negative_Score': neg_score
    })

sentiment_df = pd.DataFrame(results)
print("✅ 感情分析結果 (最初の5件):")
print(sentiment_df.head())

# 感情極性の分布を可視化
sentiment_counts = sentiment_df['Sentiment'].value_counts()
if not sentiment_counts.empty:
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999','#99ff99'])
    plt.title('レビュー感情極性の分布')
    plt.tight_layout()
    plt.savefig('sentiment_distribution_pie_chart.png')
    plt.close()
    print("円グラフを 'sentiment_distribution_pie_chart.png' として保存しました。")
else:
    print("感情分析の対象データがありませんでした。")

print("-" * 50)

# 結果の統合とCSV書き出し
output_df = df.copy()
output_df = output_df.merge(sentiment_df, left_index=True, right_index=True, how='left')
output_df['Processed_Words'] = pd.Series([processed_reviews[i] if i < len(processed_reviews) else [] for i in range(len(output_df))])

output_filename = 'scenario_evaluation_results.csv'
output_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"✅ 分析結果を '{output_filename}' に書き出しました。")