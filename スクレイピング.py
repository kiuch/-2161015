# -*- coding: utf-8 -*-
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import time
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

# ----------------------------------------------------------------------
# 1. 設定
# ----------------------------------------------------------------------
# スクレイピング対象のURL
METACRITIC_URL = "https://www.metacritic.com/game/pokemon-scarlet/user-reviews/?platform=nintendo-switch"
OUTPUT_FILENAME = "metacritic_user_reviews.csv"
MAX_REVIEWS = 300  # 収集したいレビューの最大件数
# ページ全体の読み込みタイムアウトを90秒に延長
INITIAL_TIMEOUT_MS = 90000 
# Chromeを非表示（True）で実行。デバッグ時にはFalseに設定
HEADLESS_MODE = True 

# ----------------------------------------------------------------------
# 2. スクレイピング関数
# ----------------------------------------------------------------------

async def scrape_metacritic_reviews():
    data = []
    print("Playwrightを起動中...")

    # Playwrightの非同期コンテキストを開始
    async with async_playwright() as p:
        # Chrome/Chromiumブラウザを起動
        browser = await p.chromium.launch(headless=HEADLESS_MODE)
        page = await browser.new_page()
        
        # --- Metacriticからの収集開始 ---
        print(f"\n--- Metacriticからの収集開始: {METACRITIC_URL} ---")
        
        try:
            # ページ移動時のタイムアウトを設定
            await page.goto(METACRITIC_URL, timeout=INITIAL_TIMEOUT_MS, wait_until='domcontentloaded')
            
            # 最初のレビューコンテナが表示されるまで、カスタム待機
            try:
                await page.wait_for_selector("div.userReviews", timeout=15000)
            except PlaywrightTimeoutError:
                print("警告: 初期レビューコンテナの表示に時間がかかっていますが、処理を継続します。")

        except Exception as e:
            print(f"致命的なエラーが発生しました（ページ移動失敗）: {e}")
            await browser.close()
            return data

        # レビューの全ロード処理 (スクロール戦略)
        reviews_before = 0
        scroll_count = 0
        MAX_SCROLLS = 100 
        MAX_RETRIES = 5 # ロード失敗時のリトライ回数

        while len(data) < MAX_REVIEWS and scroll_count < MAX_SCROLLS:
            
            # ページの一番下までスクロール（新しいレビューを読み込むため）
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000) # スクロール後の読み込み待ち
            
            # --- レビューロード後のチェック ---
            review_elements_all = await page.locator('div.userReviews > div.review-panel').all()
            reviews_current = len(review_elements_all)
            
            retries = 0
            while reviews_current == reviews_before and reviews_current > 0 and retries < MAX_RETRIES:
                # レビュー数が増えていない場合、再度スクロールしてリトライ
                await page.evaluate("window.scrollBy(0, 500)")
                await page.wait_for_timeout(1000)
                review_elements_all = await page.locator('div.userReviews > div.review-panel').all()
                reviews_current = len(review_elements_all)
                retries += 1
            
            if reviews_current == reviews_before and reviews_current > 0:
                print("レビューの総数が増えず、ページの終端に達したと判断します。")
                break
            
            # --- 新しいレビューの抽出 ---
            new_reviews_to_process = review_elements_all[len(data):]
            
            for review_element in new_reviews_to_process:
                try:
                    # スコアの抽出
                    score_text = await review_element.locator('div.score_value').inner_text()
                    
                    # 本文の抽出
                    review_text = await review_element.locator('div.review_body > div').inner_text()
                    
                    data.append({
                        "Title": "ポケモンSV ユーザーコメント (Metacritic)",
                        "Score": float(score_text) if score_text.isdigit() else None,
                        "Review_Text": review_text.strip()
                    })
                    
                    if len(data) >= MAX_REVIEWS:
                        break

                except Exception:
                    # 要素が見つからなかった、または抽出エラーの場合、スキップ
                    continue
            
            reviews_before = reviews_current
            scroll_count += 1

        print(f"Metacriticから {len(data)} 件のレビューを収集しました。")
        
        # ブラウザを閉じる
        await browser.close()

    # DataFrameに変換してCSVに保存
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        print(f"データは '{OUTPUT_FILENAME}' に保存されました。")
        
    return data

# ----------------------------------------------------------------------
# 3. メイン実行
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Playwrightの非同期関数を実行
    asyncio.run(scrape_metacritic_reviews())
