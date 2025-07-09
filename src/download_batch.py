import os
import time
import json
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PJ")
client = OpenAI(api_key=OPENAI_API_KEY)

# 処理済みのバッチIDを記録するファイル
PROCESSED_LOG_PATH = "processed_batches.log"
# 最終的な結果を保存するCSVファイル
OUTPUT_CSV_PATH = "data/analysis_results_from_batch.csv"

def load_processed_ids() -> set[str]:
    """処理済みのバッチIDをログファイルから読み込む"""
    if not os.path.exists(PROCESSED_LOG_PATH):
        return set()
    with open(PROCESSED_LOG_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f}

def log_processed_id(batch_id: str):
    """処理が完了したバッチIDをログファイルに追記する"""
    with open(PROCESSED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{batch_id}\n")

def process_completed_batches():
    """完了済みの未処理バッチを全て処理する"""
    print("Loading already processed batch IDs...")
    processed_ids = load_processed_ids()

    # ヘッダーの書き込み（ファイルがまだ存在しない場合）
    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, "w", encoding="utf-8") as f:
            f.write("csv_path,timestep_start,annotation,label\n")

    print("Fetching recent batch jobs from OpenAI...")
    try:
        # 最近のバッチジョブを30件取得
        recent_batches = client.batches.list(limit=30)
    except Exception as e:
        print(f"Failed to fetch batch list: {e}")
        return

    # 完了しているバッチを抽出
    completed_batches = [b for b in recent_batches if b.status == 'completed']

    if not completed_batches:
        print("No completed batches found.")
        return

    # メインのCSVファイルに結果を追記
    with open(OUTPUT_CSV_PATH, "a", encoding="utf-8") as f_out:
        for batch in completed_batches:
            if batch.id in processed_ids:
                print(f"Skipping already processed Batch ID: {batch.id}")
                continue

            print(f"Processing new completed Batch ID: {batch.id}")
            
            # 結果ファイルIDを取得
            output_file_id = batch.output_file_id
            if not output_file_id:
                print(f"No output file found for batch {batch.id}. Skipping.")
                log_processed_id(batch.id) # エラーでも次回からスキップする
                continue

            try:
                # 結果ファイルの内容を取得
                result_content_response = client.files.content(output_file_id)
                result_content = result_content_response.read().decode('utf-8')

                # 結果を1行ずつ解析してCSVに追記
                for line in result_content.strip().split('\n'):
                    result_json = json.loads(line)
                    custom_id = result_json.get('custom_id', '')
                    response_body = result_json.get('response', {}).get('body', {})
                    
                    if not custom_id or not response_body:
                        continue
                        
                    csv_path, timestep_start = custom_id.split('|')
                    content = response_body.get('choices', [{}])[0].get('message', {}).get('content')
                    
                    if content:
                        if content == "no contact":
                            content = "No contact.;no contact"
                        annotation, label = content.rsplit(";", 1)
                        f_out.write(f"{csv_path},{timestep_start},{annotation},{label}\n")
                
                print(f"Successfully processed and saved results for Batch ID: {batch.id}")
                # 正常に処理が完了したらログに記録
                log_processed_id(batch.id)

            except Exception as e:
                print(f"An error occurred while processing batch {batch.id}: {e}")

if __name__ == "__main__":
    process_completed_batches()