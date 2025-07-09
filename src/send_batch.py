import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PJ")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def start_batch_job():
    for i in range(29):
        input_file_path = f"batch_input_{i}.jsonl"
        # 1. ファイルをアップロード
        batch_input_file = client.files.create(
            file=open(input_file_path, "rb"),
            purpose="batch"
        )
        print(f"File uploaded. File ID: {batch_input_file.id}")

        print("Creating batch job...")
        # 2. バッチ処理を開始
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h", # 24時間以内に処理を完了
            metadata={
                "description": "Image analysis for simulation data"
            }
        )
        print(f"Batch job created successfully. Batch ID: {batch_job.id}")
        print("You can check the status with the next script using this Batch ID.")

if __name__ == "__main__":
    start_batch_job()