import os

import openai
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PJ")

system_prompt = """
You are a specialized language model that simplifies robotic manipulation log text according to a set of strict rules.
Your sole task is to transform the text based on the provided rules and output only the result.
Context: The experiment involves a robot arm attempting to grab, lift, and manipulate an object.
"""
user_prompt = """
Simplify the following text according to the specified rules.

### Rules
1.  Always remove "Hand" or "The hand" from the beginning of the sentence.
2.  Replace all occurrences of the word "grasp" with "grab".
3.  Completely remove any mention of stability (e.g., "stable grab", "unstable grab"). Do not use the words "stable" or "unstable".
4.  If the robot did not touch the object , do not mention the object or contact at all; do not say "no grab", "no contact". Simply explain the movement.
5.  Do not refer to anything that has not happened. Word such as "lifting unseen", "no deformation", "no contact", "no interaction" should not be used.
5.  If the robots has contact with the object, describe the interaction in a simple manner.
5.  The final output should be approximately 10 words.
6.  Output only the transformed text. Do not include any greetings, explanations, or other extraneous text.

### Examples
- Example 1 (No Contact)
  - input: "Hand moves sideways and rotates above the workspace without making contact, no deformation observed."
  - output: "Moves sideways and rotates above the workspace."
- Example 2 (Successful Grab)
  - input: "The hand moves towards the object and performs a stable grasp, lifting it successfully."
  - output: "Moves towards the object and grab, lifting it."
- Example 3 (Failed Grab)
  - input: "Hand attempts to grasp the object, but it results in an unstable grasp and the object is dropped."
  - output: "Attempts to grab the object, but it is dropped."
- Example 4 (Simple Touch)
  - input: "The hand touches the top of the object and pushes it slightly."
  - output: "Touches the top of the object and pushes it slightly."

### Input:
"""

def analyze_image_with_openai(text_to_simplify: str) -> str | None:

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY":
        print("エラー: OpenAI APIキーが設定されていません。")
        return None

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt + text_to_simplify},
                    ]
                }
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"エラー: OpenAI APIの呼び出し中に問題が発生しました: {e}")
        return None


def main():
    df = pd.read_csv("data/train_simplified.csv")
    for i, row in df.iterrows():
        # 既に簡略化結果があればスキップ
        if pd.notna(row.get("simplified_analysis_result")) and row["simplified_analysis_result"] != "":
            continue
        text_to_simplify = row["analysis_result"]
        simplified_text = analyze_image_with_openai(text_to_simplify)
        # iterrows() の row はコピーなので DataFrame に直接書き込む
        df.at[i, "simplified_analysis_result"] = simplified_text
        print(simplified_text)
        df.to_csv("data/train_simplified.csv", index=False)

if __name__ == "__main__":
    main()