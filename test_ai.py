import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_image_prompt(image_path, prompt_text="What is in this image? (within 10 output tokens)"):
    base64_image = encode_image_to_base64(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-vision-preview"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        print("Response:\n", response.choices[0].message.content)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    run_image_prompt("test_image.png")
