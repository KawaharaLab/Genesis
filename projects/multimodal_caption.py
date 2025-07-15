import os
import base64
import csv
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_csv_rows(csv_path, row_indices, has_header=True):
    with open(csv_path, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        header = reader[0] if has_header else None
        rows = reader[1:] if has_header else reader
        extracted = [rows[i] for i in row_indices]
        return header, extracted

def format_csv_data(header, rows, label):
    formatted = f"### {label} Data:\n"
    formatted += ", ".join(header) + "\n" if header else ""
    for i, row in enumerate(rows):
        formatted += f"Step {i+1}: " + ", ".join(row) + "\n"
    return formatted

def run_multistep_prompt(image_paths_by_step, force_csv, deformation_csv, step_indices):
    # 1. Extract CSV data
    force_header, force_rows = extract_csv_rows(force_csv, step_indices)
    deform_header, deform_rows = extract_csv_rows(deformation_csv, step_indices)

    force_text = format_csv_data(force_header, force_rows, "Force")
    deform_text = format_csv_data(deform_header, deform_rows, "Deformation")

    # 2. Load images into base64
    image_messages = []
    for step_index, image_paths in enumerate(image_paths_by_step):
        for img_path in image_paths:
            img_base64 = encode_image_to_base64(img_path)
            image_messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })

    # 3. Prompt content
    system_prompt = """You are analyzing robotic manipulation using multimodal data.
Each time step includes:
- 3 images from different angles
- Corresponding force-torque sensor data
- Corresponding object deformation data

Your task:
For each time step, describe in structured language:
1. Robot joint posture
2. End effector pose
3. Grasped object geometry
4. Contact points
5. Object material appearance
6. Observed motion
7. Environment references

Do NOT include actual numbers or measurements.
Be concise and detailed.
"""

    user_prompt = force_text + "\n" + deform_text

    # 4. API call
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}] + image_messages}
            ],
            max_tokens=1000
        )
        print("Response:\n", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # ✏️ Update file paths and timestep indices here
    step_indices = [250, 750, 1000]

    image_paths_by_step = [
        [
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count0Camera0.png",
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count0Camera1.png",
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count0Camera2.png"
        ],
        [
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count1Camera0.png",
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count1Camera1.png",
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count1Camera2.png"
        ],
        [
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count2Camera0.png",
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count2Camera1.png",
            "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/images/999Crayola_Crayons_24_count2Camera2.png"
        ]
    ]

    force_csv_path = "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/forces_full.csv"
    deform_csv_path = "/Users/no.166/Documents/Azka's Workspace/Genesis/projects/deformation_full.csv"

    run_multistep_prompt(image_paths_by_step, force_csv_path, deform_csv_path, step_indices)
