import base64
import os
import csv
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import time
import pandas as pd

# --- Configuration ---
BASE_PATH = "/Users/no.166/Documents/Azka's Workspace/Genesis"

# ── OpenAI client ──────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are a professional data labeler working on a project that wants to label simulations of a franka robot for the purpose of making a model to predict grip force based on a given prompt and object.
You are tasked with analyzing a set of images extracted from a video, and the corresponding object deformation csv and franka force csv.
Focus on information needed to manipulate the objects in the images.
"""
user_prompt = """
The following images are taken at the same time from three different angles. Annotate the timestamp as if you were labeling for a VLA model and wanted to include Tactile information as well. 

Use the different angles to provide a comprehensive analysis of the interaction between the end-effector and the object.
You should focus on the end-effector and the object being manipulated.
If the end-effector never touches the object, do not mention the object.
If the end-effector touches the object, carefully describe the interaction, such as grasping, pinching, and pushing.
Do not mention the shape or color of the object; simply refer to it as "the object".
If the end-effector is touching the object, explain the level of deformation using the following scale:
'none', 'slight', 'moderate', 'heavy', or 'extreme'.
Also, if the end-effector is grasping the object, describe whether the grasp seems stable or unstable.
Output only the analysis of the images. Do not include any other information. Only use alphabetical characters, spaces, commas, and periods.
Output in approximately 15 words.
"""

# Pricing information for different models
prices_per_1m = {
    # Note: These model names might not be exact. Check OpenAI's documentation for the latest models.
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-4o": {"prompt": 5.00, "completion": 15.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4.1-nano": {"prompt": 0.20, "completion": 0.80},
}
model = "gpt-4.1-nano" # Using a more standard model name

def build_messages(prompt_text: str,
                   encoded_imgs: list[str], csv_data, timestep):
    """Builds the user message content for the API call."""
    content = [{"type": "text", "text": prompt_text}]
    # Append images
    for img64 in encoded_imgs:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img64}"}})
        
    for csv in csv_data:
        section = pd.read_csv(csv, skiprows=range(1,timestep), nrows=100)

        content.append({"type": "csv",
                        "csv_url": {"url": f"data:text/csv;base64,{base64.b64encode(csv.encode()).decode()}"}})


    return {"role": "user", "content": content}

if __name__ == "__main__":
    messages = [{"role": "system", "content": system_prompt}]

    # --- Setup Paths ---
    picked_up_path = os.path.join(BASE_PATH, "main", "data", "picked_up_2")
    picked_up_photo_path = os.path.join(picked_up_path, "photos")
    picked_up_csv_path = os.path.join(picked_up_path, "csv")
    
    obj_name = "5_HTP"
    material = "Elastic"
    deformation = "hard"
    
    obj_photo_path = os.path.join(picked_up_photo_path, obj_name, material, deformation)
    obj_csv_path = os.path.join(picked_up_csv_path, obj_name, material, deformation)
    annotation_dir = os.path.join(BASE_PATH, 'main', 'data', 'picked_up_2', 'annotations')
    
    # Create annotation directory if it doesn't exist
    os.makedirs(annotation_dir, exist_ok=True)

    if not os.path.isdir(obj_photo_path):
        print(f"Error: Directory not found at {obj_photo_path}")
        exit()
    
    # --- Main Processing Loop ---
    try:
        frames = sorted(os.listdir(obj_photo_path))
    except FileNotFoundError:
        print(f"Error: Directory not found at {obj_photo_path}")
        exit()

    # Initialize DataFrame to store results
    df = pd.DataFrame(columns=['timestamp', 'annotation'])
    
    encoded_images = []
    image_counter = 0
    run_number = 0

    for frame_filename in frames:
        # Skip non-image files like .DS_Store
        if not frame_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_counter += 1
        frame_path = os.path.join(obj_photo_path, frame_filename)
        timestep = frame_filename.split('_')[-2]  # Extract timestamp from filename
        
        try:
            with open(frame_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                print(f'Read and encoded: {frame_path}')
                encoded_images.append(encoded_image)
        except Exception as e:
            print(f"Error reading or encoding file {frame_path}: {e}")
            continue

        # When we have collected 3 images, send them for analysis
        if image_counter == 3:
            message = build_messages(user_prompt, encoded_images, timestep)

            # if run_number == 0:
                # messages.append("type": "csv","image_url": {"url": f"data:image/png;base64,{img64}"})  # Add the initial user promp

            try:
                messages.append(message)  # Append the new user message (with the 3 images)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=100
                )

                assistant_message = response.choices[0].message
                messages.append({"role": "assistant", "content": assistant_message.content})  # Add assistant response

                
                resp = assistant_message.content.strip()

                print(f"Response: {resp}")

                # --- FIX 1: Access usage data using dot notation ---
                usage = response.usage
                if usage: # Check if usage data is available
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    cost = (
                        (prompt_tokens / 1000000) * prices_per_1m[model]["prompt"] +
                        (completion_tokens / 1000000) * prices_per_1m[model]["completion"]
                    )

                    print(f"Prompt tokens: {prompt_tokens}")
                    print(f"Completion tokens: {completion_tokens}")
                    print(f"Total tokens: {total_tokens}")
                    print(f"Cost: ${cost:.6f}")
                else:
                    print("Usage data not available in the response.")

                # Extract timestamp from the last processed frame's name
                timestamp = int(frame_filename.split('_')[-2])
                new_row = pd.DataFrame([{'timestamp': timestamp, 'annotation': resp}])
                df = pd.concat([df, new_row], ignore_index=True)

            except Exception as e:
                print(f"An error occurred during the API call: {e}")

            # Reset for the next batch of images
            image_counter = 0 
            encoded_images = []
            print("-" * 20)
            time.sleep(1.5) # Rate limiting

            
    
    # --- FIX 2: Save the DataFrame to CSV *after* the loop has finished ---
    if not df.empty:
        output_csv_path = os.path.join(annotation_dir, f'{obj_name}_{material}_{deformation}.csv')
        try:
            df.to_csv(output_csv_path, index=False)
            print(f"\nSuccessfully saved annotations to {output_csv_path}")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")
    else:
        print("\nNo annotations were generated, CSV file not saved.")
