import base64
import os
import csv
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import time, pandas as pd
BASE_PATH = "/Users/no.166/Documents/Azka\'s Workspace/Genesis"

# ── OpenAI client ──────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are a professional in manipulation and analysis of images.
You are tasked with analyzing a set of images extracted from a video.
Focus on information needed to manipulate the objects in the images.
"""
user_prompt = """
The following images are taken at the same time from three different angles. Tell me what is happening in this timestep.
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

# Do we want to call the object -> 'the object'? Feels like we lose a lot of nuance



def build_messages(prompt_text: str,
                   encoded_imgs: list[str]):
    
    content = [{"type": "text", "text": prompt_text}]
    # append images
    for img64 in encoded_imgs:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img64}"}})

    return {"role": "user", "content": content}

if __name__ == "__main__":
    # obj_folder_path = os.path.join(BASE_PATH, "data", "mujoco_scanned_objects", "models")
    # all_objects = os.listdir(folder_path)
    picked_up_path = os.path.join(BASE_PATH, "main", "data", "picked_up")
    picked_up_photo_path = os.path.join(picked_up_path, "photos")
    # picked_up_csv_path = os.path.join(picked_up_path, "csv")
    obj_name = "5_HTP"
    material = "Elastic"
    deformation = "hard"
    obj_path = os.path.join(picked_up_photo_path, obj_name, material, deformation)
    frames = os.listdir(obj_path)
    annotation_file = f'{BASE_PATH}/main/data/picked_up/annotations'
    encoded_images = []
    i = 0
    df = pd.DataFrame(columns=['timestamp', 'annotation'])
    for frame in sorted(frames):
        i += 1
        frame = os.path.join(obj_path, frame)
        with open(frame, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            print(f'{frame} ' + encoded_image[:100] + "...")
            encoded_images.append(encoded_image)
            if i == 3:
                message = build_messages(user_prompt, encoded_images)

        
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": message['role'], "content": message['content']}
                    ],
                    max_tokens=100
                )
                    
                resp = response.choices[0].message.content.strip()
                print(f"Response: {resp}")
               

                df.loc[len(df)] = [int(frame.split('_')[-2]), resp]

               
                    


                i = 0 
                time.sleep(1)
                encoded_images = []
    
        # Save the DataFrame to CSV after processing all frames
        with open(f'{annotation_file}/{obj_name}_{material}_{deformation}.csv', 'w') as f:
            df.to_csv(f, index=False)




        


    


