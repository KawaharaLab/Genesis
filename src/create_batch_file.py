import re
import base64
import os
import json
import re
import base64
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PJ")

system_prompt = """
You are a professional in manipulation and analysis of images.
You are tasked with analyzing a set of images extracted from a video.
Focus on information needed to manipulate the objects in the images.
"""
# user_prompt = """
# The following images are extracted from a video. Tell me what is happening in these images.
# You should focus on the end-effector and the object being manipulated.
# If the end-effector never touches the object, do not mention the object.
# If the end-effector touches the object, carefully describe the interaction, such as grasping, pinching, and pushing.
# Do not mention the shape or color of the object; simply refer to it as "the object".
# If the end-effector is touching the object, explain the level of deformation using the following scale:
# 'none', 'slight', 'moderate', 'heavy', or 'extreme'.
# Also, if the end-effector is grasping the object, describe whether the grasp seems stable or unstable.
# Output only the analysis of the images. Do not include any other information. Only use alphabetical characters, spaces, commas, and periods.
# Do not use the word "end-effector" in your response; simply refer to it as "hand".
# Output in approximately 15 words.
# """
user_prompt = """
Analyze the sequence of images and classify the event by choosing the single best description from the list below.
The focus is only on the direct interaction between the end-effector of the arm and the object.
Carefully observe the pictures. The object may be gray, the same color as part of the robot arm, so be careful not to miss it.
Use all images in the sequence to understand the interaction. Remember that the images are sequential.
Rememver to focus on the end-effector(hand and finger) of the robot arm. Even if the object is in contact with the robot arm, if the end-effector is not touching the object, it is not considered a contact.
Strictly follow the format of the output choices and rules below.
### Vocabulary Definition
- grab: The action of closing fingers to secure the object.
- hold: The state of keeping the object secured in the hand.
- push: The hand moves the object from its original position by making contact.
- touch: The hand makes contact with the object without grabbing or pushing it.
- drop: The hand unintentionally releases the object.
- no contact: The hand does not touch the object at all in the images.

### Output Choices
- No contact.;no contact
- Grabs the object.;grab
- Holds the object.;hold
- Touches the object.;touch
- Pushes the object.;push
- Drops the object.;drop

### Rules
1. Your output must be one of the exact options from the 'Output Choices' list.
2. Do not add any other information, explanation, quotation marks, or formatting.
"""
# Base64エンコード関数はそのまま使用
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

def create_batch_file():
    frictions = ["050", "150", "250"]
    object_name = [
        ["001_chips_can", "002_master_chef_can", "003_cracker_box", "004_sugar_box"], 
        ["005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box"], 
        ["009_gelatin_box", "010_potted_meat_can", "011_banana", "012_strawberry"], 
        ["013_apple", "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum"], 
        ["019_pitcher_base", "021_bleach_cleanser", "023_wine_glass", "024_bowl"], 
        ["025_mug", "026_sponge", "028_skillet_lid", "029_plate", "030_fork", "031_spoon"], 
        ["032_knife", "033_spatula", "035_power_drill", "036_wood_block", "037_scissors"], 
        ["038_padlock", "040_large_marker", "041_small_marker", "042_adjustable_wrench"], 
        ["043_phillips_screwdriver", "044_flat_screwdriver", "048_hammer"], 
        ["050_mideum_clamp", "051_large_clamp", "052_extra_large_clamp"], 
        ["053_mini_soccer_ball", "055_baseball"], ["056_tennis_ball", "057_racquetball"], 
        ["058_golf_ball", "059_chain", "061_foam_brick"], ["062_dice", "063-a_marbles"], 
        ["065-a_cups", "065-b_cups", "065-c_cups"], ["065-d_cups", "065-e_cups"], 
        ["065-f_cups", "065-i_cups", "065-j_cups"], ["070-a_colored_wood_blocks"], 
        ["071_nine_hole_peg_test", "072-a_toy_airplane", "072-c_toy_airplane"], 
        ["072-d_toy_airplane", "072-f_toy_airplane", "072-h_toy_airplane"], 
        ["072-i_toy_airplane", "072-j_toy_airplane", "072-k_toy_airplane"], 
        ["073-a_lego_duplo", "073-b_lego_duplo"], ["073-c_lego_duplo", "073-d_lego_duplo"], 
        ["073-e_lego_duplo", "073-f_lego_duplo"], ["073-g_lego_duplo", "073-h_lego_duplo"], 
        ["073-i_lego_duplo", "073-j_lego_duplo"], ["073-k_lego_duplo", "073-l_lego_duplo"], 
        ["073-m_lego_duplo", "076_timer"], ["077_rubiks_cube", "bottle", "cube"]
    ]
    materials = ["rubber"]
    for O in range(len(object_name)):
        output_filename = f"batch_input_{O}.jsonl"
        with open(output_filename, "w", encoding="utf-8") as f_batch:
            for friction in frictions:
                for name in object_name[O]:
                    for material in materials:
                        PHOTO_DIR = f"data/photos/{name}/{material}/{friction}"
                        CSV_PATH = f"data/csv/{name}/{name}_{material}_{friction}.csv"
                        if not os.path.exists(PHOTO_DIR):
                            continue

                        # (画像ファイルのスキャンとキャッシュ部分は同じ)
                        pattern = re.compile(rf"{re.escape(name)}_{re.escape(material)}_{re.escape(friction)}_(\d+)\.png$")
                        frame_nums = sorted(
                            int(m.group(1))
                            for f_name in os.listdir(PHOTO_DIR) if (m := pattern.match(f_name))
                        )
                        if not frame_nums or len(frame_nums) < 7:
                            continue
                        
                        base64_cache = {}
                        for num in frame_nums:
                            image_path = os.path.join(PHOTO_DIR, f"{name}_{material}_{friction}_{num:05d}.png")
                            if os.path.exists(image_path):
                                base64_cache[num] = encode_image_to_base64(image_path)

                        windows = [frame_nums[i : i + 7] for i in range(0, len(frame_nums) - 6)]

                        for i, win in enumerate(windows):
                            base64_images = [base64_cache.get(num) for num in win]
                            if None in base64_images:
                                continue

                            # Batch API用のリクエストオブジェクトを作成
                            request_json = {
                                # custom_id: 後でどのリクエストに対する結果か特定するために使用
                                "custom_id": f"{CSV_PATH}|{win[0]}",
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": {
                                    "model": "o4-mini",
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": user_prompt},
                                                *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in base64_images]
                                            ]
                                        }
                                    ],
                                }
                            }
                            # JSON文字列としてファイルに書き込む
                            f_batch.write(json.dumps(request_json) + "\n")
                        
                        print(f"Generated requests for {name}/{material}/{friction}")

if __name__ == "__main__":
    create_batch_file()