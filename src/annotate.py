import re
import base64
import openai
import os

import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PJ")

system_prompt = """
You are a professional in manipulation and analysis of images.
You are tasked with analyzing a set of images extracted from a video.
Focus on information needed to manipulate the objects in the images.
"""
user_prompt = """
The following images are extracted from a video. Tell me what is happening in these images.
You should focus on the end-effector and the object being manipulated.
If the end-effector never touches the object, do not mention the object.
If the end-effector touches the object, carefully describe the interaction, such as grasping, pinching, and pushing.
Do not mention the shape or color of the object; simply refer to it as "the object".
If the end-effector is touching the object, explain the level of deformation using the following scale:
'none', 'slight', 'moderate', 'heavy', or 'extreme'.
Also, if the end-effector is grasping the object, describe whether the grasp seems stable or unstable.
Output only the analysis of the images. Do not include any other information. Only use alphabetical characters, spaces, commas, and periods.
Do not use the word "end-effector" in your response; simply refer to it as "hand".
Output in approximately 15 words.
"""

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string

def analyze_image_with_openai(base64_images: list[str | None]):
    """
    Base64エンコードされた画像をOpenAI APIに送信して分析する。
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY":
        print("エラー: OpenAI APIキーが設定されていません。")
        return None

    for base64_image in base64_images:
        if base64_image is None:
            raise ValueError("Base64エンコードされた画像が無効です。")

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[0]}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[1]}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[2]}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[3]}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[4]}"},
                        },
                        {"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[5]}"},
                        },
                        {"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_images[6]}"},
                        },
                    ]
                }
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"エラー: OpenAI APIの呼び出し中に問題が発生しました: {e}")
        return None


if __name__ == "__main__":
    # 画像のパスを取得
    frictions = ["050", "150", "250"]
    object_name = ["001_chips_can", "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana", "012_strawberry", "013_apple", "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum", "019_pitcher_base", "021_bleach_cleanser", "023_wine_glass", "024_bowl", "025_mug", "026_sponge", "028_skillet_lid", "029_plate", "030_fork", "031_spoon", "032_knife", "033_spatula", "035_power_drill", "036_wood_block", "037_scissors", "038_padlock", "040_large_marker", "041_small_marker", "042_adjustable_wrench", "043_phillips_screwdriver", "044_flat_screwdriver", "048_hammer", "050_mideum_clamp", "051_large_clamp", "052_extra_large_clamp", "053_mini_soccer_ball", "055_baseball", "056_tennis_ball", "057_racquetball", "058_golf_ball", "059_chain", "061_foam_brick", "062_dice", "063-a_marbles", "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups", "065-f_cups", "065-i_cups", "065-j_cups", "070-a_colored_wood_blocks", "071_nine_hole_peg_test", "072-a_toy_airplane", "072-c_toy_airplane", "072-d_toy_airplane", "072-f_toy_airplane", "072-h_toy_airplane", "072-i_toy_airplane", "072-j_toy_airplane", "072-k_toy_airplane", "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo", "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo", "073-g_lego_duplo", "073-h_lego_duplo", "073-i_lego_duplo", "073-j_lego_duplo", "073-k_lego_duplo", "073-l_lego_duplo", "073-m_lego_duplo", "076_timer", "077_rubiks_cube"]
    # object_name = ["050_mideum_clamp", "051_large_clamp", "052_extra_large_clamp", "053_mini_soccer_ball"]
    materials = ["rubber"]
    df = pd.DataFrame(
        columns=["name", "material", "friction", "csv_path","timestep_start", "analysis_result"]
    )
    for friction in frictions:
        for name in object_name:
            for material in materials:
                PHOTO_DIR = f"data/photos/{name}/{material}/{friction}"
                CSV_PATH = f"data/csv/{name}/{name}_{material}_{friction}.csv"
                if not os.path.exists(PHOTO_DIR):
                    print(f"Error: Directory {PHOTO_DIR} does not exist.")
                    continue
                
                # 1. ディレクトリ内のPNGファイルからフレーム番号を抽出
                pattern = re.compile(rf"{re.escape(name)}_{re.escape(material)}_{re.escape(friction)}_(\d+)\.png$")
                frame_nums = sorted(
                    int(m.group(1))
                    for f in os.listdir(PHOTO_DIR)
                    if (m := pattern.match(f))
                )

                # 2. 7枚ずつ、前チャンクと1枚だけずれるステップ1でチャンク
                step = 1
                windows: list[list[int]] = []
                for i in range(0, len(frame_nums) - 6, step):
                    windows.append(frame_nums[i : i + 7])
                # 3. フルパスに変換
                image_set = [
                    [os.path.join(PHOTO_DIR, f"{name}_{material}_{friction}_{num:05d}.png")
                     for num in win]
                    for win in windows
                ]
                for i, image_paths in enumerate(image_set):
                    # Base64エンコードされた画像をリストに格納
                    base64_images = []
                    for image_path in image_paths:
                        if os.path.exists(image_path):
                            base64_image = encode_image_to_base64(image_path)
                            base64_images.append(base64_image)
                        else:
                            base64_images.append(None)

                    # OpenAI APIを使用して画像を分析
                    analysis_result = analyze_image_with_openai(base64_images)
                    
                    if analysis_result:
                        print("Analysis Result:", analysis_result)
                        # pandas 2.x では DataFrame.append が廃止されたため loc を使う
                        df.loc[len(df)] = {
                            "name": name,
                            "material": material,
                            "friction": friction,
                            "csv_path": CSV_PATH,
                            "timestep_start": windows[i][0],
                            "analysis_result": analysis_result
                        }
                        # 4. DataFrameをCSVファイルに保存
                        output_csv_path = "data/analysis_results.csv"
                        df.to_csv(output_csv_path, index=False)
                        print(f"Results saved to {output_csv_path}")
                    else:
                        print("Analysis failed.")
