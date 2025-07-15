import os
import csv
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ── OpenAI client ──────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_csv_rows(csv_path: Path) -> list[list[str]]:
    with open(csv_path, newline="") as f:
        return list(csv.reader(f))

def build_messages(prompt_text: str,
                   encoded_imgs: list[str],
                   force_row: list[str],
                   deform_row: list[str]) -> list[dict]:
    """
    Creates the message block for a single time‑step:
      • prompt text (same for all)
      • 3 images     (model sees them in order)
      • force CSV    (one row, rendered as plain text)
      • deform CSV   (one row, rendered as plain text)
    """
    content = [{"type": "text", "text": prompt_text}]
    # append images
    for img64 in encoded_imgs:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img64}"}})
    # append CSV snippets (short, human‑readable)
    force_snippet  = ",".join(force_row)
    deform_snippet = ",".join(deform_row)
    csv_block = (
        f"\nForce‑torque row: [{force_snippet}]\n"
        f"Deformation row: [{deform_snippet}]\n"
        "Associate your description with these values."
    )
    content.append({"type": "text", "text": csv_block})
    return [{"role": "user", "content": content}]

# ── Main routine ───────────────────────────────────────────────────────────────
def run_multimodal_episode(
    img_dir: str,
    force_csv: str,
    deform_csv: str,
    prompt_text: str = (
        "You are given three synchronized images of the same time‑step in a "
        "robotic manipulation simulation plus the corresponding force‑torque "
        "and object‑deformation readings.\n"
        "Describe the scene with clear spatial, geometric, and interaction "
        "detail so the caption can align with the sensor data. Focus on:\n"
        "• robot posture and gripper pose\n• object shape and orientation\n"
        "• contact locations and any visible deformation\n"
        "• environment cues (floor grid, bounding box)\n"
        "Stay under 50 words; do not guess unobservable measurements."
    ),
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 60,
):
    # read CSVs
    force_rows  = read_csv_rows(Path(force_csv))
    deform_rows = read_csv_rows(Path(deform_csv))

    assert len(force_rows) == len(deform_rows) == 3, \
        "CSVs must have exactly 3 rows."

    # gather images for each t
    img_dir = Path(img_dir)
    time_step_imgs = [
        sorted(img_dir.glob(f"999Crayola_Crayons_24_count{t}Camera*.png")) for t in range(3)
    ]
    for t, imgs in enumerate(time_step_imgs):
        assert len(imgs) == 3, f"Need 3 images for t{t} (found {len(imgs)})"

    captions = []
    for t in range(3):
        # encode images
        imgs64 = [encode_image_to_base64(p) for p in time_step_imgs[t]]

        # build chat messages
        messages = build_messages(prompt_text,
                                  imgs64,
                                  force_rows[t],
                                  deform_rows[t])

        # call OpenAI
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        caption = resp.choices[0].message.content
        captions.append((t, caption))

    # print
    for t, cap in captions:
        print(f"\nTime‑step t{t} caption ⟶ {cap}")

    return captions

# ── Run example ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_multimodal_episode(
        img_dir="data",
        force_csv="data/forces.csv",
        deform_csv="data/deformation.csv"
    )
