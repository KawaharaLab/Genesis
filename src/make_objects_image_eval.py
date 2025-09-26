import random
from pathlib import Path
from datetime import datetime
import os
import imageio.v3 as iio
import numpy as np

import torch

import genesis as gs


DATA_ROOT = Path("/home/user/Genesis/data")

MATERIAL_TYPE = "Rigid" # Rigid or Elastic
# DATA_TYPE = os.environ['DATA_TYPE']
DATA_TYPE = "eval"  # Options: "train", "eval"

## -------------------------- PATH SETUP -------------------------- ##

def setup_paths(object_name: str, target_choice: str) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "eval" in DATA_TYPE:
        input_obj_path = DATA_ROOT / "objects" / object_name / "model.obj"
    else:
        input_obj_path = DATA_ROOT / "objects/gso" / object_name / "model.obj"
    if not input_obj_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_obj_path}")

    return {
        "input_obj": input_obj_path,
        "object_name": object_name,
    }
    
def get_obj_bounding_box(obj_path):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    with open(obj_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                x, y, z = float(x), float(y), float(z)
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
    return [max_x - min_x, max_y - min_y, max_z - min_z]

def set_grasp(obj_path):
    GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH = 0.002, 0.075
    bbox = get_obj_bounding_box(obj_path)
    scale = 1.0
    if GRIPPER_MIN_WIDTH < bbox[0] < GRIPPER_MAX_WIDTH:
        euler = (0, 0, 90)
    elif GRIPPER_MIN_WIDTH < bbox[1] < GRIPPER_MAX_WIDTH:
        euler = (0, 0, 0)
    else:
        scale = (0.080 / bbox[0]) if bbox[0] < bbox[1] else (0.080 / bbox[1])
        euler = (0, 0, 90) if bbox[0] < bbox[1] else (0, 0, 0)
    return scale, euler

def main(target_choice: str = 'none'):
    
    if torch.cuda.is_available():
        gs.init(backend=gs.gpu)
    else:
        gs.init(backend=gs.cpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-2, substeps=5),
        viewer_options=gs.options.ViewerOptions(camera_pos=(3,-1,1.5), camera_lookat=(0,0,0), camera_fov=30),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))    
    
    if "eval" in DATA_TYPE:
        objects_dir = DATA_ROOT / "objects"
    else:
        objects_dir = DATA_ROOT / "objects/gso"
    if not objects_dir.exists():
        print(f"❌ Error: Input directory '{objects_dir}' not found."); return []
    object_names = [d.name for d in objects_dir.iterdir() if d.is_dir()]
    print(f"🔍 Found {len(object_names)} objects in '{objects_dir}'.")
    
    x_diff = 0.15
    y_diff = 0.15
    pos = (0.0, 0.0, 0.001)
    
    valid_objects = []
    for object_name in object_names:
        try:
            paths = setup_paths(object_name, target_choice)
            valid_objects.append(object_name)
        except FileNotFoundError:
            print(f"❌ Skipping non-existent file: {object_name}")
            continue
    
    valid_objects = valid_objects 

    num_objects = len(valid_objects)
    if num_objects == 0:
        print("No valid objects to place.")
        return
    
    # 1. グリッドサイズを計算
    nx = int(np.ceil(np.sqrt(num_objects * 2 / 4)))
    ny = int(np.ceil(num_objects / nx))

    # 2. 物体の配置範囲を計算
    x_min = 0.0
    x_max = (nx - 1) * x_diff
    y_min = 0.0
    y_max = (ny - 1) * y_diff
    z_min = 0.001

    # 3. 注視点 (lookat) を計算
    original_lookat = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
    
    # 4. カメラ位置 (pos) を計算
    half_range_y = y_max - original_lookat[1]
    
    horizontal_dist = half_range_y
    z_offset = horizontal_dist * 0.5

    cam_x = original_lookat[0]
    cam_y = original_lookat[1] - 2.5 * half_range_y
    
    original_cam_pos = np.array([cam_x, cam_y, z_min + z_offset])
    
    # ------------------ 修正箇所 ------------------
    # 1. lookatのx, yを(0,0)にするための移動量を計算
    translation_x = -original_lookat[0]
    translation_y = -original_lookat[1]
    
    # 2. 新しいlookatとカメラ位置を計算
    new_lookat = np.array([0.0, 0.0, original_lookat[2]])
    new_cam_pos = original_cam_pos + np.array([translation_x, translation_y, 0])

    print(f"Grid size: {nx}x{ny}")
    print(f"Object Placement Area: x({x_min}, {x_max}), y({y_min}, {y_max})")
    print(f"Original Camera Position: {original_cam_pos}, Original Lookat: {original_lookat}")
    print(f"New Camera Position: {new_cam_pos}, New Lookat: {new_lookat}")
    print(f"Translation applied: ({translation_x}, {translation_y})")


    # 5. カメラをセット
    # numpy配列をタプルに変換して渡す
    cam = scene.add_camera(res=(1280, 960), pos=new_cam_pos, lookat=new_lookat, fov=30)
    
    # 6. オブジェクトを配置
    for i, object_name in enumerate(valid_objects):
        paths = setup_paths(object_name, target_choice)
        object_scale, object_euler = set_grasp(str(paths['input_obj']))
        color = random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)])
        
        # グリッドの座標を計算
        x_idx = i % nx
        y_idx = i // nx
        
        original_pos = (x_idx * x_diff, y_idx * y_diff, pos[2])
        
        # 3. オブジェクトの位置にも並行移動を適用
        new_pos = (original_pos[0] + translation_x, original_pos[1] + translation_y, original_pos[2])
        
        scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(file=str(paths['input_obj']), scale=object_scale, pos=new_pos, euler=object_euler),
            surface=gs.surfaces.Default(color=color)
        ) 

    scene.build()
    for _ in range(5):
        scene.step()
    
    rgb, _, _, _ = cam.render(rgb=True)
    filepath = "camera_test.png"
    iio.imwrite(filepath, rgb)
    
    return

if __name__ == "__main__":
    main("none")
