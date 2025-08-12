import os

from load_object_data import load_object_sheet 

# CSVファイルを読み込む
file_path = 'src/object_sheet.csv'
object_data_list = load_object_sheet(file_path)

tsdf_list = ["001_chips_can", "023_wine_glass", "029_plate", "033_spatula"]
google16k_list = ["003_cracker_box", "022_windex_bottle", "028_skillet_lid", "029_plate", "030_fork", "031_spoon", "032_knife", "035_power_drill", "036_wood_block", "037_scissors", "038_padlock", "040_large_marker", "042_adjustable_wrench", "043_phillips_screwdriver", "044_flat_screwdriver", "048_hammers", "049_small_clamp", "050_miduim_clamp", "051_large_clamp", "052_exstra_large_clamp", "053_mini_soccer_ball", "054_softtball", "055_baseball", "056_tennis_ball", "057_racquet_ball", "058_golf_ball", "059_chain"]

for object_data in object_data_list:
    object_id = object_data['id']
    link_dir = f"data/objects/{object_id}"
    link_path = os.path.join(link_dir, "model.obj")

    if os.path.islink(link_path):
        os.remove(link_path)

    if object_id in tsdf_list:
        target_relative_path = "tsdf/textured.obj"
    elif object_id in google16k_list:
        target_relative_path = "google_16k/textured.obj"
    else:
        target_relative_path = "poisson/textured.obj"
    
    # シンボリックリンクを作成
    os.symlink(target_relative_path, link_path)
    print(f"Created symlink for {object_id}: model.obj -> {target_relative_path}")

