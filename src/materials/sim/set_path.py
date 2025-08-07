import os

def set_path(object_name, coup_friction, material_type):
    frc = f"{int(round(coup_friction * 100)):03d}"
    video_object_path = f"data/videos/{object_name}"
    outfile_object_path = f"data/csv/{object_name}"
    photo_object_path = f"data/photos/{object_name}"
    photo_material_path = f"{photo_object_path}/{material_type}"
    photo_friction_path = f"{photo_material_path}/{frc}"

    if not os.path.exists(video_object_path):
        os.makedirs(video_object_path)
    if not os.path.exists(outfile_object_path):
        os.makedirs(outfile_object_path)
    if not os.path.exists(photo_object_path):
        os.makedirs(photo_object_path)
    if not os.path.exists(photo_material_path):
        os.makedirs(photo_material_path)
    if not os.path.exists(photo_friction_path):
        os.makedirs(photo_friction_path)
    default_video_path = f"{video_object_path}/{object_name}_{material_type}_{frc}.mp4"
    default_outfile_path = f"{outfile_object_path}/{object_name}_{material_type}_{frc}.csv"
    base_photo_name = f"{photo_friction_path}/{object_name}_{material_type}_{frc}"
    
    return default_video_path, default_outfile_path, base_photo_name