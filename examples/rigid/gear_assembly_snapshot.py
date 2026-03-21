import argparse
import math
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
import trimesh

import genesis as gs

GEAR_DIR = Path(__file__).resolve().parents[2] / "genesis" / "assets" / "meshes" / "gears"

def shoelace_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def mesh_bounds(stl_path: Path):
    mesh = trimesh.load(stl_path, force="mesh")
    mins, maxs = mesh.bounds
    return mesh, mins.astype(np.float64), maxs.astype(np.float64)

def detect_holes_xy(plate_mesh: trimesh.Trimesh):
    z_min, z_max = plate_mesh.bounds[:, 2]
    z_mid = 0.5 * (z_min + z_max)

    section = plate_mesh.section(plane_origin=[0.0, 0.0, z_mid], plane_normal=[0.0, 0.0, 1.0])
    if section is None:
        return []

    loops = [np.asarray(loop, dtype=np.float64) for loop in section.discrete if len(loop) >= 4]
    if len(loops) <= 1:
        return []

    areas = [abs(shoelace_area(loop[:, :2])) for loop in loops]
    outer_idx = int(np.argmax(areas))

    holes = []
    for i, loop in enumerate(loops):
        if i == outer_idx:
            continue
        pts = loop[:, :3]
        center = pts.mean(axis=0)
        area = max(areas[i], 1e-12)
        radius = np.sqrt(area / np.pi)
        holes.append((center, radius))

    holes.sort(key=lambda h: h[1], reverse=True)
    return holes[:4]

def default_holes_from_bbox(plate_min: np.ndarray, plate_max: np.ndarray):
    center_xy = 0.5 * (plate_min[:2] + plate_max[:2])
    z_mid = 0.5 * (plate_min[2] + plate_max[2])
    span = plate_max[:2] - plate_min[:2]
    offs = 0.28 * span
    return [
        (np.array([center_xy[0] + offs[0], center_xy[1] + offs[1], z_mid]), min(span) * 0.08),
        (np.array([center_xy[0] + offs[0], center_xy[1] - offs[1], z_mid]), min(span) * 0.08),
        (np.array([center_xy[0] - offs[0], center_xy[1] + offs[1], z_mid]), min(span) * 0.08),
        (np.array([center_xy[0] - offs[0], center_xy[1] - offs[1], z_mid]), min(span) * 0.08),
    ]

def select_gear_hole_centers(holes):
    centers = [np.asarray(center_xyz, dtype=np.float64) for center_xyz, _ in holes]
    if len(centers) <= 3:
        return centers

    xy = np.stack([c[:2] for c in centers], axis=0)
    xy_mean = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - xy_mean, full_matrices=False)
    axis = vt[0]
    proj = (xy - xy_mean) @ axis

    order = np.argsort(proj)
    ordered_centers = [centers[i] for i in order]
    proj_sorted = proj[order]

    if len(ordered_centers) != 4:
        return ordered_centers[:3]

    gaps = np.diff(proj_sorted)
    screw_is_start = gaps[0] <= gaps[-1]

    if screw_is_start:
        return ordered_centers[1:4]

    return [ordered_centers[2], ordered_centers[1], ordered_centers[0]]

def adjust_chain_centers(centers, distance_offset: float):
    c0 = np.asarray(centers[0], dtype=np.float64).copy()
    c1 = np.asarray(centers[1], dtype=np.float64).copy()
    c2 = np.asarray(centers[2], dtype=np.float64).copy()

    axis = c1[:2] - c0[:2]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        axis = np.array([0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / norm

    d01 = float(np.linalg.norm(c1[:2] - c0[:2])) + float(distance_offset)
    d12 = float(np.linalg.norm(c2[:2] - c1[:2])) + float(distance_offset)

    c1[:2] = c0[:2] + axis * d01
    c2[:2] = c1[:2] + axis * d12
    c1[2] = c0[2]
    c2[2] = c0[2]
    return [c0, c1, c2]

def yaw_quat_deg(yaw_deg: float):
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))

def yaw_rotmat(yaw_deg: float):
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def outer_loop_xy(mesh: trimesh.Trimesh):
    z_min, z_max = mesh.bounds[:, 2]
    best_loop = None
    best_area = -1.0

    for z in np.linspace(z_min + 1e-3, z_max - 1e-3, 9):
        section = mesh.section(plane_origin=[0.0, 0.0, float(z)], plane_normal=[0.0, 0.0, 1.0])
        if section is None or not section.discrete:
            continue
        loops = [np.asarray(loop[:, :2], dtype=np.float64) for loop in section.discrete if len(loop) >= 4]
        if not loops:
            continue
        areas = [abs(shoelace_area(lp)) for lp in loops]
        i = int(np.argmax(areas))
        if areas[i] > best_area:
            best_area = areas[i]
            best_loop = loops[i]

    return best_loop

def mid_section_loops_xy(mesh: trimesh.Trimesh):
    mins, maxs = mesh.bounds
    z_mid = 0.5 * float(mins[2] + maxs[2])
    section = mesh.section(plane_origin=[0.0, 0.0, z_mid], plane_normal=[0.0, 0.0, 1.0])
    if section is None or not section.discrete:
        return []
    return [np.asarray(loop[:, :2], dtype=np.float64) for loop in section.discrete if len(loop) >= 4]

def detect_shaft_center_local(mesh: trimesh.Trimesh):
    mins, maxs = mesh.bounds
    center = 0.5 * (mins.astype(np.float64) + maxs.astype(np.float64))
    loops = mid_section_loops_xy(mesh)
    if loops:
        areas = [abs(shoelace_area(lp)) for lp in loops]
        outer_idx = int(np.argmax(areas))
        cxy = loops[outer_idx].mean(axis=0)
        center[0] = float(cxy[0])
        center[1] = float(cxy[1])
    return center

def detect_gear_center_local(mesh: trimesh.Trimesh):
    mins, maxs = mesh.bounds
    center = 0.5 * (mins.astype(np.float64) + maxs.astype(np.float64))
    z_mid = center[2]

    section = mesh.section(plane_origin=[0.0, 0.0, float(z_mid)], plane_normal=[0.0, 0.0, 1.0])
    if section is None or not section.discrete:
        return center

    loops = [np.asarray(loop[:, :2], dtype=np.float64) for loop in section.discrete if len(loop) >= 4]
    if len(loops) <= 1:
        return center

    areas = [abs(shoelace_area(lp)) for lp in loops]
    outer_idx = int(np.argmax(areas))
    hole_candidates = [i for i in range(len(loops)) if i != outer_idx]
    if not hole_candidates:
        return center

    # use the largest inner loop as shaft hole center
    hole_idx = max(hole_candidates, key=lambda i: areas[i])
    cxy = loops[hole_idx].mean(axis=0)
    center[0] = float(cxy[0])
    center[1] = float(cxy[1])
    return center

def points_in_poly(points: np.ndarray, poly: np.ndarray):
    x = points[:, 0]
    y = points[:, 1]
    xp = poly[:, 0]
    yp = poly[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = (yp[i] > y) != (yp[j] > y)
        xints = (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i] + 1e-12) + xp[i]
        inside ^= cond & (x < xints)
        j = i
    return inside

def _cross2d(a: np.ndarray, b: np.ndarray):
    return float(a[0] * b[1] - a[1] * b[0])

def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, eps: float = 1e-9):
    r = p2 - p1
    s = q2 - q1
    rxs = _cross2d(r, s)
    qpxr = _cross2d(q1 - p1, r)

    if abs(rxs) < eps and abs(qpxr) < eps:
        rr = float(np.dot(r, r))
        if rr < eps:
            return float(np.linalg.norm(p1 - q1)) < eps
        t0 = float(np.dot(q1 - p1, r) / rr)
        t1 = float(np.dot(q2 - p1, r) / rr)
        tmin, tmax = min(t0, t1), max(t0, t1)
        return tmax >= -eps and tmin <= 1.0 + eps

    if abs(rxs) < eps:
        return False

    t = _cross2d(q1 - p1, s) / rxs
    u = _cross2d(q1 - p1, r) / rxs
    return -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps

def polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray):
    na, nb = len(poly_a), len(poly_b)
    for i in range(na):
        a1 = poly_a[i]
        a2 = poly_a[(i + 1) % na]
        for j in range(nb):
            b1 = poly_b[j]
            b2 = poly_b[(j + 1) % nb]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    if points_in_poly(poly_a[:1], poly_b)[0]:
        return True
    if points_in_poly(poly_b[:1], poly_a)[0]:
        return True
    return False

def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def polygon_min_distance(poly_a: np.ndarray, poly_b: np.ndarray):
    dmin = 1e12
    for i in range(len(poly_a)):
        p = poly_a[i]
        for j in range(len(poly_b)):
            a = poly_b[j]
            b = poly_b[(j + 1) % len(poly_b)]
            dmin = min(dmin, point_segment_distance(p, a, b))
    for i in range(len(poly_b)):
        p = poly_b[i]
        for j in range(len(poly_a)):
            a = poly_a[j]
            b = poly_a[(j + 1) % len(poly_a)]
            dmin = min(dmin, point_segment_distance(p, a, b))
    return float(dmin)

def radial_distance_to_loop(loop_xy: np.ndarray, center_xy: np.ndarray, angle_local: float):
    d = np.array([math.cos(angle_local), math.sin(angle_local)], dtype=np.float64)
    best_t = 0.0
    n = len(loop_xy)
    for i in range(n):
        a = loop_xy[i]
        b = loop_xy[(i + 1) % n]
        e = b - a
        den = _cross2d(d, e)
        if abs(den) < 1e-12:
            continue
        t = _cross2d(a - center_xy, e) / den
        u = _cross2d(a - center_xy, d) / den
        if t >= 0.0 and 0.0 <= u <= 1.0:
            best_t = max(best_t, float(t))
    return best_t

def gear_pose_from_center(center_world: np.ndarray, local_center: np.ndarray, yaw_deg: float):
    rot = yaw_rotmat(yaw_deg)
    pos = center_world - rot @ local_center
    quat = yaw_quat_deg(yaw_deg)
    return pos, quat

def world_loop_from_pose(loop_local: np.ndarray, center_world: np.ndarray, local_center: np.ndarray, yaw_deg: float):
    pos, _ = gear_pose_from_center(center_world, local_center, yaw_deg)
    rot = yaw_rotmat(yaw_deg)[:2, :2]
    return (loop_local @ rot.T) + pos[:2]

def find_meshing_yaw(
    prev_loop,
    prev_center_local,
    prev_center_world,
    prev_yaw_deg,
    cur_loop,
    cur_center_local,
    cur_center_world,
    line_backlash,
    yaw_samples,
):
    dvec = cur_center_world[:2] - prev_center_world[:2]
    dist = float(np.linalg.norm(dvec))
    alpha = math.atan2(dvec[1], dvec[0])

    prev_angle_local = alpha - math.radians(prev_yaw_deg)
    r_prev = radial_distance_to_loop(prev_loop, prev_center_local[:2], prev_angle_local)

    best_yaw = 0.0
    best_score = 1e18
    best_gap = None

    n = max(24, int(yaw_samples))
    for yaw in np.linspace(0.0, 360.0, n, endpoint=False):
        cur_angle_local = alpha + math.pi - math.radians(float(yaw))
        r_cur = radial_distance_to_loop(cur_loop, cur_center_local[:2], cur_angle_local)

        line_gap = dist - (r_prev + r_cur)
        # Fast score: prioritize non-penetrating gap and target backlash.
        if line_gap < 0.0:
            score = 1e6 + abs(line_gap)
        else:
            score = abs(line_gap - line_backlash)

        if score < best_score:
            best_score = score
            best_yaw = float(yaw)
            best_gap = line_gap

    if best_gap is not None:
        print(
            f"Meshing line gap (target={line_backlash:.4f}) -> selected={best_gap:.6f} "
            f"(samples={n})"
        )

    return best_yaw

def report_gear_contacts(gear_entities, gear_names):
    print("=== Gear contact check (simulation data) ===")
    any_contact = False
    for i in range(len(gear_entities)):
        for j in range(i + 1, len(gear_entities)):
            contacts = gear_entities[i].get_contacts(with_entity=gear_entities[j])
            n_contacts = int(contacts["geom_a"].shape[0])
            force_sum = 0.0
            if n_contacts > 0:
                force_sum = float(np.linalg.norm(contacts["force_a"].detach().cpu().numpy(), axis=-1).sum())
                any_contact = True
            print(
                f"{gear_names[i]} <-> {gear_names[j]}: contacts={n_contacts}, total_contact_force={force_sum:.6f}"
            )
    if not any_contact:
        print("No gear-gear contacts detected at this step.")

def place_mesh_with_color(
    scene,
    stl_name: str,
    pos: np.ndarray,
    color,
    quat=None,
    fixed: bool = True,
    decimate: bool = False,
    scale=1.0,
):
    mesh_kwargs = dict(
        file=f"meshes/gears/{stl_name}",
        pos=tuple(pos),
        fixed=fixed,
        collision=True,
        decimate=decimate,
        convexify=False,
        scale=scale,
    )
    if quat is not None:
        mesh_kwargs["quat"] = tuple(quat)

    return scene.add_entity(
        gs.morphs.Mesh(**mesh_kwargs),
        surface=gs.surfaces.Rough(color=color),
    )

def bounds_after_translation(mins: np.ndarray, maxs: np.ndarray, pos: np.ndarray):
    return mins + pos, maxs + pos

def render_rgb_image(cam):
    rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

def build_camera(scene, center, radius, view: str, vis: bool):
    fov = 42.0
    if view == "top":
        pos = center + np.array([0.0, 0.0, 1.35 * radius], dtype=np.float64)
    elif view == "side":
        pos = center + np.array([0.0, -1.55 * radius, 0.35 * radius], dtype=np.float64)
    else:
        raise ValueError(f"Unknown view: {view}")

    dist = np.linalg.norm(pos - center)
    near = max(0.05, dist - 2.5 * radius)
    far = dist + 3.5 * radius + 10.0

    return scene.add_camera(
        res=(1280, 960),
        pos=tuple(pos),
        lookat=tuple(center),
        fov=fov,
        near=near,
        far=far,
        GUI=vis,
    )

def save_mp4(frames, path: Path, fps: int = 30):
    out_path = path.resolve()
    with imageio.get_writer(out_path, fps=max(1, int(fps)), codec="libx264", quality=8) as writer:
        for fr in frames:
            if isinstance(fr, Image.Image):
                arr = np.asarray(fr)
            else:
                arr = np.asarray(fr)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            writer.append_data(arr)

def validate_initial_non_penetration(shaft_world_centers, gear_mesh_data, gear_yaws, ignore_indices=()):
    ignore = set(int(i) for i in ignore_indices)
    world_loops = []
    for i in range(3):
        if i in ignore:
            continue
        _, _, _, local_center, loop_local = gear_mesh_data[i]
        if loop_local is None:
            continue
        world_loop = world_loop_from_pose(loop_local, shaft_world_centers[i], local_center, gear_yaws[i])
        world_loops.append((i, world_loop))

    for a in range(len(world_loops)):
        for b in range(a + 1, len(world_loops)):
            ia, loop_a = world_loops[a]
            ib, loop_b = world_loops[b]
            if polygons_intersect(loop_a, loop_b):
                raise RuntimeError(f"Initial penetration detected between gears index {ia} and {ib}.")
            gap = polygon_min_distance(loop_a, loop_b)
            print(f"Initial gap gear[{ia}]<->gear[{ib}] = {gap:.6f}")

def validate_height_alignment(shaft_world_centers, tol=1e-3):
    z_values = np.array([c[2] for c in shaft_world_centers], dtype=np.float64)
    dz = float(z_values.max() - z_values.min())
    print(f"Gear shaft z spread: {dz:.6f}")
    if dz > tol:
        raise RuntimeError(f"Gear height mismatch too large: {dz:.6f} > tol={tol:.6f}")

def validate_rigid_entities(gear_entities, gear_names):
    for ent, name in zip(gear_entities, gear_names):
        nd = int(getattr(ent, "n_dofs", 0))
        is_fixed = nd < 1
        print(f"{name}: n_dofs={nd}, inferred_fixed={is_fixed}")

def estimate_shaft_and_hole_radii(shaft_mesh: trimesh.Trimesh, medium_mesh: trimesh.Trimesh):
    shaft_loops = mid_section_loops_xy(shaft_mesh)
    medium_loops = mid_section_loops_xy(medium_mesh)

    shaft_r = None
    hole_r = None

    if shaft_loops:
        shaft_areas = [abs(shoelace_area(lp)) for lp in shaft_loops]
        shaft_outer = shaft_loops[int(np.argmax(shaft_areas))]
        shaft_r = float(np.sqrt(max(abs(shoelace_area(shaft_outer)), 1e-12) / np.pi))

    if len(medium_loops) >= 2:
        areas = [abs(shoelace_area(lp)) for lp in medium_loops]
        outer_idx = int(np.argmax(areas))
        inner_indices = [i for i in range(len(medium_loops)) if i != outer_idx]
        if inner_indices:
            hole_idx = max(inner_indices, key=lambda i: areas[i])
            hole_r = float(np.sqrt(max(areas[hole_idx], 1e-12) / np.pi))

    return shaft_r, hole_r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="gear_assembly_snapshot.png")
    parser.add_argument("--top_video_output", type=str, default="gear_drive_top.mp4")
    parser.add_argument("--side_video_output", type=str, default="gear_drive_side.mp4")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--drop_height", type=float, default=20.0)
    parser.add_argument("--settle_steps", type=int, default=120)
    parser.add_argument("--center_distance_offset", type=float, default=0.0)
    parser.add_argument("--line_backlash", type=float, default=0.05)
    parser.add_argument("--yaw_samples", type=int, default=120)
    parser.add_argument("--only_medium_drop", action="store_true", default=False)
    parser.add_argument("--shaft_scale", type=float, default=0.98)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    plate_mesh, plate_min, plate_max = mesh_bounds(GEAR_DIR / "Gear_Plate.STL")
    shaft_mesh, shaft_min, shaft_max = mesh_bounds(GEAR_DIR / "Gear_Shaft.STL")
    plate_center = 0.5 * (plate_min + plate_max)
    scene_shift = -plate_center

    holes = detect_holes_xy(plate_mesh)
    if not holes:
        holes = default_holes_from_bbox(plate_min, plate_max)
    shaft_center = detect_shaft_center_local(shaft_mesh)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, gravity=(0.0, 0.0, -300.0)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -180.0, 90.0),
            camera_lookat=(0.0, 0.0, -5.0),
            camera_fov=35,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(ambient_light=(1.0, 1.0, 1.0)),
        show_viewer=args.vis,
        renderer=gs.renderers.Rasterizer(),
    )

    world_mins = []
    world_maxs = []

    plate_world_min, plate_world_max = bounds_after_translation(plate_min, plate_max, scene_shift)
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, plate_world_min[2] - 20.0)))
    place_mesh_with_color(
        scene,
        "Gear_Plate.STL",
        scene_shift,
        color=(0.82, 0.82, 0.85, 1.0),
        fixed=True,
        decimate=True,
    )
    world_mins.append(plate_world_min)
    world_maxs.append(plate_world_max)

    shaft_z_offset = plate_max[2] - shaft_min[2]
    gear_hole_centers = select_gear_hole_centers(holes)
    if abs(float(args.center_distance_offset)) > 1e-12:
        gear_hole_centers = adjust_chain_centers(gear_hole_centers, args.center_distance_offset)
    else:
        print("Using detected plate hole centers without center-distance adjustment.")

    shaft_world_centers = []
    shaft_alignment_errors = []
    shaft_scale_xy = float(args.shaft_scale)
    for center_xyz in gear_hole_centers:
        shaft_pos = scene_shift.copy()
        # IMPORTANT: mesh scaling is applied around local origin.
        # Compensate with scaled local center so shaft axis stays on plate hole center.
        shaft_pos[:2] += center_xyz[:2] - shaft_scale_xy * shaft_center[:2]
        shaft_pos[2] += shaft_z_offset
        place_mesh_with_color(
            scene,
            "Gear_Shaft.STL",
            shaft_pos,
            color=(0.20, 0.90, 0.40, 1.0),
            fixed=True,
            decimate=True,
            scale=(shaft_scale_xy, shaft_scale_xy, 1.0),
        )
        smin, smax = bounds_after_translation(shaft_min, shaft_max, shaft_pos)
        world_mins.append(smin)
        world_maxs.append(smax)

        shaft_world_center = np.array(
            [
                shaft_pos[0] + shaft_scale_xy * shaft_center[0],
                shaft_pos[1] + shaft_scale_xy * shaft_center[1],
                shaft_pos[2] + shaft_center[2],
            ],
            dtype=np.float64,
        )
        shaft_world_centers.append(shaft_world_center)

        target_center = scene_shift + center_xyz
        shaft_alignment_errors.append(float(np.linalg.norm(shaft_world_center[:2] - target_center[:2])))

    gear_specs = [
        ("Gear_Large.STL", (0.92, 0.38, 0.26, 1.0)),
        ("Gear_Medium.STL", (0.22, 0.60, 0.92, 1.0)),
        ("Gear_Small.STL", (0.96, 0.82, 0.22, 1.0)),
    ]

    gear_mesh_data = []
    gear_mesh_objs = []
    for stl_name, _ in gear_specs:
        mesh, gear_min, gear_max = mesh_bounds(GEAR_DIR / stl_name)
        gear_center_local = detect_gear_center_local(mesh)
        loop_local = outer_loop_xy(mesh)
        gear_mesh_data.append((stl_name, gear_min, gear_max, gear_center_local, loop_local))
        gear_mesh_objs.append(mesh)

    # Fast init: skip expensive yaw optimization and use authored phase.
    # Medium gear is spawned above, so only fixed gears are checked for initial penetration.
    gear_yaws = [0.0, 0.0, 0.0]

    if args.only_medium_drop:
        validate_initial_non_penetration(shaft_world_centers, gear_mesh_data, gear_yaws, ignore_indices=(0, 1, 2))
    else:
        validate_initial_non_penetration(shaft_world_centers, gear_mesh_data, gear_yaws, ignore_indices=(1,))
    validate_height_alignment(shaft_world_centers, tol=1e-3)
    if shaft_alignment_errors:
        print(f"Shaft-to-hole XY alignment max error: {max(shaft_alignment_errors):.6f}")

    gear_entities = []
    gear_names = []
    active_indices = [1] if args.only_medium_drop else [0, 1, 2]
    for i in active_indices:
        stl_name, color = gear_specs[i]
        center_world = shaft_world_centers[i]
        _, gear_min, gear_max, gear_center_local, _ = gear_mesh_data[i]
        gear_pos, gear_quat = gear_pose_from_center(center_world, gear_center_local, gear_yaws[i])

        # medium gear only: spawn above shaft and let it descend during simulation
        gear_fixed = i != 1
        if i == 1:
            gear_pos = gear_pos.copy()
            gear_pos[2] += float(args.drop_height)

        gear = place_mesh_with_color(scene, stl_name, gear_pos, color=color, quat=gear_quat, fixed=gear_fixed)
        gear_entities.append(gear)
        gear_names.append(stl_name)

        gmin, gmax = bounds_after_translation(gear_min, gear_max, gear_pos)
        world_mins.append(gmin)
        world_maxs.append(gmax)

    shaft_xy = np.stack([c[:2] for c in shaft_world_centers], axis=0)
    center_xy = shaft_xy.mean(axis=0)
    center_z = float(np.mean([c[2] for c in shaft_world_centers]))
    scene_center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
    shaft_span = float(np.linalg.norm(shaft_xy.max(axis=0) - shaft_xy.min(axis=0)))
    max_gear_r = max([0.5 * float(gmax[0] - gmin[0]) for _, gmin, gmax, _, _ in gear_mesh_data])
    scene_radius = max(12.0, 0.75 * shaft_span + 1.1 * max_gear_r)

    cam_top = build_camera(scene, scene_center, scene_radius, view="top", vis=args.vis)
    cam_side = build_camera(scene, scene_center, scene_radius, view="side", vis=args.vis)

    scene.build()
    validate_rigid_entities(gear_entities, gear_names)

    medium_center_world = shaft_world_centers[1]
    shaft_xy_offset = np.linalg.norm(medium_center_world[:2] - shaft_world_centers[1][:2])
    print(
        f"Placement checks: only_medium_drop={args.only_medium_drop}, "
        f"drop_height={float(args.drop_height):.3f}, shaft_scale={float(args.shaft_scale):.4f}"
    )
    print(f"Center alignment (medium hole center vs shaft center, XY): {float(shaft_xy_offset):.6f}")

    shaft_r, hole_r = estimate_shaft_and_hole_radii(
        shaft_mesh=mesh_bounds(GEAR_DIR / 'Gear_Shaft.STL')[0], medium_mesh=gear_mesh_objs[1]
    )
    if shaft_r is not None and hole_r is not None:
        shaft_r_scaled = shaft_r * float(args.shaft_scale)
        clearance = hole_r - shaft_r_scaled
        print(
            f"Radius check: medium_hole_r={hole_r:.6f}, shaft_r_scaled={shaft_r_scaled:.6f}, "
            f"clearance={clearance:.6f}"
        )

    for _ in range(20):
        scene.step()
    report_gear_contacts(gear_entities, gear_names)

    rgb = render_rgb_image(cam_top)
    output_path = Path(args.output).resolve()
    Image.fromarray(rgb).save(output_path)

    n_frames = max(2, int(args.frames))

    print(
        f"Dropping medium gear from +{float(args.drop_height):.3f} in z, "
        f"recording {n_frames} frames"
    )

    top_frames = []
    side_frames = []
    for _ in range(n_frames):
        scene.step()
        top_frames.append(Image.fromarray(render_rgb_image(cam_top)))
        side_frames.append(Image.fromarray(render_rgb_image(cam_side)))

    for _ in range(max(0, int(args.settle_steps))):
        scene.step()

    top_video_path = Path(args.top_video_output).resolve()
    side_video_path = Path(args.side_video_output).resolve()
    save_mp4(top_frames, top_video_path, fps=30)
    save_mp4(side_frames, side_video_path, fps=30)

    print(f"Shafts placed: {len(gear_hole_centers)}")
    print(f"Scene bounds min: {np.min(np.stack(world_mins, axis=0), axis=0)}, max: {np.max(np.stack(world_maxs, axis=0), axis=0)}")
    print(f"Gear yaws (initial): {gear_yaws}")
    print(f"Saved snapshot to: {output_path}")
    report_gear_contacts(gear_entities, gear_names)
    print(f"Saved top-view animation to: {top_video_path}")
    print(f"Saved side-view animation to: {side_video_path}")

if __name__ == "__main__":
    main()