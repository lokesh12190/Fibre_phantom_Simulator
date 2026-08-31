import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from scipy.interpolate import splprep, splev
import nibabel as nib

# Define the attenuation coefficients
# NOTE: ATTENUATION_FIBER was originally 255.0 - testing confirmed this extreme
# contrast (vs resin=100) combined with the very thin (1-voxel) fibres was a
# major cause of streak/starburst artifacts in reconstruction (std dev 11.877
# at fiber=255 vs 3.264 at fiber=130, in a controlled A/B test). Lowered to
# 130 - still clearly separable from resin for later segmentation, but far
# less extreme contrast.
ATTENUATION_AIR = 0.0
ATTENUATION_RESIN = 100.0
ATTENUATION_FIBER = 130.0

# Slicer reads .nii
def save_as_nifti(array, file_path):
    nifti_img = nib.Nifti1Image(array, affine=np.eye(4))  
    nib.save(nifti_img, file_path)

def is_within_bounds(point, dimensions, radius):
    x, y, z = point
    return (radius <= x < dimensions[0] - radius and
            radius <= y < dimensions[1] - radius and
            radius <= z < dimensions[2] - radius)

def is_within_pipe(point, center_y, center_z, pipe_radius):
    y, z = point[1], point[2]
    return ((y - center_y)**2 + (z - center_z)**2) <= pipe_radius**2

def can_place_sphere(center, volume, radius, pipe_radius=50):
    center_y, center_z = volume.shape[1] // 2, volume.shape[2] // 2
    cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    z0, z1 = cz - radius, cz + radius + 1

    if x0 < 0 or y0 < 0 or z0 < 0 or x1 > volume.shape[0] or y1 > volume.shape[1] or z1 > volume.shape[2]:
        return False

    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    zs = np.arange(z0, z1) - cz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    sphere_mask = X**2 + Y**2 + Z**2 <= radius**2

    ys_abs = np.arange(y0, y1) - center_y
    zs_abs = np.arange(z0, z1) - center_z
    Yp, Zp = np.meshgrid(ys_abs, zs_abs, indexing='ij')
    pipe_mask_2d = (Yp**2 + Zp**2) <= pipe_radius**2
    pipe_mask = np.broadcast_to(pipe_mask_2d[None, :, :], sphere_mask.shape)

    sub_vol = volume[x0:x1, y0:y1, z0:z1]
    fail_mask = sphere_mask & (~pipe_mask | (sub_vol == ATTENUATION_FIBER))
    return not np.any(fail_mask)

def add_voxel_sphere_to_volume(volume, center, radius, pipe_radius=50, intensity=ATTENUATION_FIBER):
    center_y, center_z = volume.shape[1] // 2, volume.shape[2] // 2
    cx, cy, cz = int(center[0]), int(center[1]), int(center[2])
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, volume.shape[0])
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, volume.shape[1])
    z0, z1 = max(cz - radius, 0), min(cz + radius + 1, volume.shape[2])
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return

    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    zs = np.arange(z0, z1) - cz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    sphere_mask = X**2 + Y**2 + Z**2 <= radius**2

    ys_abs = np.arange(y0, y1) - center_y
    zs_abs = np.arange(z0, z1) - center_z
    Yp, Zp = np.meshgrid(ys_abs, zs_abs, indexing='ij')
    pipe_mask_2d = (Yp**2 + Zp**2) <= pipe_radius**2
    pipe_mask = np.broadcast_to(pipe_mask_2d[None, :, :], sphere_mask.shape)

    write_mask = sphere_mask & pipe_mask
    volume[x0:x1, y0:y1, z0:z1][write_mask] = intensity

def update_volume_with_filament(volume, filament, radius, pipe_radius=50, intensity=ATTENUATION_FIBER):
    for point in filament:
        add_voxel_sphere_to_volume(volume, point, radius, pipe_radius, intensity)

def fill_pipe_with_resin(volume, pipe_radius=50):
    # Vectorized: was a 295M-iteration Python triple loop for an 820x600x600
    # volume - this is very likely the actual runtime bottleneck, not fibre
    # placement. Now a single vectorized numpy operation.
    center_y, center_z = volume.shape[1] // 2, volume.shape[2] // 2
    ys = np.arange(volume.shape[1]) - center_y
    zs = np.arange(volume.shape[2]) - center_z
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    pipe_mask_2d = (Y**2 + Z**2) <= pipe_radius**2  # shape (Ny, Nz)

    fill_mask = (volume == 0.0) & pipe_mask_2d[np.newaxis, :, :]
    volume[fill_mask] = ATTENUATION_RESIN

def generate_radius_normal(radius_range, mean, std_dev):
    radius = np.random.normal(loc=mean, scale=std_dev)
    # Clamp the value between the min and max of the range
    radius = max(radius_range[0], min(radius_range[1], radius))
    return int(radius)


def generate_and_count_filaments(volume, num_filaments, generator, defect_generator, pipe_radius=50, min_length=512, max_length=512, radius_range=(1, 6), bias=0.90, preferred_direction=[1, 0, 0], cluster_centers=None, cluster_radii=None, cluster_percentages=None, orientation_jitter_deg=0.0):
    # NOTE: previously defaulted to hard-coded clusters (50% of all fibres
    # forced into one tight clump near [180,180,180]) whenever the caller
    # didn't explicitly pass cluster params - main.py never did, so every
    # run was silently clustering fibres into dense clumps. Default is now
    # no clustering (uniform placement across the pipe cross-section) unless
    # cluster params are explicitly provided.
    cluster_centers = cluster_centers or []
    cluster_radii = cluster_radii or []
    cluster_percentages = cluster_percentages or []

    if len(cluster_centers) != len(cluster_radii) or len(cluster_centers) != len(cluster_percentages):
        raise ValueError("The number of cluster centers, radii, and percentages must be the same")

    filaments_per_cluster = [int((p / 100) * num_filaments) for p in cluster_percentages]
    
    successful_filaments = 0
    filaments = []
    total_attempts = 0
    # Was a fixed 10000 regardless of num_filaments - fine for small counts,
    # but blocks ever reaching higher filament counts (collision-rejection
    # failure rate rises with density, so more filaments genuinely need more
    # attempt budget, not just a proportional amount).
    max_total_attempts = max(10000, num_filaments * 30)

    current_cluster_idx = 0
    filaments_in_cluster = 0

    while successful_filaments < num_filaments and total_attempts < max_total_attempts:
        if current_cluster_idx < len(cluster_centers):
            if filaments_in_cluster < filaments_per_cluster[current_cluster_idx]:
                generator.cluster_center = cluster_centers[current_cluster_idx]
                generator.cluster_radius = cluster_radii[current_cluster_idx]
                filaments_in_cluster += 1
            else:
                current_cluster_idx += 1
                filaments_in_cluster = 0
        else:
            generator.cluster_center = None
            generator.cluster_radius = None
        
        # uniform distribution
        # filament_radius = random.randint(radius_range[0], radius_range[1])

        # normal distribution
        mean = (radius_range[0] + radius_range[1]) / 2
        filament_radius = generate_radius_normal(radius_range, mean=mean, std_dev=0.5)

        filament = generate_3d_filament(volume, generator, min_length, max_length, filament_radius, pipe_radius, bias, preferred_direction, orientation_jitter_deg)

        if filament is not None:
            update_volume_with_filament(volume, filament, filament_radius, pipe_radius, ATTENUATION_FIBER)
            filaments.append(filament)
            successful_filaments += 1

        total_attempts += 1

    if successful_filaments < num_filaments:
        print(f"Warning: Only able to place {successful_filaments} filaments after {total_attempts} attempts.")

    fill_pipe_with_resin(volume, pipe_radius)

    volume = defect_generator.apply(volume)

    num_voids = 50  
    add_many_small_resin_voids(volume, num_voids, pipe_radius, void_radius=1)
    return successful_filaments, filaments

def generate_3d_filament(volume, generator, min_length=512, max_length=512, filament_radius=3, pipe_radius=50, bias=0.50, preferred_direction=[1, 0, 0], orientation_jitter_deg=0.0):
    starting_point = generator.initialize_starting_point(volume.shape, filament_radius)

    if not (is_within_bounds(starting_point, volume.shape, filament_radius) and 
            is_within_pipe(starting_point, volume.shape[1] // 2, volume.shape[2] // 2, pipe_radius) and
            can_place_sphere(starting_point, volume, filament_radius, pipe_radius)):
        return None

    # Skin-core orientation: fibres starting near the pipe wall (skin) stay closely
    # aligned with preferred_direction; fibres nearer the center (core) get more
    # random tilt. Only affects 'straight' mode - kink_curve/c_curve compute their
    # own direction and ignore tilt_direction.
    if orientation_jitter_deg > 0 and hasattr(generator.point_generator, "tilt_direction"):
        center_y, center_z = volume.shape[1] // 2, volume.shape[2] // 2
        radial_dist = np.sqrt((starting_point[1] - center_y) ** 2 + (starting_point[2] - center_z) ** 2)
        radial_frac = min(radial_dist / pipe_radius, 1.0)  # 0 at center (core), 1 at wall (skin)
        local_jitter_deg = orientation_jitter_deg * (1 - radial_frac)  # core -> full jitter, skin -> ~0

        max_rad = np.radians(local_jitter_deg)
        theta = np.random.uniform(-max_rad, max_rad) if max_rad > 0 else 0.0
        phi = np.random.uniform(-max_rad, max_rad) if max_rad > 0 else 0.0
        tilt = np.array([1.0, np.tan(theta), np.tan(phi)])
        generator.point_generator.tilt_direction = tilt / np.linalg.norm(tilt)

    filament = [starting_point]

    all_directions = [
        [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [-1, -1, 0], [1, -1, 0],
        [1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1]
    ]
    all_directions = [np.array(d) for d in all_directions if not np.array_equal(d, preferred_direction)]

    num_biased = int(bias * 100)
    num_other = min(100 - num_biased, len(all_directions))

    biased_direction_choices = [np.array(preferred_direction)] * num_biased + random.sample(all_directions, k=num_other)

    step_size = 1
    direction = np.array(preferred_direction).astype(float)

    for _ in range(max_length * 10):
        next_point = generator.suggest_next_point(filament, direction, step_size, len(filament), max_length)

        if next_point is None:
            break

        if not (is_within_bounds(next_point, volume.shape, filament_radius) and 
                is_within_pipe(next_point, volume.shape[1] // 2, volume.shape[2] // 2, pipe_radius) and 
                can_place_sphere(next_point, volume, filament_radius, pipe_radius)):
            break

        if generator.point_generator.grow_from_start:
            filament.insert(0, next_point)
        else:
            filament.append(next_point)

        if len(filament) >= max_length:
            break

        generator.toggle_growth_direction()
        direction = random.choice(biased_direction_choices).astype(float)

    if len(filament) < min_length:
        return None

    return filament

def add_many_small_resin_voids(volume, num_voids, pipe_radius, void_radius=1):
    center_y, center_z = volume.shape[1] // 2, volume.shape[2] // 2
    voids_added = 0
    attempts = 0
    max_attempts = 1000000

    resin_positions = np.argwhere(volume == ATTENUATION_RESIN)

    if len(resin_positions) == 0:
        print("No resin areas")
        return voids_added

    while voids_added < num_voids and attempts < max_attempts:
        random_index = random.randint(0, len(resin_positions) - 1)
        x, y, z = resin_positions[random_index]

        for i in range(-void_radius, void_radius + 1):
            for j in range(-void_radius, void_radius + 1):
                for k in range(-void_radius, void_radius + 1):
                    if i**2 + j**2 + k**2 <= void_radius**2:
                        xi, yj, zk = x + i, y + j, z + k
                        if (0 <= xi < volume.shape[0] and
                            0 <= yj < volume.shape[1] and
                            0 <= zk < volume.shape[2] and
                            is_within_pipe((xi, yj, zk), center_y, center_z, pipe_radius)):
                            volume[xi, yj, zk] = ATTENUATION_AIR

        voids_added += 1
        attempts += 1

    if voids_added < num_voids:
        print(f"Warning: Only able to add {voids_added} small resin voids after {attempts} attempts.")
    else:
        print(f"Successfully added {voids_added} small resin voids.")

    return voids_added
