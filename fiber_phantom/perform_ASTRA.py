import astra
import numpy as np
import pylab
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.optimize as sco

def save_as_nifti(array, file_path): # for 3D Slicer visualization
    nifti_img = nib.Nifti1Image(array, affine=np.eye(4))  
    nib.save(nifti_img, file_path)

# def perform_tomography(volume, volume_dimensions, num_angles, geometry_type, det_width_u, det_width_v, det_count_x, det_count_y, i0, gamma, algorithm, show_plots, source_origin, origin_det):
#     vol_geom = astra.create_vol_geom(volume_dimensions)
#     angles = np.linspace(0, np.pi, num_angles, False)
#     proj_geom = astra.create_proj_geom(geometry_type, det_width_u, det_width_v, det_count_x, det_count_y, angles, source_origin, origin_det)
    
#     proj_id_original, proj_data_original = astra.create_sino3d_gpu(volume, proj_geom, vol_geom)

#     # gamma is linear attenuation coefficient of the materials and is used
#     # in the exponential attenuation model during tomography simulations
#     # to convert attenuation values into photon counts and vv. 
#     virtual_photon_count = i0 * np.exp(-gamma * proj_data_original)
#     noisy_virtual_photon_counts = np.random.poisson(virtual_photon_count)
#     noisy_virtual_photon_counts[noisy_virtual_photon_counts == 0] = 1
#     proj_data_noisy = -np.log(noisy_virtual_photon_counts/i0)/gamma

#     proj_id_noisy = astra.data3d.create('-sino', proj_geom, proj_data_noisy)

#     rec_id_original, rec_id_noisy = [astra.data3d.create('-vol', vol_geom) for _ in range(2)]
#     cfgs = {
#         'original': {'ReconstructionDataId': rec_id_original, 'ProjectionDataId': proj_id_original},
#         'noisy': {'ReconstructionDataId': rec_id_noisy, 'ProjectionDataId': proj_id_noisy}
#     }

#     results = {}
#     for key, cfg in cfgs.items():
#         cfg = astra.astra_dict(algorithm)
#         cfg.update(cfgs[key])
#         alg_id = astra.algorithm.create(cfg)
#         astra.algorithm.run(alg_id, 200)
#         rec = astra.data3d.get(cfg['ReconstructionDataId'])
#         results[key] = rec
#         # np.save(f'reconstructed_{key}.npy', rec)
#         # save_as_nifti(rec, f'reconstructed_{key}.nii')
#         astra.algorithm.delete(alg_id)
#         astra.data3d.delete(cfg['ReconstructionDataId'])
#         astra.data3d.delete(cfg['ProjectionDataId'])

#     if show_plots:
#         pylab.gray()
#         pylab.figure(figsize=(10, 8))
#         pylab.imshow(proj_data_original[:, int(proj_data_original.shape[1] / 2), :])
#         pylab.figure(figsize=(10, 8))
#         pylab.imshow(proj_data_noisy[:, int(proj_data_noisy.shape[1] / 2), :])

#     return results['original'], results['noisy']

# Estimate the absorption factor based on average absorption ratio
def estimate_absorption_factor(projections, average_absorption_ratio):
    projs = projections[projections > 0]
    initial_guess = 1.0 / projs.mean()
    print("DIAG: initial_guess =", initial_guess, " projs.mean() =", projs.mean())
    def error_function(fac):
        rat = 1 - np.exp(-projs * fac[0]).mean()
        return (rat - average_absorption_ratio) ** 2
    result = sco.fmin(error_function, initial_guess, disp=False)[0]
    print("DIAG: fitted fac[0] =", result)
    return result


def perform_tomography(volume, volume_dimensions, num_angles, geometry_type, det_width_u, det_width_v, det_count_x, det_count_y, i0, algorithm, show_plots, source_origin, origin_det, angular_range_degrees=180):
    # astra.create_vol_geom(Yin, Xin, Zin) internally requires the data array
    # to have shape (Zin, Yin, Xin) - NOT a simple 2-axis swap of the input.
    # Derived empirically from real ASTRA error messages (verified against
    # two independent test cases). To get a geometry matching our actual
    # volume.shape == (Vx, Vy, Vz), we need Zin=Vx, Yin=Vy, Xin=Vz:
    vol_geom = astra.create_vol_geom(volume_dimensions[1], volume_dimensions[2], volume_dimensions[0])
    # angular_range_degrees=180 matches the original behaviour; set to 360 in
    # parameters.json to match a real scan that covered the full rotation.
    angles = np.linspace(0, np.radians(angular_range_degrees), num_angles, False)

    if geometry_type == 'parallel3d':
        # Parallel-beam has no point source, so there is no source/detector
        # distance to specify - passing source_origin/origin_det here would
        # raise an error, so they're simply omitted.
        proj_geom = astra.create_proj_geom(geometry_type, det_width_u, det_width_v, det_count_x, det_count_y, angles)
    elif geometry_type == 'cone':
        proj_geom = astra.create_proj_geom(geometry_type, det_width_u, det_width_v, det_count_x, det_count_y, angles, source_origin, origin_det)
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}")

    proj_id_original, proj_data_original = astra.create_sino3d_gpu(volume, proj_geom, vol_geom)

    avg_absorption_ratio = 0.5
    absorption_factor = estimate_absorption_factor(proj_data_original, avg_absorption_ratio)

    virtual_photon_count = i0 * np.exp(-absorption_factor * proj_data_original)
    noisy_virtual_photon_counts = np.random.poisson(virtual_photon_count)
    noisy_virtual_photon_counts[noisy_virtual_photon_counts == 0] = 1  # Avoid log(0)
    proj_data_noisy = -np.log(noisy_virtual_photon_counts / i0) / absorption_factor

    rec_id_original, rec_id_noisy = [astra.data3d.create('-vol', vol_geom) for _ in range(2)]
    cfgs = {
        'original': {'ReconstructionDataId': rec_id_original, 'ProjectionDataId': proj_id_original},
        'noisy': {'ReconstructionDataId': rec_id_noisy, 'ProjectionDataId': astra.data3d.create('-sino', proj_geom, proj_data_noisy)}
    }


    results = {}
    for key, cfg in cfgs.items():
        cfg_alg = astra.astra_dict(algorithm)
        cfg_alg.update(cfg)
        alg_id = astra.algorithm.create(cfg_alg)
        astra.algorithm.run(alg_id, 200)
        rec = astra.data3d.get(cfg['ReconstructionDataId'])
        results[key] = rec
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(cfg['ReconstructionDataId'])
        astra.data3d.delete(cfg['ProjectionDataId'])

    astra.data3d.delete(proj_id_original)
    
    return results['original'], results['noisy']
