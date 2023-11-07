import time
import yaml
import numpy as np
import argparse
from abacusnbody.hod.abacus_hod import AbacusHOD
from pyrecon import utils
from cosmoprimo.utils import DistanceToRedshift
# from densitysplit.pipeline import DensitySplit
from pathlib import Path
# from pypower import setup_logging
# from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.cosmology import Cosmology
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def get_rsd_positions(hod_dict):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    data = hod_dict['LRG']
    x = data['x']
    y = data['y']
    z_rsd = data['z']
    return x, y, z_rsd


def density_split(data_positions, boxsize, cellsize=5.0, seed=42,
    smooth_radius=20, nquantiles=5, filter_shape='tophat'):
    """Split random points according to their local density
    density."""
    ds = DensitySplit(data_positions, boxsize)
    np.random.seed(seed=seed)
    sampling_positions = np.random.uniform(0,
        boxsize, (5 * len(data_positions), 3))
    density = ds.get_density(smooth_radius=smooth_radius,
        cellsize=cellsize, sampling_positions=sampling_positions,
        filter_shape=filter_shape)
    quantiles, mean_density = ds.get_quantiles(
        nquantiles=nquantiles, return_density=True)
    return quantiles, mean_density, density

def get_hod(p, param_mapping, param_tracer, data_params, Ball, nthread):
    # read the parameters 
    print(p)
    print(param_mapping)
    print(param_tracer)
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        if key == 'sigma' and tracer_type == 'LRG':
            Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
        # Ball.tracers[tracer_type][key] = p[mapping_idx]
    Ball.tracers['LRG']['ic'] = 1 # a lot of this is a placeholder for something more suited for multi-tracer
    ngal_dict = Ball.compute_ngal(Nthread = nthread)[0]
    N_lrg = ngal_dict['LRG']
    Ball.tracers['LRG']['ic'] = min(1, data_params['tracer_density_mean']['LRG']*Ball.params['Lbox']**3/N_lrg)
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = nthread)
    return mock_dict

def setup_hod(config):
    print(f"Processing {config['sim_params']['sim_name']}")
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    fit_params = config['fit_params']    
    # create a new abacushod object and load the subsamples
    allzs = [0.450, 0.500, 0.575]
    Balls = []
    for ez in allzs:
        sim_params['z_mock'] = ez
        Balls += [AbacusHOD(sim_params, HOD_params)]
    # parameters to fit
    param_mapping = {}
    param_tracer = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
    return Balls, param_mapping, param_tracer, data_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    config_dir = './'
    config_fn = Path(config_dir, 'ds_abacus_lightcone.yaml')
    config = yaml.safe_load(open(config_fn))

    # setup_logging(level='WARNING')
    dataset = 'wideprior_AB'
    boxsize = 2000
    cellsize = 5.0
    redshift = 0.5
    split = 'z'
    filter_shape = 'Gaussian'
    smooth_ds = 10
    redges = np.hstack(
        [np.arange(0, 5, 1),
        np.arange(7, 30, 3),
        np.arange(31, 155, 5)]
    )
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)
    nquantiles = 5

    # Patchy cosmology as our fiducial
    fid_cosmo = Cosmology(
        Omega_m=0.307115,
        Omega_b=0.048,
        sigma8=0.8288,
        h=0.677,
        engine='class'
    )

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        # cosmology of the mock as the truth
        mock_cosmo = AbacusSummit(cosmo)
        az = 1 / (1 + redshift)
        hubble = 100 * mock_cosmo.efunc(redshift)

        hods_dir = f'hod_parameters/'
        hods_fn = Path(hods_dir, f'hod_parameters_c000.csv')
        hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=',')

        for phase in range(start_phase, start_phase + n_phase):
            sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
            config['sim_params']['sim_name'] = sim_fn
            Balls, param_mapping, param_tracer, data_params = setup_hod(config)

            for hod in range(start_hod, start_hod + n_hod):
                start_time = time.time()

                for i, newBall in enumerate(Balls):
                    hod_dict = get_hod(hod_params[hod], param_mapping, param_tracer,
                                data_params, newBall, 256)
                    x, y, z_rsd = get_rsd_positions(hod_dict)

                    dist, ra, dec = utils.cartesian_to_sky(np.c_[x, y, z_rsd])
                    d2z = DistanceToRedshift(mock_cosmo.comoving_radial_distance)
                    z = d2z(dist)

                    cout = np.c_[ra, dec, z]
                    np.save(f'data{i}.npy', cout)

                    fig, ax = plt.subplots()
                    ax.scatter(ra, z, s=0.1, marker='.')
                    plt.savefig(f'slice{i}.png')

                # # if output files exist, skip to next iteration
                # cross_fn = Path(
                #     f'/global/homes/e/epaillas/carolscratch/ds_boss/ds_cross_xi_smu/HOD/{dataset}/',
                #     f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/',
                #     f'ds_cross_xi_smu_{split}split_{filter_shape.lower()}_Rs{smooth_ds}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                # )
                # auto_fn = Path(
                #     f'/global/homes/e/epaillas/carolscratch/ds_boss/ds_auto_xi_smu/HOD/{dataset}/',
                #     f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/',
                #     f'ds_auto_xi_smu_{split}split_{filter_shape.lower()}_Rs{smooth_ds}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                # )
                # # if os.path.exists(cross_fn) and os.path.exists(auto_fn):
                # #     print(f'c{cosmo:03} ph{phase:03} hod{hod} already exists')
                # #     continue

                # cross_los = []
                # auto_los = []
                # mean_density_los = []
                # for los in ['x', 'y', 'z']:
                #     if split == 'z':
                #         xpos = x_rsd if los == 'x' else x
                #         ypos = y_rsd if los == 'y' else y
                #         zpos = z_rsd if los == 'z' else z
                #     else:
                #         xpos, ypos, zpos = x, y, z

                #     data_positions = np.c_[xpos, ypos, zpos]

                #     data_positions_ap = get_distorted_positions(data_positions, q_perp, q_para)
                #     boxsize_ap = np.array([boxsize/q_perp, boxsize/q_perp, boxsize/q_para])

                #     quantiles, mean_density, density = density_split(
                #         data_positions=data_positions_ap, boxsize=boxsize_ap,
                #         cellsize=cellsize, seed=phase, filter_shape=filter_shape,
                #         smooth_radius=smooth_ds, nquantiles=5)

                #     np.save('density_pdf_cubic.npy', density)

                #     # for i in range(5):
                #     #     quantiles[i] = revert_ap(quantiles[i], q_perp, q_para)

                #     # # QUINTILE-GALAXY CROSS-CORRELATION
                #     # cross_ds = []
                #     # for i in range(5):
                #     #     print(f'cosmo {cosmo}, hod {hod}, los {los}, ds{i}')
                #     #     result = TwoPointCorrelationFunction(
                #     #         'smu', edges=edges, data_positions1=quantiles[i],
                #     #         data_positions2=data_positions, los=los,
                #     #         engine='corrfunc', boxsize=boxsize, nthreads=256,
                #     #         compute_sepsavg=False, position_type='pos'
                #     #     )

                #     #     s, multipoles = get_distorted_multipoles(result, q_perp, q_para, ells=(0, 2, 4))
                #     #     cross_ds.append(multipoles)
                #     # cross_los.append(cross_ds)

                #     # # QUINTILE AUTOCORRELATION
                #     # auto_ds = []
                #     # for i in range(5):
                #     #     result = TwoPointCorrelationFunction(
                #     #         'smu', edges=edges, data_positions1=quantiles[i],
                #     #         los=los, engine='corrfunc', boxsize=boxsize, nthreads=256,
                #     #         compute_sepsavg=False, position_type='pos'
                #     #     )
                #     #     s, multipoles = get_distorted_multipoles(result, q_perp, q_para, ells=(0, 2, 4))
                #     #     auto_ds.append(multipoles)
                #     # auto_los.append(auto_ds)

                #     # mean_density_los.append(mean_density)

                # # cross_los = np.asarray(cross_los)
                # # auto_los = np.asarray(auto_los)

                # # cout = {
                #     # 's': s,
                #     # 'multipoles': cross_los
                # # }
                # # output_dir = Path(f'/global/homes/e/epaillas/carolscratch/ds_boss/ds_cross_xi_smu/HOD/{dataset}/',
                #     # f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                # # Path(output_dir).mkdir(parents=True, exist_ok=True)
                # # output_fn = Path(
                #     # output_dir,
                #     # f'ds_cross_xi_smu_{split}split_{filter_shape.lower()}_Rs{smooth_ds}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                # # )
                # # np.save(output_fn, cout)

                # # cout = {
                #     # 's': s,
                #     # 'multipoles': auto_los
                # # }
                # # output_dir = Path(f'/global/homes/e/epaillas/carolscratch/ds_boss/ds_auto_xi_smu/HOD/{dataset}/',
                #     # f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                # # Path(output_dir).mkdir(parents=True, exist_ok=True)
                # # output_fn = Path(
                #     # output_dir,
                #     # f'ds_auto_xi_smu_{split}split_{filter_shape.lower()}_Rs{smooth_ds}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                # # )
                # # np.save(output_fn, cout)

                # # # MEAN DENSITY PER QUINTILE
                # # output_dir = Path(f'/global/homes/e/epaillas/carolscratch/ds_boss/ds_mean_density/HOD/{dataset}/',
                #     # f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                # # Path(output_dir).mkdir(parents=True, exist_ok=True)
                # # output_fn = Path(
                #     # output_dir,
                #     # f'ds_mean_density_{split}split_{filter_shape.lower()}_Rs{smooth_ds}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                # # )
                # # np.save(output_fn, mean_density)


                # # end_time = time.time() - start_time
                # # print(f'c{cosmo:03} ph{phase:03} hod{hod} {end_time:.3f} sec')

