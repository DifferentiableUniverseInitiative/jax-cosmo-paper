import hmc
import desy1
import numpy as np
import sys
import cobaya.run

GELMAN_RUBIN_TARGET = 0.03

try:
  # FIXME(https://github.com/abseil/abseil-py/issues/99)
  # FIXME(https://github.com/abseil/abseil-py/issues/102)
  # Unfortunately, many libraries that include absl (including Tensorflow)
  # will get bitten by double-logging due to absl's incorrect use of
  # the python logging library:
  #   2019-07-19 23:47:38,829 my_logger   779 : test
  #   I0719 23:47:38.829330 139904865122112 foo.py:63] test
  #   2019-07-19 23:47:38,829 my_logger   779 : test
  #   I0719 23:47:38.829469 139904865122112 foo.py:63] test
  # The code below fixes this double-logging.  FMI see:
  #   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
  
  import logging
  
  import absl.logging
  logging.root.removeHandler(absl.logging._absl_handler)
  absl.logging._warn_preinit_stderr = False
except Exception as e:
  print("Failed to fix absl logging bug", e)
  pass




limits = [
    (0.5, 0.9), # sigma8
    (0.1, 0.5), # Omega_c
    (0.03, 0.06), # Omega_b
    (0.5,  0.9), # h
    (0.9,  1.05), # n_s
    (-2.0,  -0.5), # w0
    (-0.06, 0.06), #m1
    (-0.06, 0.06), #m2
    (-0.06, 0.06), #m3
    (-0.06, 0.06), #m4
    (-0.1, 0.1), #dz1
    (-0.1, 0.1), #dz2
    (-0.1, 0.1), #dz3
    (-0.1, 0.1), #dz4
    (0.0, 3.0),  #A
    (-3., 3.), #eta
    (0.8, 3.0), #bias1
    (0.8, 3.0), #bias2
    (0.8, 3.0), #bias3
    (0.8, 3.0), #bias4
    (0.8, 3.0), #bias5
]

def gelman_rubin_r_minus_1(chain, comm):
    """
    From cobaya
    """
    cov = np.cov(chain.T)
    mean = np.mean(chain.T, axis=1)
    length  = len(chain)

    covs = comm.gather(cov)
    means = comm.gather(mean)
    lengths = comm.gather(length)

    if comm.rank == 0:
        covs = np.array(covs)
        means = np.array(means)
        mean_of_covs = np.average(covs, weights=lengths, axis=0)
        cov_of_means = np.atleast_2d(np.cov(means.T))
        diagSinvsqrt = np.diag(np.power(np.diag(cov_of_means), -0.5))
        corr_of_means = diagSinvsqrt.dot(cov_of_means).dot(diagSinvsqrt)
        norm_mean_of_covs = diagSinvsqrt.dot(mean_of_covs).dot(diagSinvsqrt)
        try:
            L = np.linalg.cholesky(norm_mean_of_covs)
            Linv = np.linalg.inv(L)
            eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
            Rminus1 = np.max(np.abs(eigvals))
        except np.linalg.LinAlgError:
            Rminus1 = np.nan
        print(f"After {length} samples R - 1 = {Rminus1:.4f}")
    else:
        Rminus1 = 0.0

    Rminus1 = comm.bcast(Rminus1)

    return Rminus1

def run_hmc(n_it, filebase, epsilon, steps_per_iteration):
    from mpi4py.MPI import COMM_WORLD
    rank  = COMM_WORLD.rank
    filename = f'{filebase}.{rank}.txt'
    y1 = desy1.MockY1Likelihood()
    np.random.seed(100 + rank)

     # mass matrix
    C = y1.cov_estimate()

    sampler = hmc.HMC(y1.posterior_and_gradient, C, epsilon, steps_per_iteration, limits)

    # first sample starts at fid
    sampler.sample(n_it, y1.fid_params)

    # continue
    while True:
        # Save chain
        chain = np.array(sampler.trace)
        np.savetxt(filename, chain)

        # check convergence
        Rminus1 = gelman_rubin_r_minus_1(chain, COMM_WORLD)
        if Rminus1 < GELMAN_RUBIN_TARGET:
            print("DONE")
            break

        # next round of samples
        sampler.sample(n_it)



mh_calls = 0
def run_mh(filename):
    # Define the lists of input params
    input_params = [
            "sigma8",
            "Omega_c",
            "Omega_b",
            "h",
            "n_s",
            "w0",
            "m1",
            "m2",
            "m3",
            "m4",
            "dz1",
            "dz2",
            "dz3",
            "dz4",
            "A",
            "eta",
            "bias1",
            "bias2",
            "bias3",
            "bias4",
            "bias5",
    ]

    y1 = desy1.MockY1Likelihood()
    def like(**kwargs):
        p = [kwargs[name] for name in input_params]
        global mh_calls
        mh_calls += 1
        post = y1.posterior(p)
        return post, {}

    covmat = np.loadtxt("./cov_estimate_from_fisher.txt")

    params = {}
    for i, name in enumerate(input_params):
        pmin, pmax = limits[i]

        params[name] = {
            'prior': {'min': pmin, 'max': pmax},
            'ref': float(y1.fid_params[i]),
            # 'proposal': (pmax - pmin) / 300.,
        }

    likelihood = {
            "desy1": {
                "external": like,
                "input_params": input_params,
                "output_params": []
            }
    }

    sampler = {
        'mcmc': {
            'learn_proposal': True,
            'covmat': covmat,
            'covmat_params': input_params,
            'Rminus1_stop': GELMAN_RUBIN_TARGET,
        }
    }

    info = {
        'sampler': sampler,
        'likelihood': likelihood,
        'params': params,
        'output': 'cob',
    }



    updated_info, sampler = cobaya.run.run(info)


if __name__ == '__main__':
    if sys.argv[1]  == 'hmc':
        run_hmc(3, "hmc_002_25", 0.02, 25)
    else:
        run_mh('cob_fisher_proposal.txt')
