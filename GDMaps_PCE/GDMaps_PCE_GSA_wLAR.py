# GDMaps, PceModel, ErrorEstimation are taken from and built upon https://github.com/katiana22/GDM-PCE

import numpy as np
import time
import os, subprocess
import random

import matplotlib.pyplot as plt
import matplotlib as mpl

from UQpy.surrogates import *
from UQpy.sensitivity import *

from sklearn.model_selection import train_test_split

from DimensionReduction import Grassmann
from DimensionReduction import DiffusionMaps


#######################################################################################################################
#######################################################################################################################
#                                                   GDMaps PCE GSA                                                    #
#######################################################################################################################
#######################################################################################################################

def nonparsim_dcoord(n_samples, n_keep, evecs, evals):
    """
    Obtains non-parsim dcoords without rerunning GDMaps
    """
    dcoords = np.zeros([n_samples, n_keep])
    for i in range(n_keep):
        dcoords[:, i] = evals[i+1] * evecs[:, i+1]
        
    return dcoords

class GDMaps:
    """
    Performs GDMaps for a given dataset.
    n_evecs must be greater than n_keep
    """

    def __init__(self, data, n_evecs, n_keep, p, parsim=True, verbose=False):
        self.data = data
        self.n_evecs = n_evecs
        self.n_keep = n_keep
        self.p = p
        self.parsim = parsim
        self.verbose = verbose

    def get(self):
        Gr = Grassmann(distance_method=Grassmann.grassmann_distance, 
                       kernel_method=Grassmann.projection_kernel,
                       karcher_method=Grassmann.gradient_descent)
        Gr.manifold(p=self.p, samples=self.data)

        dfm = DiffusionMaps(alpha=0.5, 
                            n_evecs=self.n_evecs + 1, 
                            kernel_object=Gr, 
                            kernel_grassmann='prod')
        g, evals, evecs = dfm.mapping()
        
#         print('Grassmann projection rank is: ', Gr.p)
        
        if self.parsim:
            print('Running with parsimonious representation')

            # Residuals used to identify the most parsimonious low-dimensional representation.
            index, residuals = dfm.parsimonious(num_eigenvectors=self.n_evecs, visualization=False)
            
            coord = index[1:self.n_keep + 1]
            g_k = g[:, coord]
            
            return g_k, coord, Gr, residuals, index, evals, evecs
            
        else:
            print(f'Keeping first {self.n_keep} nontrivial eigenvectors')
            
            coord = np.arange(1, self.n_keep+1) # keeping first n_keep nontrivial eigenvectors
            g_k = g[:, coord]
            
            return g_k, coord, Gr, evals, evecs

            
    
class PceModel:
    """
    Constructs a PCE surrogate on the Grassmannian diffusion manifold.
    """

    def __init__(self, x, g, dist_obj, max_degree, regression='OLS', verbose=False):
        self.x = x
        self.g = g
        self.dist_obj = dist_obj
        self.max_degree = max_degree
        self.regression = regression
        self.verbose = verbose

    def get(self):

        # Polynomial basis
        polynomial_basis = TotalDegreeBasis(distributions=self.dist_obj, 
                                            max_degree=self.max_degree)
        
        # Regression
        if self.regression == 'OLS':
            reg = LeastSquareRegression()
        
        elif self.regression == 'Lasso':
            # function parameters need to be tuned
            reg = LassoRegression(learning_rate=0.001, iterations=1000, penalty=0.05)
            
        elif self.regression == 'Ridge':
            # function parameters need to be tuned
            reg = RidgeRegression(learning_rate=0.001, iterations=10000, penalty=0.8)
            
        else:
            raise ValueError('The only allowable input strings are `OLS`, `Lasso`, and `Ridge`.')
        
        pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=reg)

        x_train, x_test, \
        g_train, g_test = train_test_split(self.x, self.g, train_size=2 / 3, random_state=1)
        
        # Fit model
        pce.fit(x_train, g_train)
        
        print('Size of the full set of PCE basis:', pce.polynomial_basis.polynomials_number)
        print('Shape of the training set (x)):', x_train.shape)
        print('Shape of the training set (y)):', g_train.shape)
    
        error_val = ErrorEstimation(surr_object=pce).validation(x_test, g_test)

        if self.verbose:
            # Plot accuracy of PCE
            if os.path.exists('pce_accuracy'):
                command = ['rm', '-r', 'pce_accuracy']
                subprocess.run(command)

            command = ['mkdir', 'pce_accuracy']
            subprocess.run(command)

            print(g_test[0, :])
            print(pce.predict(x_test)[0, :])

            for i in range(5):
                r = random.randint(0, x_test.shape[0])
                plt.figure()
                plt.plot(g_test[r, :], 'b-o', label='true')
                plt.plot(pce.predict(x_test)[r, :], 'r-*', label='pce')
                plt.legend()
                plt.savefig('pce_accuracy/pce_{}.png'.format(i), bbox_inches='tight')
                plt.show()

        return pce, error_val

       

class ErrorEstimation:
    """
    Class for estimating the error of a PCE surrogate, based on a validation
    dataset. Used in PceModel

    **Inputs:**

    * **surr_object** ('class'):
        Object that defines the surrogate model.

    **Methods:**
    """

    def __init__(self, surr_object):
        self.surr_object = surr_object

    def validation(self, x, y):
        """
        Returns the validation error.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the samples of the validation dataset.

        * **y** (`ndarray`):
            `ndarray` containing model evaluations for the validation dataset.

        **Outputs:**

        * **eps_val** (`float`)
            Validation error.

        """
        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)

        y_val = self.surr_object.predict(x)

        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        eps_val = (n_samples - 1) / n_samples * (
                (np.sum((y - y_val) ** 2, axis=0)) / (np.sum((y - mu_yval) ** 2, axis=0)))

        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)
    

def run_GDMaps(p, 
               data, 
               num_runs=10, 
               n_keep=3):
    
    """
    Always parsimonious
    """
    
    evals_diff_runs  = []
    evecs_diff_runs  = []
    coord_diff_runs = []
    g_diff_runs = []
    residuals_diff_runs = []
    index_diff_runs = []
    
    for i in range(num_runs):
        print('Run: ', i)
        data_all = data[i]

        # Perform GDMAps
        start_time = time.time()
        
        g, coord, Grass, residuals, index, evals, evecs = GDMaps(data=data_all, 
                                                                 n_evecs=20,
                                                                 n_keep=n_keep,
                                                                 parsim=True,
                                                                 p=p).get()

        evals_diff_runs.append(evals)
        evecs_diff_runs.append(evecs)
        coord_diff_runs.append(coord)
        g_diff_runs.append(g)
        residuals_diff_runs.append(residuals)
        index_diff_runs.append(index)
        
        print("--- GDMaps - %s seconds ---" % (time.time() - start_time))
    
    return (evals_diff_runs, evecs_diff_runs, coord_diff_runs, g_diff_runs,
            residuals_diff_runs, index_diff_runs)


#######################################################################################################################
#######################################################################################################################
#                                                   PCE LAR MODEL SELECTION                                                    #
#######################################################################################################################
#######################################################################################################################

def PCE_LAR_model_selection(x, g, n_params, dist_obj, max_degree):
  
    n_dcoords = g.shape[1]
    
    # construct total-degree polynomial basis
    polynomial_basis = TotalDegreeBasis(dist_obj, max_degree)

    least_squares = LeastSquareRegression()
    pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=least_squares)

    xx_train, xx_test, \
        gg_train, gg_test = train_test_split(x, g, train_size=2 / 3, random_state=1)

    error_pce = np.zeros((n_dcoords))
    LOO_pce   = np.zeros((n_dcoords))
    pce_fo_si = np.zeros((n_params, n_dcoords))
    pce_to_si = np.zeros((n_params, n_dcoords))

    error_pceLAR = np.zeros((n_dcoords))
    MAE_pceLAR   = np.zeros((n_dcoords))
    LOO_pceLAR   = np.zeros((n_dcoords))
    pceLAR_fo_si = np.zeros((n_params, n_dcoords))
    pceLAR_to_si = np.zeros((n_params, n_dcoords))

    for i in range(3):
#         print('Theta',i)
        pce.fit(xx_train, gg_train[:, i])

        error_pce[i] = ErrorEstimation(surr_object=pce).validation(xx_test, gg_test[:, i])
        LOO_pce[i] = pce.leaveoneout_error()

        # sensitivity from PCE
        pce_sensitivity = PceSensitivity(pce)
        pce_sensitivity.run()

        sobol_first = pce_sensitivity.first_order_indices
        sobol_total = pce_sensitivity.total_order_indices

        pce_fo_si[:, i] = sobol_first.T
        pce_to_si[:, i] = sobol_total.T

        # Lars model selection
        print('Size of the full set of PCE basis:', polynomial_basis.polynomials_number)

        target_error = 1
        CheckOverfitting = True
        pceLAR = polynomial_chaos.regressions.LeastAngleRegression.model_selection(pce, 
                                                                                   target_error, 
                                                                                   CheckOverfitting)

        print('Size of the LAR PCE basis:', pceLAR.polynomial_basis.polynomials_number)

        n_samples_val = len(xx_test)
        gg_test_pce = pceLAR.predict(xx_test).flatten()
        errors = np.abs(gg_test[:, i].flatten() - gg_test_pce)
        MAE = (np.linalg.norm(errors, 1) / n_samples_val)

        error_pceLAR[i] = ErrorEstimation(surr_object=pceLAR).validation(xx_test, gg_test[:, i])
        MAE_pceLAR[i]   = MAE               
        LOO_pceLAR[i]   = pceLAR.leaveoneout_error()

        print('Mean absolute error:', MAE)
        print('Leave-one-out cross validation on ED:', pceLAR.leaveoneout_error())

        pceLAR_sensitivity = PceSensitivity(pceLAR)
        pceLAR_sensitivity.run()

        LAR_sobol_first = pceLAR_sensitivity.first_order_indices
        LAR_sobol_total = pceLAR_sensitivity.total_order_indices

        pceLAR_fo_si[:, i] = LAR_sobol_first.T
        pceLAR_to_si[:, i] = LAR_sobol_total.T
        
    return (error_pce, LOO_pce, pce_fo_si, pce_to_si,
            error_pceLAR, LOO_pceLAR, MAE_pceLAR, pceLAR_fo_si, pceLAR_to_si)



def three_regressions_PCE(x, dist_obj, s_max, d_coords, n_runs, n_dcoords):
    
    n_params = x.shape[1]

    ols_fo_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    ols_to_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    ols_error = np.zeros((n_runs, len(s_max), n_dcoords))
    ols_LOO   = np.zeros((n_runs, len(s_max), n_dcoords))

    ridge_fo_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    ridge_to_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    ridge_error = np.zeros((n_runs, len(s_max), n_dcoords))
    ridge_LOO   = np.zeros((n_runs, len(s_max), n_dcoords))

    LAR_fo_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    LAR_to_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    LAR_error = np.zeros((n_runs, len(s_max), n_dcoords))
    LAR_LOO   = np.zeros((n_runs, len(s_max), n_dcoords))
    LAR_MAE   = np.zeros((n_runs, len(s_max), n_dcoords))

    for run in range(10):
        print("Run:", run)
        g = d_coords[run]

        for i, s in enumerate(s_max):
            print("Max polynomial degree:", s)

            start_time = time.time()
            LAR_result = PCE_LAR_model_selection(x=x, 
                                                 g=g, 
                                                 n_params=n_params, 
                                                 dist_obj=dist_obj, 
                                                 max_degree=s)

            ols_fo_si[run, i] = LAR_result[2]
            ols_to_si[run, i] = LAR_result[3]
            ols_error[run, i] = LAR_result[0]
            ols_LOO[run, i]   = LAR_result[1]

            LAR_fo_si[run, i] = LAR_result[7]
            LAR_to_si[run, i] = LAR_result[8]
            LAR_error[run, i] = LAR_result[4]
            LAR_LOO[run, i]   = LAR_result[5]
            LAR_MAE[run, i]   = LAR_result[6]

            print("--- PCE-OLS and PCE-LAR - %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            Ridge_pce, Ridge_error = PceModel(x=x, 
                                            g=g, 
                                            dist_obj=joint, 
                                            max_degree=s, 
                                            regression='Ridge',
                                            verbose=False).get()

            ridge_fo_si[run, i] = PceSensitivity(Ridge_pce).calculate_first_order_indices()
            ridge_to_si[run, i] = PceSensitivity(Ridge_pce).calculate_total_order_indices()
            ridge_error[run, i] = Ridge_error
            ridge_LOO[run, i]   = Ridge_pce.leaveoneout_error()
            print("--- PCE-Ridge - %s seconds ---" % (time.time() - start_time))  
            
    return (ols_fo_si, ols_to_si, ols_error, ols_LOO,
            ridge_fo_si, ridge_to_si, ridge_error, ridge_LOO,
            LAR_fo_si, LAR_to_si, LAR_error, LAR_LOO, LAR_MAE)


def two_regressions_PCE(x, dist_obj, s_max, d_coords, n_runs, n_dcoords):
    
    n_params = x.shape[1]

    ols_fo_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    ols_to_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    ols_error = np.zeros((n_runs, len(s_max), n_dcoords))
    ols_LOO   = np.zeros((n_runs, len(s_max), n_dcoords))

    LAR_fo_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    LAR_to_si = np.zeros((n_runs, len(s_max), n_params, n_dcoords))
    LAR_error = np.zeros((n_runs, len(s_max), n_dcoords))
    LAR_LOO   = np.zeros((n_runs, len(s_max), n_dcoords))
    LAR_MAE   = np.zeros((n_runs, len(s_max), n_dcoords))

    for run in range(10):
        print("Run:", run)
        g = d_coords[run]

        for i, s in enumerate(s_max):
            print("Max polynomial degree:", s)

            start_time = time.time()
            LAR_result = PCE_LAR_model_selection(x=x, 
                                                 g=g, 
                                                 n_params=n_params, 
                                                 dist_obj=dist_obj, 
                                                 max_degree=s)

            ols_fo_si[run, i] = LAR_result[2]
            ols_to_si[run, i] = LAR_result[3]
            ols_error[run, i] = LAR_result[0]
            ols_LOO[run, i]   = LAR_result[1]

            LAR_fo_si[run, i] = LAR_result[7]
            LAR_to_si[run, i] = LAR_result[8]
            LAR_error[run, i] = LAR_result[4]
            LAR_LOO[run, i]   = LAR_result[5]
            LAR_MAE[run, i]   = LAR_result[6]

            print("--- PCE-OLD and PCE-LAR - %s seconds ---" % (time.time() - start_time))

    return (ols_fo_si, ols_to_si, ols_error, ols_LOO,
            LAR_fo_si, LAR_to_si, LAR_error, LAR_LOO, LAR_MAE)