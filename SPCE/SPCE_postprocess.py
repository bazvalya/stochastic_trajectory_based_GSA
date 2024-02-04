from UQpy.distributions import Normal, Uniform, JointIndependent
from SPCE_optuna import *

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss


def sample_from_cond_pdf(x, n_samples, c, sigma, D, A, dist_obj):
    """
    Used to obtain samples from conditional distibution from SPCE model.
    """
    
    if D=="Normal":
        Z = Normal(loc=0, scale=1)
    elif D=="Uniform":
        Z = Uniform(loc=-1, scale=2)
        
    z = Z.rvs(n_samples)
    
    marg_x  = dist_obj.marginals
    marg_xz = marg_x.copy()
    marg_xz.append(Z)  
    joint_xz = JointIndependent(marginals=marg_xz)
    
    A_basis = CustomBasis(joint_xz, A)
    
    Xz = np.concatenate((np.repeat(np.asarray([x]), len(z), axis=0), z), axis=1)
    
    Psi = A_basis.evaluate_basis(Xz)
    
    pce_approx_ = np.dot(Psi, c)
    
    eps = np.random.normal(0, sigma, n_samples)
    
    return pce_approx_ + eps



def explicit_pdf(y_pts, x, c, sigma, D, A, dist_obj, NQ=40):
    """
    Used to obtain explicit form of the conditional response distribution.
    """

    if D=="Normal":
        Z = Normal(loc=0, scale=1)
        zs, ws_ = hermegauss(NQ)
        ws = ws_ / (2 * np.pi * sigma)
    elif D=="Uniform":
        Z = Uniform(loc=-1, scale=2)
        zs, ws_ = leggauss(NQ) #points and weights
        ws = ws_ / (2 * np.pi * sigma)
        
    marg_x  = dist_obj.marginals
    marg_xz = marg_x.copy()
    marg_xz.append(Z)  
    joint_xz = JointIndependent(marginals=marg_xz)
    
    y_pdf = []
    for y in y_pts:
        Xz = np.concatenate((np.repeat(np.asarray([x]), len(zs), axis=0), 
                     zs.reshape(-1, 1)), axis=1)

        y_expanded = np.repeat(y, len(zs), axis=0)

        A_basis = CustomBasis(joint_xz, A)
        Psi = A_basis.evaluate_basis(Xz)

        pce_approx = np.dot(Psi, c)

        exponent = -((y_expanded - pce_approx) ** 2) / (2 * sigma**2)
        integrand = np.exp(exponent) * ws
        integrand_sums = np.sum(integrand)
        
        y_pdf.append(integrand_sums)

    return y_pdf


def SPCE_first_order_indices(pce_coefficients,
                             dist_object_x,
                             multi_index_set,
                             sigma,
                             with_Z = False): 
    """
    SPCE estimates for the first order Sobol indices.
    """
    outputs_number = np.shape(pce_coefficients)[1]

    variance = np.sum(pce_coefficients[1:] ** 2, axis=0)
    
    if with_Z:
        inputs_number = len(dist_object_x.marginals) + 1
    else:
        inputs_number = len(dist_object_x.marginals)
        
    multi_index_set = multi_index_set

    first_order_indices = np.zeros([inputs_number, outputs_number])
    # take all multi-indices except 0-index
    idx_no_0 = np.delete(multi_index_set, 0, axis=0)
    for nn in range(inputs_number):
        # remove nn-th column
        idx_no_0_nn = np.delete(idx_no_0, nn, axis=1)
        # we want the rows with all indices (except nn) equal to zero
        sum_idx_rows = np.sum(idx_no_0_nn, axis=1)
        zero_rows = np.asarray(np.where(sum_idx_rows == 0)).flatten() + 1
        variance_contribution = np.sum(pce_coefficients[zero_rows, :] ** 2, axis=0)
        first_order_indices[nn, :] = variance_contribution / (variance + sigma**2)
    first_order_indices = first_order_indices
    return first_order_indices


def SPCE_total_order_indices(pce_coefficients,
                             dist_object_x,
                             multi_index_set,
                             sigma,
                             with_Z = False):
    """
    SPCE estimates for the total order Sobol indices.
    """

    outputs_number = np.shape(pce_coefficients)[1]

    variance = np.sum(pce_coefficients[1:] ** 2, axis=0)
    if with_Z:
        inputs_number = len(dist_object_x.marginals) + 1
    else:
        inputs_number = len(dist_object_x.marginals)
        
    multi_index_set = multi_index_set

    total_order_indices = np.zeros([inputs_number, outputs_number])
    for nn in range(inputs_number):
        # we want all multi-indices where the nn-th index is NOT zero
        idx_column_nn = np.array(multi_index_set)[:, nn]
        nn_rows = np.asarray(np.where(idx_column_nn != 0)).flatten()
        variance_contribution = np.sum(pce_coefficients[nn_rows, :] ** 2, axis=0)
        total_order_indices[nn, :] = variance_contribution / (variance + sigma**2)
    total_order_indices = total_order_indices
    return total_order_indices


