import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

import time
import os

from typing import Union
from UQpy.distributions.baseclass import Distribution
from UQpy.distributions.collection import JointIndependent, JointCopula
from UQpy.distributions import Normal, Uniform, JointIndependent
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import PolynomialBasis
from UQpy.surrogates import *

from scipy.optimize import minimize
from sklearn.model_selection import KFold

import optuna
import pickle



def _heading(text, length):
    """
    "Pretty" printing headings flanked by dashes.
    """
    dash_length = max(length - len(text), 0)
    ldash_length = dash_length // 2
    rdash_length = dash_length - ldash_length
    ldash = "-" * ldash_length
    rdash = "-" * rdash_length

    return ldash + text + rdash



class CustomBasis(PolynomialBasis):
    """
    A custom polynomial basis set constructed as a hyperbolic set from total-degree polynomial basis set.

    This class extends the PolynomialBasis class to create a custom basis set representing a hyperbolic set.
    It constructs the basis set based on given a list of univariate distributions and a multi_index_set.
    """
    
    def __init__(self, distributions: Union[Distribution, list[Distribution]], multi_index_set):

        inputs_number = 1 if not isinstance(distributions, (JointIndependent, JointCopula)) \
            else len(distributions.marginals)
        polynomials = PolynomialBasis.construct_arbitrary_basis(inputs_number, distributions, multi_index_set)
        super().__init__(inputs_number, len(multi_index_set), multi_index_set, polynomials, distributions)


class StochasticPCE:
    def __init__(self, X, y, dist_obj, D, p_values, q_values, scale_e_LOO, sim_name):
        self.X = X
        self.y = y
        self.dist_obj = dist_obj
        self.D = D
        self.p_values = p_values
        self.q_values = q_values
        self.sim_name = sim_name
        self.scale_e_LOO = scale_e_LOO

        self.Ns = 15 # the number of points for the warm-start strategy to estimate c
        self.NQ = 50 # the number of Gaussian quadrature integration points 

        self.A = None
        self.A_m = None
        self.c_m = None
        self.eps_loo = None
        self.pdf_Z = None
        self.joint_xz = None
        

    def OLS_mean_function(self, A_m, X, y):
        """
        Fits an OLS regression model to estimate the PCE coefficients 
        for the mean function using a custom polynomial chaos basis.

        It first constructs a custom basis set using the multi-index set 'A_m', 
        and then fits an OLS regression model using the provided input
        data 'X' and corresponding output data 'y'.
        """
        
        A_m_basis = CustomBasis(self.joint_xz, A_m)
        
        least_squares = LeastSquareRegression()
        pce = PolynomialChaosExpansion(polynomial_basis=A_m_basis, regression_method=least_squares)
        pce.fit(X, y)
        
        c_m = pce.coefficients
        eps_loo = pce.leaveoneout_error()
        
        return pce, c_m, eps_loo

    @staticmethod
    def hybridLAR(pce):
        """
        Performs a hybrid PCE by using the Least Angle Regression (LAR) algorithm for model selection.
        The approach incrementally selects a subset of the full PCE basis while maintaining a target error level.
        """
        # check the size of the basis
        # print('Size of the full set of PCE basis:', pce.polynomials_number)
        target_error = 1
        CheckOverfitting = True
        pceLAR = polynomial_chaos.regressions.LeastAngleRegression.model_selection(pce, 
                                                                                   target_error, 
                                                                                   CheckOverfitting)
        # print('Size of the LAR PCE basis:', pceLAR.polynomial_basis.polynomials_number)
        A_n = pceLAR.multi_index_set
    
        return A_n

    def estimate_c(self, X, y, sigma):
        """
        Estimates the PCE coefficients 'c_hat' with given sigma using maximum likelihood estimation (MLE) 
        and warm-start strategy from section 4.3 of the reference paper (p. 12). 

        Arguments:
            X (numpy.ndarray): Input data matrix with shape (n_samples, n_features),
                               where 'n_samples': number data samples and 'n_features': number of features.
            y (numpy.ndarray): Output data vector with shape (n_samples, 1),
                               containing 'n_samples' output values corresponding to the input data.
            sigma (float):     The standard deviation of the error term 'eps'.

        Returns:
            c_hat: Estimated PCE coefficients.
        """

        # OLS for specific data 
        _, c_m, eps_loo = self.OLS_mean_function(self.A_m, X, y)

        # print("sqrt of eps_loo in estimate_c function:", np.sqrt(eps_loo))
        # print("eps_LOO in estimate c:", eps_loo)
        # print("Shape of A:", self.A.shape)
        # Randomly initialize c^0_alpha for alpha in A\A_m
        c_0 = np.random.random((self.A.shape[0], 1))
        # To speed-up, opt for zeros for alpha in A\A_m
        # c_0 = np.zeros((self.A.shape[0], 1))
        # print("Shape c_0:", c_0.shape)
        
        # print("c_m: ", c_m)
        matching_indices = np.where(np.all(self.A[:, np.newaxis] == self.A_m, axis=2))
        c_0[matching_indices[0]] = c_m[matching_indices[1]]
        # print("c_0: ", c_0)
        c_prev = c_0

        # Generate the array of sigma values
        sigma_log = np.linspace(np.log(np.sqrt(eps_loo)), np.log(sigma), self.Ns)
        # print("eps_LOO: ", eps_loo)
        # print("Running BO with sigma:", sigma)
        sigmas = np.exp(sigma_log)
        # print("Sigmas in warm-start:", sigmas)

        # iterate over sigma values
        for i in range(self.Ns):
            # print("Sigma_i:", sigmas[i])
            # Minimize the negative log-likelihood to estimate the PCE coeffs with BFGS method
            result = minimize(self.neg_log_likelihood, c_prev, 
                              args=(X, y, sigmas[i]), 
                              method='BFGS', jac=True)
            c_i = result.x
            # print("Spipy result: ", result)
            # print("Spipy nfev: ", result.nfev)
            c_prev = c_i
            # print("c_i: ", c_i)
        c_hat = c_i
        # print("c_hat: ", c_hat)

        return c_hat

    def neg_log_likelihood(self, c, X, y, sigma):
        """
        Objective function to be optimized in the 'estimate_c' function.

        Arguments:
            c (numpy.ndarray): PCE coefficients.
            X (numpy.ndarray): Input data matrix with shape (n_samples, n_features),
                               where 'n_samples': number data samples and 'n_features': number of features.
            y (numpy.ndarray): Output data vector with shape (n_samples, 1),
                               containing 'n_samples' output values corresponding to the input data.
            sigma (float):     The standard deviation of the error term 'eps'.

        Returns:
            -likelihood_sum:      Negative sum of logarithm of the likelihoods 
                                  (the sum in Eq. 25 in the reference paper).
            -log_likelihood_grad: Negative gradient.
        """
        
        likelihoods, gradients = self.likelihood_tilde_with_grad(c, X, y, sigma, grad=True)
        likelihood_sum = np.sum(np.log(likelihoods))

        divisions = gradients / likelihoods

        log_likelihood_grad = np.sum(divisions, axis=0)
        # print("minus division_sums:", -log_likelihood_grad) 

        return -likelihood_sum, -log_likelihood_grad


    def likelihood_tilde_with_grad(self, c, X, y, sigma, grad=False):
        """
        Estimation of the likelihood of the PCE coefficients c conditioned on 
        noise standard deviation sigma and gradient of Eq. 24 (reference paper)
        using Gaussian quadrature.
        Used in 'neg_log_likelihood' to obtain obhjective function 
        for the optimization in the'estimate_c' function.
        """
    
        if self.pdf_Z == "Uniform":
            zs, ws_ = leggauss(self.NQ) #points and weights
            # scale weights 
            ws = ws_ * (1 / (np.sqrt(2 * np.pi) * sigma))
        elif self.pdf_Z == "Normal":
            zs, ws_ = hermegauss(self.NQ)
            # scale weights 
            ws = ws_ * (1 / (np.sqrt(2 * np.pi) * sigma))

        # print("coefficient c:", c)
        Xz = np.concatenate((np.repeat(X, len(zs), axis=0), 
                         np.tile(zs.reshape(-1, 1), (len(X), 1))), axis=1)

        y_expanded = np.repeat(y, len(zs), axis=0).reshape(len(y), len(zs), -1)
        ws_expanded = np.tile(ws, len(X)).reshape(len(X), len(zs), -1)

        A_basis = CustomBasis(self.joint_xz, self.A)
        # print("A_basis polynomials number:", A_basis.polynomials_number)
        Psi = A_basis.evaluate_basis(Xz)

        pce_approx_ = np.dot(Psi, c)
        pce_approx = pce_approx_.reshape(len(X), len(zs), -1)
        # print("pce_approx.shape:", pce_approx.shape)

        diff = y_expanded - pce_approx
        exponent = -(diff ** 2) / (2 * sigma**2)
        # print("exponent shape:", exponent.shape)
        integrand = np.exp(exponent) * ws_expanded
        integrand_sums = np.sum(integrand, axis=1)

        if grad:
            # rehaping  and recalculating needed arrays 
            y_expanded_rsh = np.repeat(y, len(zs), axis=0).reshape(len(y), len(zs))
            ws_expanded_rsh = np.tile(ws, len(X)).reshape(len(X), len(zs))

            pce_approx_rsh = pce_approx_.reshape(len(X), len(zs))

            diff_rsh = y_expanded_rsh - pce_approx_rsh
            exponent_rsh = -(diff_rsh ** 2) / (2 * sigma**2)
            integrand_rsh = np.exp(exponent_rsh) * ws_expanded_rsh

            len_poly_basis = A_basis.polynomials_number
            grad_list = [np.sum(integrand_rsh * diff_rsh * Psi[:, beta].reshape(len(X), len(zs))/(sigma**2), axis=1) for beta in range(len_poly_basis)]
            gradients = np.asarray(grad_list).T

            return integrand_sums, gradients
            
        else:
            return integrand_sums



    def optimize_sigma(self, trial):
        """
        Estimates the CV scores for given PCE coefficients 'c_k' for each cross-validation 
        fold using the provided training data and the current value of sigma.

        The function corresponds to the method outlined in section 4.4 of the reference paper.
        """
        # for optuna
        # sigma = trial.suggest_float("sigma", 0.1 * np.sqrt(self.eps_loo), 1 * np.sqrt(self.eps_loo)
        sigma = trial.suggest_uniform("sigma", 0.1 * np.sqrt(self.eps_loo), 1 * np.sqrt(self.eps_loo))

        if len(self.X) < 200:
            num_cv_folds = 10
        elif len(self.X) < 1000:
            num_cv_folds = 5
        else:
            num_cv_folds = 3
        # print("Number of k-folds:", num_cv_folds)

        # cross-validation folds
        kf = KFold(n_splits=num_cv_folds, shuffle=True)

        sigma_scores = []

        for train_indices, val_indices in kf.split(self.X):
            # print("Fold #: ", train_indices)
            X_train, X_val = self.X[train_indices], self.X[val_indices]
            y_train, y_val = self.y[train_indices], self.y[val_indices]
            
            start_time = time.time()
            # estimate c_k for the current sigma using training data
            c_k = self.estimate_c(X_train, y_train, sigma)
            # print("Estimate c_k for one fold:  - %s seconds -" % (time.time() - start_time))
            
            # sum of the log-likelihood for the validation set
            start_time = time.time()
            l_k = np.sum(np.log(self.likelihood_tilde_with_grad(c_k, X_val, y_val, sigma)))
            # print("CV score estimation for one fold:  - %s seconds -" % (time.time() - start_time))

            sigma_scores.append(l_k) 

        # compute the CV score as the sum for all CV-folds
        cv_score = np.sum(sigma_scores)

        return cv_score


    def maximize_with_optuna(self, objective_f, add_param_string, n_jobs, verbose=False):
        
        # TPESampler: Bayesian optimisation
        study_sigma = optuna.create_study(study_name=f"{self.sim_name}_"+ add_param_string,
                                          direction="maximize",
                                          sampler=optuna.samplers.TPESampler(n_startup_trials=30)
                                          )

        study_sigma.optimize(func=objective_f, 
                             n_trials=150,
                             n_jobs=n_jobs,
                             show_progress_bar=True)

        best_sigma = study_sigma.best_params["sigma"]
        best_l = study_sigma.best_value

        if verbose:
            return best_sigma, best_l, study_sigma

        else:
            return best_sigma, best_l


    def fit_adaptively(self):
        """
        Performs adaptive algorithm for building a stochastic PCE.
        Return a model with the maximal CV score.
        """
        marg_x = self.dist_obj.marginals
        M = len(marg_x)  # dimensionality of the problem

        # Initialization
        iz = 0
        ip = 0
        iq = 0

        Nz = len(self.D)
        Np = len(self.p_values)
        Nq = len(self.q_values)

        consecutive_qs = 0

        l_final = None
        D_final = None
        A_final = None
        c_hat_final = None
        sigma_hat_final = None
        p_final = None
        q_final = None

        init_score = -np.inf

        l_best_in_curr_D = None
        l_best_in_prev_D = None

        while iz < Nz:
            l_best_in_curr_p = None
            l_best_in_prev_p = None
            consecutive_qs = 0

            if self.D[iz] == "Uniform":
                self.pdf_Z = "Uniform"
                Z = Uniform(loc=-1, scale=2)
                marg_xz = marg_x.copy()
                marg_xz.append(Z)
                self.joint_xz = JointIndependent(marginals=marg_xz)

            elif self.D[iz] == "Normal":
                self.pdf_Z = "Normal"
                Z = Normal(loc=0, scale=1)
                marg_xz = marg_x.copy()
                marg_xz.append(Z)
                self.joint_xz = JointIndependent(marginals=marg_xz)

            else:
                print("This implementation only considers U(-1, 1) and N(0, 1) distributions")

            while ip < Np:
                while iq < Nq:
                    print(_heading(f"Running with {self.D[iz]} Distribution, p={self.p_values[ip]}, q={self.q_values[iq]}", 80))
                    A_ = PolynomialBasis.calculate_hyperbolic_set(inputs_number=M + 1,
                                                                  degree=self.p_values[ip],
                                                                  q=self.q_values[iq])

                    self.A_m = np.array([alpha for alpha in A_ if alpha[-1] == 0])
                    A_c = np.array([alpha for alpha in A_ if tuple(alpha) not in map(tuple, self.A_m)])

                    mean_f_spce, self.c_m, self.eps_loo = self.OLS_mean_function(self.A_m, self.X, self.y)

                    A_n = self.hybridLAR(mean_f_spce)
                    self.A = np.concatenate((A_n, A_c), axis=0)
                    print("A:", self.A)
                    
                    print("sqrt of eps_loo in main function before scaling:", np.sqrt(self.eps_loo))
                    self.eps_loo = self.eps_loo * self.scale_e_LOO

                    print("sqrt of eps_loo in main function after scaling:", np.sqrt(self.eps_loo))

                    adaptive_algo_param = f"D_{self.D[iz]}_p_{self.p_values[ip]}_q_{self.q_values[iq]}"

                    start_time = time.time()

                    opt_sigma, opt_cv_score, study_ = self.maximize_with_optuna(objective_f=self.optimize_sigma, 
                                                                                add_param_string=adaptive_algo_param,
                                                                                n_jobs=10, 
                                                                                verbose=True)

                    print("Bayesian optimization with Optuna took:  - %s seconds -" % (time.time() - start_time))

                    if not os.path.exists(f"optuna_verbose/{self.sim_name}"):
                        os.makedirs(f"optuna_verbose/{self.sim_name}")

                    # save study
                    with open(f"optuna_verbose/{self.sim_name}/optuna_"+adaptive_algo_param+".pickle", "wb") as f:
                        pickle.dump(study_, f)
                    # joblib.dump(study_, f"optuna_verbose/{self.sim_name}/optuna_"+adaptive_algo_param+".pkl")
                    
                    # # plot optimisation history
                    # plt.rcParams["figure.figsize"] = (10,5)
                    # optuna.visualization.matplotlib.plot_optimization_history(study_)
                    # if not os.path.exists(f"optuna_verbose/{self.sim_name}/opt_hist_"+adaptive_algo_param+".png"):
                    #     plt.savefig(f"optuna_verbose/{self.sim_name}/opt_hist_"+adaptive_algo_param+".png", 
                    #             bbox_inches="tight")
                    # plt.show()

                    # # # plot slice values
                    # optuna.visualization.matplotlib.plot_slice(study_)
                    # if not os.path.exists(f"optuna_verbose/{self.sim_name}/opt_slice_"+adaptive_algo_param+".png"):
                    #     plt.savefig(f"optuna_verbose/{self.sim_name}/opt_slice_"+adaptive_algo_param+".png", 
                    #             bbox_inches="tight")
                    # plt.show()

                    #  # # plot timeline
                    # optuna.visualization.matplotlib.plot_timeline(study_)
                    # if not os.path.exists(f"optuna_verbose/{self.sim_name}/opt_timeline_"+adaptive_algo_param+".png"):
                    #     plt.savefig(f"optuna_verbose/{self.sim_name}/opt_timeline_"+adaptive_algo_psaram+".png", 
                    #             bbox_inches="tight")
                    # plt.show()


                    c = self.estimate_c(X=self.X,
                                        y=self.y,
                                        sigma=opt_sigma)

                    print("Optimal sigma:", opt_sigma)
                    print("CV score:", opt_cv_score)

                    # save intermediate results
                    if not os.path.exists(f"optuna_verbose/{self.sim_name}/inter_res_D_{self.D[iz]}_p_{self.p_values[ip]}_q_{self.q_values[iq]}.npz"):
                        np.savez(f"optuna_verbose/{self.sim_name}/inter_res_D_{self.D[iz]}_p_{self.p_values[ip]}_q_{self.q_values[iq]}.npz", 
                        x1 = self.D[iz],  # D
                        x2 = self.p_values[ip],  # p
                        x3 = self.q_values[iq],  # q
                        x4 = self.A,  # A
                        x5 = c,  # c_hat
                        x6 = opt_sigma,  # sigma_hat
                        x7 = opt_cv_score,  # cv_score
                        )


                    if opt_cv_score > init_score:
                        init_score = opt_cv_score
                        q_opt = self.q_values[iq]
                        A_opt = self.A
                        c_hat = c
                        sigma_hat = opt_sigma
                        consecutive_qs = 0
                    else:
                        consecutive_qs += 2

                    if consecutive_qs >= 2:
                        break
                    
                    best_q_in_p = q_opt
                    best_A_in_p = A_opt
                    best_c_hat_in_p = c_hat
                    best_sigma_hat_in_p = sigma_hat
                    
                    l_best_in_curr_p = init_score
                    iq += 1

                consecutive_qs = 0
                init_score = -np.inf

                iq = 0
                ip += 1

                if l_best_in_prev_p is not None and l_best_in_prev_p >= l_best_in_curr_p:
                    break

                l_best_in_prev_p = l_best_in_curr_p

                best_p_in_D = self.p_values[ip-1]
                best_q_in_D = best_q_in_p
                best_A_in_D = best_A_in_p   
                best_c_hat_in_D = best_c_hat_in_p   
                best_sigma_hat_in_D = best_sigma_hat_in_p  

            l_best_in_curr_D = l_best_in_prev_p

            print("best_p_in_D:", best_p_in_D)
            print("best_q_in_p", best_q_in_D)
            print("l_best_in_curr_D:", l_best_in_curr_D)

            l_best_in_curr_p = None
            l_best_in_prev_p = None
            init_score = -np.inf
            consecutive_qs = 0

            iq = 0
            ip = 0
            iz += 1

            if l_best_in_prev_D is not None and l_best_in_prev_D >= l_best_in_curr_D:
                break

            l_best_in_prev_D = l_best_in_curr_D
            p_final = best_p_in_D
            q_final = best_q_in_D
            D_final = self.D[iz-1]
            A_final = best_A_in_D
            sigma_hat_final = best_sigma_hat_in_D
            c_hat_final = best_c_hat_in_D

        l_final = l_best_in_prev_D

        return (D_final, p_final, q_final, A_final, c_hat_final, sigma_hat_final, l_final)



