#%%

# [1] A. Solin and S. Särkkä, “Hilbert Space Methods for Reduced-Rank Gaussian Process Regression,” arXiv:1401.5508 [stat], Jan. 2014, Accessed: Oct. 18, 2019. [Online]. Available: http://arxiv.org/abs/1401.5508.
# [2] A. Solin, M. Kok, N. Wahlström, T. B. Schön, and S. Särkkä, “Modeling and interpolation of the ambient magnetic field by Gaussian processes,” arXiv:1509.04634 [cs, stat], Sep. 2015, Accessed: Oct. 18, 2019. [Online]. Available: http://arxiv.org/abs/1509.04634.

import tensorflow as tf
import numpy as np
import gpflow
import math

# from gpflow.inducing_variables import InducingVariables
# from gpflow.base import TensorLike
from gpflow.utilities import to_default_float

# from gpflow import covariances as cov

import itertools

from gpflow.models.model import GPModel
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction, Zero
from typing import Optional, Tuple

# from gpflow.models.model import Data, DataPoint, GPModel, MeanAndVariance
# from gpflow import kullback_leiblers as kl
# from gpflow.ci_utils import ci_niter

Data = Tuple[tf.Tensor, tf.Tensor]
DataPoint = Tuple[tf.Tensor, tf.Tensor]
MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]

# Make use of the structured covariance matrices that are computationally efficient.
# We take advantage of this using TensorFlow's LinearOperators:
# BlockDiag = tf.linalg.LinearOperatorBlockDiag
Diag = tf.linalg.LinearOperatorDiag
# LowRank = tf.linalg.LinearOperatorLowRankUpdate

# class gp_domain:
#     def __init__(self, mask, xlim, ylim, m):
#         return 1


class gp_domain:
    def __init__(self, boundary, m):

        self.boundary = boundary
        self.input_dim = boundary.shape[0]  # Input dimension
        self.m = m  # Total number of basis functions

        self.m_hat = np.round(self.m ** (1 / self.input_dim)).astype(
            int
        )  # Number of basis function in each direction see 4.2 in [1]
        self.j = np.array(
            list(itertools.product(range(1, self.m_hat + 1), repeat=self.input_dim)),
            dtype=float,
        )  # Select m permutations

    def __len__(self):
        return self.m  # len(self.ms)

    def eigenfun(self, x):
        """
        Evaluate eigenfunctions.
        """
        nx = x.shape[0]
        ev = tf.ones((nx, self.m))
        # ev = tf.ones(self.m)
        for i in range(3):
            # ev *= self.boundary[i] ** (-1 / 2) * np.sin(np.pi * (x[i] + self.boundary[i] ) / 2 / self.boundary[i]*self.j[:, i][np.newaxis,:])
            ev *= self.boundary[i] ** (-1 / 2) * np.sin(
                np.pi
                * tf.matmul(
                    (x[:, i][:, np.newaxis] + self.boundary[i]) / 2 / self.boundary[i],
                    self.j[:, i][np.newaxis, :],
                )
            )

        # Numpy kron method
        # ev2 = np.ones((nx, self.m))
        # for i in range(3):
        #     ev2 *= self.boundary[i] ** (-1 / 2) * np.sin(np.pi * np.kron(self.j[:, i],( x[:,i][:,np.newaxis] + self.boundary[i] ) / 2 / self.boundary[i] ))

        return ev

    def eigenfungrad(self, x):
        """
        Evaluate the gradient of the eigenfunctions (16, 18, 19) in [2]
        :param x:
        :return:
        """

        nx = x.shape[0]
        jacobian = np.zeros((self.input_dim * x.shape[0], self.m))

        npi = np.zeros(shape=(nx, self.m, self.input_dim))

        for i in range(self.input_dim):
            # npi[:, :, i] = tf.matmul(((x[:, i][:,np.newaxis] + self.boundary[i]) / 2 / self.boundary[i]), self.j[:, i][np.newaxis,:] * np.pi)
            npi[:, :, i] = np.dot(
                ((x[:, i][:, np.newaxis] + self.boundary[i]) / 2 / self.boundary[i]),
                self.j[:, i][np.newaxis, :] * np.pi,
            )

        sinnpi = np.sin(npi)
        cosnpi = np.cos(npi)

        # lsqrtinv = np.prod(np.array(self.L)[:,1])**(-1/2)
        lsqrtinv = np.prod(self.boundary) ** (-1 / 2)

        jacobian[0 :: self.input_dim, :] = (
            np.pi
            / (2 * self.boundary[0])
            * lsqrtinv
            * self.j[:, 0]
            * cosnpi[:, :, 0]
            * sinnpi[:, :, 1]
            * sinnpi[:, :, 2]
        )
        jacobian[1 :: self.input_dim, :] = (
            np.pi
            / (2 * self.boundary[1])
            * lsqrtinv
            * self.j[:, 1]
            * cosnpi[:, :, 1]
            * sinnpi[:, :, 0]
            * sinnpi[:, :, 2]
        )
        jacobian[2 :: self.input_dim, :] = (
            np.pi
            / (2 * self.boundary[2])
            * lsqrtinv
            * self.j[:, 2]
            * cosnpi[:, :, 2]
            * sinnpi[:, :, 0]
            * sinnpi[:, :, 1]
        )

        # Numpy implementation

        # nx = x.shape[0]
        # jacobian = np.zeros((self.input_dim * x.shape[0], self.m))

        # npi = np.zeros(shape=(nx, self.m, self.input_dim))

        # for i in range(self.input_dim):
        #     npi[:, :, i] = np.kron(self.j[:, i].T * np.pi, ((x[:, i] + self.boundary[i])
        #                                                              / 2 / self.boundary[i])[:,np.newaxis])

        # sinnpi = np.sin(npi)
        # cosnpi = np.cos(npi)

        # lsqrtinv = np.prod(self.boundary)**(-1/2)

        # jacobian[0::self.input_dim, :] = np.pi / (2 * self.boundayr[0]) * lsqrtinv * self.j[:, 0] * cosnpi[:, :, 0] * sinnpi[:, :, 1] * sinnpi[:, :, 2]
        # jacobian[1::self.input_dim, :] = np.pi / (2 * self.boundary[1]) * lsqrtinv * self.j[:, 1] * cosnpi[:, :, 1] * sinnpi[:, :, 0] * sinnpi[:, :, 2]
        # jacobian[2::self.input_dim, :] = np.pi / (2 * self.boundary[2]) * lsqrtinv * self.j[:, 2] * cosnpi[:, :, 2] * sinnpi[:, :, 0] * sinnpi[:, :, 1]
        return jacobian
        # return tf.concat(values=[tf.tile(np.eye(self.input_dim), (nx, 1)) ,tf.constant(jacobian)],axis=1)

    def eigenval(self):
        """
        Evaluate eigenvalues.

        Equation 16 from [2]
        """
        lambd2 = np.zeros(self.m)  # Lambda

        for i in range(self.input_dim):
            lambd2 += (self.j[:, i] * np.pi / 2.0 / self.boundary[i]) ** 2

        return lambd2  # return lambda squared


class DGP(GPModel):
    def __init__(
        self,
        data: Data,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        domain=[],
    ):
        """
        Make model from GPflow template
        """

        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(
                data, kernel, likelihood
            )

        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
        )

        self.domain = domain
        self.ms = domain.m  # Number of basis functions
        self.m_lin = 3  # Linear kernel
        self.data = data  # Training data
        self.n_data, self.input_dim = self.data[0].shape
        self.output_dim = self.data[1].shape[1]
        self.vecY = tf.reshape(data[1], [-1, 1])  # Vectorize outputs

        # Calculate the eigenfunctions based on domain and training data
        self.NablaPhi = tf.concat(
            values=[
                tf.tile(np.eye(self.input_dim), (self.n_data, 1)),
                tf.constant(self.domain.eigenfungrad(self.data[0])),
            ],
            axis=1,
        )
        self.NablaPhiNablaPhi = tf.matmul(
            self.NablaPhi, self.NablaPhi, transpose_a=True
        )
        self.NablaPhiY = tf.matmul(self.NablaPhi, self.vecY, transpose_a=True)

        # initiate eigenvalues based on training data, domain and (initial) hyperparams
        if kernel.kernels[0].name == "squared_exponential":
            self.spectral_kernel = self.spectral_kernel_rbf

        if kernel.kernels[0].name == "matern12":
            self.spectral_kernel = lambda: self.spectral_kernel_matern(1 / 2)
        elif kernel.kernels[0].name == "matern32":
            self.spectral_kernel = lambda: self.spectral_kernel_matern(3 / 2)
        elif kernel.kernels[0].name == "matern52":
            self.spectral_kernel = lambda: self.spectral_kernel_matern(5 / 2)

        self.calculate_eigenvalues()

        self.pi = tf.constant(math.pi, dtype=tf.float64)

        # self.trainingKuf = tf.transpose( self.domain.eigenfungrad(self.data[0]))
        # self.predictFlag = 1

        # assert self.data[1].shape[1] == 1
        # self.num_data = self.data[0].shape[0]
        # self.num_input = self.data[0].shape[1]
        # self.num_latent = self.data[1].shape[1]

        # self.Kuu = self.Kuu_rbf_dgp()

        # self.q_mu = gpflow.Parameter(np.zeros((self.ms+self.m_lin, 1)))
        # self.q_sqrt = gpflow.Parameter(np.ones(self.ms+self.m_lin), transform=gpflow.utilities.positive())

    # @tf.function(autograph=False)
    # def optimization_step(optimizer, model: gpflow.models.SVGP, batch):
    #     with tf.GradientTape(watch_accessed_variables=False) as tape:
    #         tape.watch(model.trainable_variables)
    #         # objective = - model.elbo(*batch)
    #         grads = tape.gradient(objective, model.trainable_variables)
    #     # optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     # return objective

    # def compute_loss_and_gradients(self):#loss_closure: LossClosure, variables: V) -> Tuple[tf.Tensor, V]:
    #     with tf.GradientTape(watch_accessed_variables=False) as tape:
    #         tape.watch(self.trainable_variables)
    #         loss = -self.maximum_log_likelihood_objective()#loss_closure()
    #     grads = tape.gradient(loss, self.trainable_variables)
    #     return loss, grads

    def calculate_eigenvalues(self):
        sigma2 = self.likelihood.variance  # meas noise
        sigma2_lin = self.kernel.kernels[1].variance  # linear variance
        # Equation 20 in Solin 2015
        self.Lambda = tf.concat(
            values=[
                sigma2_lin * tf.ones(self.input_dim, dtype=tf.dtypes.float64),
                self.spectral_kernel(),
            ],
            axis=0,
        )

        # # Calculate the Cholesky factor
        self.L = tf.linalg.cholesky(
            self.NablaPhiNablaPhi + tf.linalg.diag(sigma2 / self.Lambda)
        )  # O(m^3)

        self.v = tf.linalg.triangular_solve(self.L, self.NablaPhiY)

        self.foo = tf.linalg.triangular_solve(tf.transpose(self.L), self.v, lower=False)

    def maximum_log_likelihood_objective(self):  # ELBO or log marg likelihood

        ## Extract parameters
        sigma2 = self.likelihood.variance  # meas noise

        # # % Number of n=observations*3 and m=basis functions
        n = self.vecY.shape[0]
        m = self.NablaPhiY.shape[0]

        # % Evaluate terms based on eigenvalues
        self.calculate_eigenvalues()

        yiQy = (
            tf.matmul(self.vecY, self.vecY, transpose_a=True)
            - tf.matmul(self.v, self.v, transpose_a=True)
        ) / sigma2
        logdetQ = (
            (self.input_dim * n - m) * tf.math.log(sigma2)
            + tf.reduce_sum(tf.math.log(self.Lambda))
            + 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)))
        )

        # % Return approx. negative log marginal likelihood
        e = -0.5 * (
            tf.squeeze(yiQy + logdetQ) + n * self.input_dim * tf.math.log(2 * self.pi)
        )
        return e

    def predict_f(
        self, predict_at: DataPoint, full_cov=False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        n_predict = predict_at.shape[0]
        sigma2 = self.likelihood.variance  # meas noise

        NablaPhit = tf.concat(
            values=[
                tf.tile(np.eye(self.input_dim), (n_predict, 1)),
                tf.constant(self.domain.eigenfungrad(predict_at)),
            ],
            axis=1,
        )

        fvar = tf.expand_dims(
            sigma2
            * tf.reduce_sum(
                (tf.linalg.triangular_solve(self.L, tf.transpose(NablaPhit))) ** 2,
                axis=0,
            ),
            axis=1,
        )
        fmean = tf.matmul(NablaPhit, self.foo)  # eq(14)

        return np.reshape(fmean, [-1, self.output_dim]), np.reshape(
            fvar, [-1, self.output_dim]
        )

    def spectral_kernel_rbf(self):

        assert (
            self.kernel.kernels[0].name == "squared_exponential"
        )  # Halt if incorrect kernel

        domain = self.domain
        input_dim = self.input_dim
        eigenValues = tf.constant(domain.eigenval(), dtype=gpflow.default_float())

        S = (
            self.kernel.kernels[0].variance
            * to_default_float(tf.pow(2.0 * np.pi, input_dim / 2.0))
            * tf.pow(self.kernel.kernels[0].lengthscales, input_dim)
            * tf.exp(
                -eigenValues * tf.square(self.kernel.kernels[0].lengthscales) / 2.0
            )
        )

        return S

    def spectral_kernel_matern(self, nu):
        # TODO
        # assert self.kernel.kernels[0].name == '' # Halt if incorrect kernel

        domain = self.domain
        input_dim = self.input_dim
        eigenValues = tf.constant(domain.eigenval(), dtype=gpflow.default_float())

        # eq10 in know your boundaries (Kok, Solin)
        S1 = self.kernel.kernels[0].variance * to_default_float(
            tf.exp(tf.math.lgamma(nu + input_dim / 2.0)) / tf.exp(tf.math.lgamma(nu))
        )

        S2 = (
            2 ** input_dim
            * np.pi ** (input_dim / 2)
            * (2 * nu) ** nu
            / tf.pow(self.kernel.kernels[0].lengthscales, 2 * nu)
        )

        S3 = tf.pow(
            2 * nu / self.kernel.kernels[0].lengthscales ** 2 + eigenValues,
            (-nu - input_dim / 2.0),
        )

        # S = self.kernel.kernels[0].variance * \
        #     to_default_float(tf.pow(2.*np.pi, input_dim/2.)) * \
        #     tf.pow(self.kernel.kernels[0].lengthscales, input_dim) * \
        #     tf.exp(-eigenValues * tf.square(self.kernel.kernels[0].lengthscales)/2.)
        S = S1 * S2 * S3

        return S

    # def Kuu_rbf_dgp(self):
    #     """
    #     Make a representation of the Kuu matrices.
    #     """

    #     # domain, ms, input_dim = (lambda u: (u.domain, u.ms, u.input_dim))(inducing_variable)
    #     domain = self.domain
    #     input_dim = self.input_dim

    #     # Extract eigenvalues
    #     eigenValues = tf.constant(domain.eigenval(), dtype=gpflow.default_float())

    #     # Compute prior: magnSigma2*sqrt(2*pi)^d*lengthScale^d*exp(-w.^2*lengthScale^2/2)
    #     # Note that in Matlab the function is called on sqrt(lambda)
    #     # In this implementation, the square is instead omitted and the
    #     # function works directly on lambda (or omega squared)
    #     # if kern.name == 'squared_exponential':  # Equation 9

    #     sigma2_lin = self.kernel.kernels[1].variance

    #     omega = self.kernel.kernels[0].variance * \
    #         to_default_float(tf.pow(2.*np.pi, input_dim/2.)) * \
    #         tf.pow(self.kernel.kernels[0].lengthscales, input_dim) * \
    #         tf.exp(-eigenValues * tf.square(self.kernel.kernels[0].lengthscales)/2.)

    #     # return tf.linalg.diag(tf.concat(values=[sigma2_lin*np.ones(input_dim),1./omega],axis=0))
    #     return Diag(tf.concat(values=[sigma2_lin*np.ones(input_dim),1./omega],axis=0))


#
# @cov.Kuf.register(DGP, gpflow.kernels.RBF, TensorLike)
# def Kuf_rbf_dgp(self, X):

#     # domain, ms, input_dim = (lambda u: (u.domain, u.ms, u.input_dim))(inducing_variable)
#     # domain = inducing_variable.domain

#     return tf.transpose(self.domain.eigenfungrad(X))

# @gpflow.conditionals.conditional.register(
#     TensorLike, DGP, gpflow.kernels.Kernel, TensorLike
# )
# def conditional_dgp(
#     Xnew,
#     inducing_variable,
#     kernel,
#     f,
#     *,
#     full_cov=False,
#     full_output_cov=False,
#     q_sqrt=None,
#     white=False
# ):
#     fmean, fvar = 0, 0
#     return fmean, fvar

# if __name__ == '__main__':
#%%
m_basis = 8  # ** 3  # Requires power of 3
boundary = np.array([4, 4, 4])
cubic_domain = gp_domain(boundary, m_basis)
x_test = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#%%
# %timeit cubic_domain.eigenfungrad(x_test)
#%%
# print(cubic_domain.eigenfun(x_test))
print(cubic_domain.eigenfungrad(x_test))
# %%
# %timeit cubic_domain.eigenfungrad(x_test)
# %%
