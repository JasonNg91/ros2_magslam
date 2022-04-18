from math import prod
import objax
import numpy as nnp
import jax.numpy as np
import jax.scipy as sp
import time
import itertools
import math
from functools import partial
from jax import lax, grad, jit, vmap, jacrev, jacfwd, jvp
from bayesnewton.kernels import (
    Matern32,
    Kernel,
    Separable,
    StationaryKernel,
)  # , StationaryKernel, Independent
from bayesnewton.basemodels import BaseModel  # , GaussianProcess
from bayesnewton.utils import softplus, softplus_inv, solve, transpose
from bayesnewton.likelihoods import Gaussian

LOG2PI = math.log(2 * math.pi)

# [1] A. Solin and S. Särkkä, “Hilbert Space Methods for Reduced-Rank Gaussian Process Regression,” arXiv:1401.5508 [stat], Jan. 2014, Accessed: Oct. 18, 2019. [Online]. Available: http://arxiv.org/abs/1401.5508.
# [2] A. Solin, M. Kok, N. Wahlström, T. B. Schön, and S. Särkkä, “Modeling and interpolation of the ambient magnetic field by Gaussian processes,” arXiv:1509.04634 [cs, stat], Sep. 2015, Accessed: Oct. 18, 2019. [Online]. Available: http://arxiv.org/abs/1509.04634.

# transposed jacobian
def our_jacfwd(f):
    def jacfun(x):
        _jvp = lambda s: jvp(f, (x,), (s,))[1]
        Jt = vmap(_jvp, in_axes=1)(np.eye(len(x)))
        return Jt
        # return jnp.transpose(Jt)
    return jacfun
class gp_domain:
    def __init__(self, boundary, m):

        self.boundary = boundary
        self.input_dim = len(boundary)  # Input dimension
        self.m = m  # Total number of basis functions

        # m_hat = np.round(m ** (1 / input_dim)).astype(int)  # Number of basis function in each direction see 4.2 in [1]
        # j = np.array(list(itertools.product(range(1, m_hat + 1), repeat=input_dim)), dtype=float)  # Select m permutations
        # j = 0
        self.computePermutationList()
        self.computeEigenValuesAndIndices()

    def __len__(self):
        return self.m  # len(ms)

    def computePermutationList(self):
        # m_hat = np.round(self.m ** (1 / self.input_dim)).astype(int)
        m_hat = 20
        # self.j = tuple(itertools.product(range(1, m_hat + 1), repeat=self.input_dim))  # Select m permutations
        self.j = np.array(
            list(itertools.product(range(1, m_hat + 1), repeat=self.input_dim)),
            dtype=int,
        )  # Select m permutations

    def computeEigenValuesAndIndices(self):
        """
        Evaluate eigenvalues.

        Equation 16 from [2]
        """
        self.lambd2 = nnp.zeros(self.j.shape[0])  # Lambda

        for i in range(self.input_dim):
            self.lambd2 += (self.j[:, i] * nnp.pi / 2.0 / self.boundary[i]) ** 2

        # sort according eigenvalues
        sorted_indices = nnp.argsort(self.lambd2)

        # pick first m sorted values
        self.lambd2 = self.lambd2[sorted_indices][: self.m]
        self.j = self.j[sorted_indices][: self.m]

        # print(self.lambd2)

        # return lambd2 # return lambda squared

        # Numpy implementation

        # nx = x.shape[0]
        # jacobian = np.zeros((input_dim * x.shape[0], m))

        # npi = np.zeros(shape=(nx, m, input_dim))

        # for i in range(input_dim):
        #     npi[:, :, i] = np.kron(j[:, i].T * np.pi, ((x[:, i] + boundary[i])
        #                                                              / 2 / boundary[i])[:,np.newaxis])

        # sinnpi = np.sin(npi)
        # cosnpi = np.cos(npi)

        # lsqrtinv = np.prod(boundary)**(-1/2)

        # jacobian[0::input_dim, :] = np.pi / (2 * self.boundayr[0]) * lsqrtinv * j[:, 0] * cosnpi[:, :, 0] * sinnpi[:, :, 1] * sinnpi[:, :, 2]
        # jacobian[1::input_dim, :] = np.pi / (2 * boundary[1]) * lsqrtinv * j[:, 1] * cosnpi[:, :, 1] * sinnpi[:, :, 0] * sinnpi[:, :, 2]
        # jacobian[2::input_dim, :] = np.pi / (2 * boundary[2]) * lsqrtinv * j[:, 2] * cosnpi[:, :, 2] * sinnpi[:, :, 0] * sinnpi[:, :, 1]

        # return tf.concat(values=[tf.tile(np.eye(input_dim), (nx, 1)) ,tf.constant(jacobian)],axis=1)

    # def eigenval(self):
    #     """
    #     Evaluate eigenvalues.

    #     Equation 16 from [2]
    #     """
    #     lambd2 = np.zeros(self.m)  # Lambda

    #     for i in range(self.input_dim):
    #         lambd2 += (self.j[:, i] * np.pi / 2. / boundary[i]) ** 2

    #     return lambd2 # return lambda squared


def spectral_kernel_matern(nu, variance, lengthscale, eigenValues, input_dim):
    # TODO
    # assert self.kernel.kernels[0].name == '' # Halt if incorrect kernel

    # domain = self.domain
    # input_dim = self.input_dim
    # eigenValues = np.constant(domain.eigenval())#, dtype=gpflow.default_float())

    # eq10 in know your boundaries (Kok, Solin)
    S1 = variance * np.exp(lax.lgamma(nu + input_dim / 2.0)) / np.exp(lax.lgamma(nu))

    S2 = 2 ** input_dim * np.pi ** (input_dim / 2) * (2 * nu) ** nu / lengthscale ** (2 * nu)
    
    S3 = (2 * nu / lengthscale ** 2 + eigenValues) ** (-nu - input_dim / 2.0)

    # S = self.kernel.kernels[0].variance * \
    #     to_default_float(tf.pow(2.*np.pi, input_dim/2.)) * \
    #     tf.pow(self.kernel.kernels[0].lengthscales, input_dim) * \
    #     tf.exp(-eigenValues * tf.square(self.kernel.kernels[0].lengthscales)/2.)
    S = S1 * S2 * S3

    return S


class DGPmodel(BaseModel):
    def __init__(self, kernel, likelihood, X, Y, domain):
        # if isinstance(kernel, Separable):
        #     func_dim = 1
        # elif isinstance(kernel, Independent):
        #     func_dim = kernel.num_kernels
        # else:
        func_dim = 2  # lin + matern
        if X is not None:
            self.input_dim = X.shape[1]
        if Y is not None:
            self.output_dim = Y.shape[1]

        super().__init__(kernel, likelihood, X, Y, func_dim=func_dim)

        self.domain = domain
        self.ms = domain.m  # Number of basis functions
        self.m_lin = 3  # Linear kernel
        self.e3 = np.eye(3)

        # self.PhiArray = np.zeros(self.ms + 3)
        # self.vecY = np.reshape(Y, [-1, 1])

        #        self.NablaPhi = np.concatenate([np.tile(np.eye(self.input_dim), (self.num_data, 1)) ,self.eigenfungrad(self.X)],axis=1)
        #        self.NablaPhiNablaPhi = np.dot(np.transpose(self.NablaPhi),self.NablaPhi)
        #        self.NablaPhiY = np.dot(np.transpose(self.NablaPhi),self.vecY)

        if kernel.kernel1.__class__.__name__ == "Squared_Exponential":
            self.spectral_kernel = self.spectral_kernel_rbf

        if kernel.kernel1.__class__.__name__ == "Matern12":
            self.spectral_kernel = lambda: self.spectral_kernel_matern(1 / 2)
        elif kernel.kernel1.__class__.__name__ == "Matern32":
            self.spectral_kernel = lambda: self.spectral_kernel_matern(3 / 2)
        elif kernel.kernel1.__class__.__name__ == "Matern52":
            self.spectral_kernel = lambda: self.spectral_kernel_matern(5 / 2)

        self.eigenfun_const1 = 1. / nnp.sqrt(self.domain.boundary) # ** (-1 / 2)
        self.eigenfun_const2 = nnp.pi * self.domain.j / (2. * self.domain.boundary)

        # self.eigenfungrad = jit(jacfwd(self.eigenfun))
        # self.eigenfungrad_vmap = vmap(self.eigenfungrad)
        self.predict_seq_vmap = vmap( self.predict_seq, in_axes=(0, None, None) )

        self.NablaPhi = jacfwd(self.Phi)
        self.NablaPhi2 = jacrev(self.Phi)
        # self.NablaPhi_vmap = vmap(self.NablaPhi, in_axes=(None, 0, None), out_axes=0)
        # self.calculate_eigenvalues()

    def batch_processing(self, X, Y):
        self.__init__(
            kernel=self.kernel, likelihood=self.likelihood, X=X, Y=Y, domain=self.domain
        )
        self.vecY = np.reshape(Y, [-1, 1])
        self.calculate_eigenfunctions(X=X, vecY=self.vecY)
        self.calculate_eigenvalues()

    @partial(jit, static_argnums=(0,))
    def sequential_processing_step(self, x, y, mu, Sigma):
        # vecY = np.reshape(y, [-1, 1])
        # self.calculate_eigenfunctions(X=x, y=y)
        # self.calculate_eigenfunctions(X=x[None,:],vecY=vecY)

        # self.NablaPhi = transpose(
        #     np.concatenate([self.e3, self.eigenfungrad(x)])
        # )  # Measurement matrix

        NablaPhi = self.NablaPhi(x).T

        S = NablaPhi @ Sigma @ NablaPhi.T + self.likelihood.variance * np.eye(
            self.output_dim
        )

        HP = NablaPhi @ Sigma
        K = solve(S, HP).T

        mu = mu + K @ (y - NablaPhi @ mu)
        Sigma = Sigma - K @ HP

        # K = sp.linalg.solve(S, self.NablaPhi @ Sigma, sym_pos=True).T
        # #K = np.dot(Sigma, sp.linalg.solve(np.transpose(self.NablaPhi), S, assume_a='pos'))
        # mu = mu + K @ ( y - self.NablaPhi @ mu )
        # Sigma = Sigma - K @ S @ K.T
        # Sigma = (Sigma + Sigma.T)/2.

        return mu, Sigma


    @partial(jit, static_argnums=(0,))
    def predict_seq(self, xstar, mu, Sigma):
        
        NablaPhiStar_T = self.NablaPhi(xstar)
        # NablaPhiStar = NablaPhiStar_T.T
        fmean = NablaPhiStar_T.T @ mu
        fvar = NablaPhiStar_T.T @ Sigma @ NablaPhiStar_T

        return fmean, np.diag(fvar)


    # @partial(jit, static_argnums=(0,))
    # def predict_seq(self, Xstar, mu, Sigma):
    #     # NablaPhiStar = self.calculate_eigenfunctions(X=Xstar)
    #     # n_predict = Xstar.shape[0]
    #     # NablaPhiStar = np.concatenate([np.tile(np.eye(self.input_dim), (num_data, 1)) ,self.eigenfungrad(Xstar)],axis=1)
    #     # NablaPhiStar = np.concatenate([np.eye(3), self.eigenfungrad(Xstar)]).T
    #     tmp = self.NablaPhi_vmap(Xstar)
    #     NablaPhiStar = tmp.transpose(0,2,1).reshape(Xstar.shape[0]*3, -1)

    #     # NablaPhiStar = self.NablaPhi(Xstar).T
    #     fmean = NablaPhiStar @ mu
    #     fvar = NablaPhiStar @ Sigma @ NablaPhiStar.T

    #     return np.reshape(fmean, [-1, self.output_dim]), np.reshape(
    #         np.diag(fvar), [-1, self.output_dim]
    #     )

        # return mean, np.diag(covariance)
        # return np.reshape(mean,[-1,self.output_dim]), np.reshape(np.diag(covariance),[-1,self.output_dim])

    # @partial(jit, static_argnums=[1,2])

    @partial(jit, static_argnums=(0,))
    def eigenfun(self, x):  # x:nnp.ndarray or np.DeviceArray ,m ,boundary ,j ):
        """
        Evaluate eigenfunctions.
        """
        # boundary = self.domain.boundary
        # j = self.domain.j
        # m = self.ms
        # dim = j.shape[1]
        # (nx,input_dim) = x.shape

        # self.eigenfun_const1 = boundary ** ( -1 / 2 )
        # self.eigenfun_const2 = np.pi * j / (2. * boundary )
        ev = np.prod(
            self.eigenfun_const1
            * np.sin(self.eigenfun_const2 * (x + self.domain.boundary)),
            axis=1,
        )
        # ev = np.concatenate([np.ones(3), np.prod(self.eigenfun_const1 * np.sin(self.eigenfun_const2*(x+self.domain.boundary)),axis=1)])

        # ev = np.ones(m)
        # for i in range(dim):
        #     ev *= boundary[i] ** ( -1 / 2 ) * np.sin((x[i] + boundary[i] ) * np.pi * j[:, i] / 2. / boundary[i] )

        # 1D example
        # m1 = (np.pi / (2 * self.domain.boundary)) * np.tile(self.domain.boundary + x, self.ms)
        # m2 = np.diag(np.linspace(1, self.ms, num=self.ms))
        # num = np.sin(m1 @ m2)
        # den = np.sqrt(self.domain.boundary)
        # ev = num / den
        # return num / den

        return ev

    # def eigenfungrad(self,x):
    #     jacfwd(self.eigenfun)

    def eigenfungrad(self,x):#,m,boundary,j):
            """
            Evaluate the gradient of the eigenfunctions (16, 18, 19) in [2]
            :param x:
            :return:
            """

            # boundary = self.domain.boundary
            # j = self.domain.j
            # m = self.ms

            # (nx,input_dim) = x.shape
            # jacobian = np.zeros((input_dim * nx, m))

            # npi = np.zeros(shape=(nx, m, input_dim))

            # for i in range(input_dim):
            #     npi = npi.at[:,:,i].set(np.dot(((x[:, i][:,np.newaxis] + boundary[i]) / 2 / boundary[i]), j[:, i][np.newaxis,:] * np.pi))

            # sinnpi = np.sin(npi)
            # cosnpi = np.cos(npi)

            lsqrtinv = prod(self.domain.boundary)**(-1/2)

            const3 = np.pi / (2 * self.domain.boundary)

            sinnpi = np.sin( self.eigenfun_const2 * ( x + self.domain.boundary ) )
            cosnpi = np.cos( self.eigenfun_const2 * ( x + self.domain.boundary ) )

            jacobian1 = const3[0] * lsqrtinv * self.domain.j[:,0] * cosnpi[:, 0] * sinnpi[:, 1] * sinnpi[:, 2]
            jacobian2 = const3[1] * lsqrtinv * self.domain.j[:,1] * cosnpi[:, 1] * sinnpi[:, 0] * sinnpi[:, 2]
            jacobian3 = const3[2] * lsqrtinv * self.domain.j[:,2] * cosnpi[:, 2] * sinnpi[:, 0] * sinnpi[:, 1]
            jacobian = np.vstack([jacobian1,jacobian2,jacobian3])
            return jacobian

    def calculate_eigenfunctions(self, X, y):
        self.NablaPhi = np.concatenate([np.eye(3), self.eigenfungrad(X)]).T
        self.NablaPhiNablaPhi = self.NablaPhi.T @ self.NablaPhi
        self.NablaPhiY = y @ self.NablaPhi  # @ y[None,:]

        # self.NablaPhi = np.concatenate([np.tile(np.eye(self.input_dim), (self.num_data, 1)) ,self.eigenfungrad(X)],axis=1)
        # self.NablaPhiNablaPhi = np.dot(np.transpose(self.NablaPhi),self.NablaPhi)
        # self.NablaPhiY = np.dot(np.transpose(self.NablaPhi),vecY)

    def calculate_spectral_eigenvalues(self):
        sigma2_lin = self.kernel.kernel0.variance

        # sigma2_lin = self.kernel.kernels[1].variance # linear variance
        # Equation 20 in Solin 2015
        self.Lambda = np.concatenate(
            [sigma2_lin * np.ones(self.input_dim), self.spectral_kernel()], axis=0
        )
        # Calculate Lambda (eq 20 in [2])

    def calculate_eigenvalues(self):
        sigma2 = self.likelihood.variance  # meas noise
        self.calculate_spectral_eigenvalues()

        # # Calculate the Cholesky factor
        self.L = np.linalg.cholesky(
            self.NablaPhiNablaPhi + np.diag(sigma2 / self.Lambda)
        )  # O(m^3)

        self.v = lax.linalg.triangular_solve(
            self.L, self.NablaPhiY, lower=True, left_side=True
        )

        self.foo = lax.linalg.triangular_solve(
            self.L, self.v, lower=True, left_side=True, transpose_a=True
        )

        # alternatively:
        self.foo = np.linalg.solve(
            self.NablaPhiNablaPhi + np.diag(sigma2 / self.Lambda), self.NablaPhiY
        )

    def spectral_kernel_rbf(self):

        # assert self.kernel.kernels[0].name == 'squared_exponential' # Halt if incorrect kernel

        # domain = self.domain
        input_dim = self.input_dim
        eigenValues = self.domain.lambd2

        variance = self.kernel.kernel1.variance
        lengthscale = self.kernel.kernel1.lengthscale

        S = (
            variance
            * np.power(2.0 * np.pi, input_dim / 2.0)
            * np.power(lengthscale, input_dim)
            * np.exp(-eigenValues * np.square(lengthscale) / 2.0)
        )

        return S

    def spectral_kernel_matern(self, nu):
        # TODO
        # assert self.kernel.kernels[0].name == '' # Halt if incorrect kernel

        # domain = self.domain
        input_dim = self.input_dim
        eigenValues = (
            self.domain.lambd2
        )  # tf.constant(domain.eigenval(), dtype=gpflow.default_float())

        variance = self.kernel.kernel1.variance
        lengthscale = self.kernel.kernel1.lengthscale

        # eq10 in know your boundaries (Kok, Solin)
        S1 = (
            variance * np.exp(lax.lgamma(nu + input_dim / 2.0)) / np.exp(lax.lgamma(nu))
        )

        S2 = (
            2 ** input_dim
            * np.pi ** (input_dim / 2)
            * (2 * nu) ** nu
            / np.power(lengthscale, 2 * nu)
        )

        S3 = np.power(2 * nu / lengthscale ** 2 + eigenValues, (-nu - input_dim / 2.0))

        # S = self.kernel.kernels[0].variance * \
        #     to_default_float(tf.pow(2.*np.pi, input_dim/2.)) * \
        #     tf.pow(self.kernel.kernels[0].lengthscales, input_dim) * \
        #     tf.exp(-eigenValues * tf.square(self.kernel.kernels[0].lengthscales)/2.)
        S = S1 * S2 * S3

        return S

        # self.data = data # Training data
        # self.n_data, self.input_dim = self.X
        # self.output_dim = Y.shape[1]
        # self.vecY = tf.reshape(data[1], [-1,1]) # Vectorize outputs

        # # Calculate the eigenfunctions based on domain and training data
        # self.NablaPhi = tf.concat(values=[tf.tile(np.eye(self.input_dim), (self.n_data, 1)) ,tf.constant(self.domain.eigenfungrad(self.data[0]))],axis=1)
        # self.NablaPhiNablaPhi = tf.matmul(self.NablaPhi,self.NablaPhi,transpose_a=True)
        # self.NablaPhiY = tf.matmul(self.NablaPhi,self.vecY,transpose_a=True)

        # # initiate eigenvalues based on training data, domain and (initial) hyperparams
        # if kernel.kernels[0].name == 'squared_exponential':
        #     self.spectral_kernel = self.spectral_kernel_rbf

        # if kernel.kernels[0].name == 'matern12':
        #     self.spectral_kernel = lambda: self.spectral_kernel_matern(1/2)
        # elif kernel.kernels[0].name == 'matern32':
        #     self.spectral_kernel = lambda: self.spectral_kernel_matern(3/2)
        # elif kernel.kernels[0].name == 'matern52':
        #     self.spectral_kernel = lambda: self.spectral_kernel_matern(5/2)

        # self.calculate_eigenvalues()

        # self.pi = tf.constant(math.pi, dtype=tf.float64)

    def energy(self, batch_ind=None, **kwargs):

        ## Extract parameters
        sigma2 = self.likelihood.variance  # meas noise

        # # % Number of n=observations*3 and m=basis functions
        n = self.num_data * 3
        m = self.NablaPhiY.shape[0]

        # % Evaluate terms based on eigenvalues, eigenvalues depend on hyperparam values
        self.calculate_eigenvalues()

        yiQy = (
            np.dot(np.transpose(self.vecY), self.vecY)
            - np.dot(np.transpose(self.v), self.v)
        ) / sigma2
        logdetQ = (
            (self.input_dim * n - m) * np.log(sigma2)
            + np.sum(np.log(self.Lambda))
            + 2 * np.sum(np.log(np.diag(self.L)))
        )

        # % Return approx. negative log marginal likelihood
        e = 0.5 * (np.squeeze(yiQy + logdetQ) + n * self.input_dim * LOG2PI)
        return e

    # @partial(jit, static_argnums=(0,))
    def Phi(self, x):
        #equation 18 [2]
        sinTerm = self.eigenfun_const1 * np.sin(self.eigenfun_const2 * (x + self.domain.boundary))
        # sinTerm2 = self.eigenfun_const1 * np.sin(np.kron(self.eigenfun_const2, x[:,None].T + self.domain.boundary))
        prodTerm = np.prod(sinTerm, axis=1)
        Phi = np.concatenate((x, prodTerm))
        # Phi = np.zeros(self.ms + 3)
        # Phi = Phi.at[:3].set(x)
        # Phi = Phi.at[3:].set(prodTerm)

        return Phi

    

    def predict(
        self, X: np.ndarray, R=None
    ):  # , full_cov = False, full_output_cov: bool = False):

        n_predict = X.shape[0]
        sigma2 = self.likelihood.variance  # meas noise

        # NablaPhit = np.concatenate(
        #     [np.tile(np.eye(self.input_dim), (n_predict, 1)), self.eigenfungrad_vmap(X)],
        #     axis=1,
        # )

        NablaPhit = self.NablaPhi_vmap(X).transpose(0,2,1).reshape(n_predict*3,-1)

        fvar = np.expand_dims(
            sigma2
            * np.sum(
                (
                    lax.linalg.triangular_solve(
                        self.L, NablaPhit.T, left_side=True, lower=True
                    )
                )
                ** 2,
                axis=0,
            ),
            axis=1,
        )
        fmean = np.dot(NablaPhit, self.foo)  # eq(14)

        return np.reshape(fmean, [-1, self.output_dim]), np.reshape(
            fvar, [-1, self.output_dim]
        )

def Phi(x, const1, const2, boundary):
    #equation 18 [2]
    sinTerm = const1 * np.sin(const2 * (x + boundary))
    prodTerm = np.prod(sinTerm, axis=1)
    Phi = np.concatenate((x, prodTerm))
    # Phi = np.zeros(self.ms + 3)
    # Phi = Phi.at[:3].set(x)
    # Phi = Phi.at[3:].set(prodTerm)

    return Phi


class Linear(Kernel):
    def __init__(self, variance=1.0, fix_variance=0):

        # check whether the parameters are to be optimised
        if fix_variance:
            self.transformed_variance = objax.StateVar(softplus_inv(np.array(variance)))
        else:
            self.transformed_variance = objax.TrainVar(softplus_inv(np.array(variance)))

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def K(self, X, X2):
        return np.dot(X * self.variance, np.transpose(X2))


class Squared_Exponential(StationaryKernel):

    """
    The Squared Exponential kernel. The kernel equation is

    k(r) = σ² exp{-0.5*r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    @property
    def state_dim(self):
        return 1

    def K_r(self, r):
        return self.variance * np.exp(-0.5 * r)

    # def kernel_to_state_space(self, R=None):
    #     F = np.array([[-1.0 / self.lengthscale]])
    #     L = np.array([[1.0]])
    #     Qc = np.array([[2.0 * self.variance / self.lengthscale]])
    #     H = np.array([[1.0]])
    #     Pinf = np.array([[self.variance]])
    #     return F, L, Qc, H, Pinf

    # def stationary_covariance(self):
    #     Pinf = np.array([[self.variance]])
    #     return Pinf

    # def measurement_model(self):
    #     H = np.array([[1.0]])
    #     return H

    # def state_transition(self, dt):
    #     """
    #     Calculation of the discrete-time state transition matrix A = expm(FΔt) for the exponential prior.
    #     :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
    #     :return: state transition matrix A [1, 1]
    #     """
    #     A = np.broadcast_to(np.exp(-dt / self.lengthscale), [1, 1])
    #     return A

    # def feedback_matrix(self):
    #     F = np.array([[-1.0 / self.lengthscale]])
    #     return F
