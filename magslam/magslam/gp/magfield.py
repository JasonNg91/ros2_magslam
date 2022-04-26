import sys
from pathlib import Path

working_dir = str(Path(__file__).parent)
sys.path.append(working_dir)# / "../"))

import DGP.dgp_domain_mag_jax as dgp_jax
import jax.numpy as jnp
import numpy as np

# import time
import matplotlib.pyplot as plt
from jax import jit, make_jaxpr, jacfwd, jacrev

# from jax import lax, grad, jit, vmap, jacrev, jacfwd,

from DGP.utils import loadData, determineBoundary

class GPMagMap:
	def __init__(self, boundary=[2.5, 2.5, 1.0], m_basis=1000, lin_var=1.0, stat_var=0.005, stat_ls=0.2, lik_var=0.02):
		self.boundary = np.array(boundary)
		
		# Setup domain
		self.cubic_domain = dgp_jax.gp_domain(boundary=self.boundary,m=m_basis)
		
		# Setup GP Model
		self.kern_lin = dgp_jax.Linear(variance=lin_var)
		self.kern_stationary = dgp_jax.Matern32(variance=stat_var, lengthscale=stat_ls)
		self.kern = dgp_jax.Separable([self.kern_lin, self.kern_stationary])
		self.lik = dgp_jax.Gaussian(variance=lik_var, fix_variance=True)

		self.model_seq = dgp_jax.DGPmodel(kernel=self.kern, likelihood=self.lik,  X=np.zeros((1,3)), Y=np.zeros((1,3)), domain=self.cubic_domain)

		self.model_seq.calculate_spectral_eigenvalues()
		self.Sigma = np.diag(self.model_seq.Lambda) # initial mean
		self.mu = np.zeros(self.model_seq.ms + self.model_seq.input_dim) # initial cov
	
	def train(self,x,y):
		self.mu, self.Sigma = self.model_seq.sequential_processing_step(x,y,self.mu,self.Sigma)
	
	def predict(self,x):
		mean, cov = self.model_seq.predict_seq(xstar=x, mu=self.mu, Sigma=self.Sigma)
		return mean, cov
