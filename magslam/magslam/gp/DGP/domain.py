# Copyright 2018-2019 Manon Kok and Arno Solin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import numpy as np
from scipy.sparse import coo_matrix#, lil_matrix, find
from scipy.sparse.linalg import eigsh

class gp_domain:

    def __init__(self, mask, xlim, ylim, m):
        """
        Make domain from mask.
        """

        # Assert that the mask represents a square area
        assert mask.shape[0]==mask.shape[1]
        assert xlim==ylim

        # t = time.time()
        # Composition of the stencil matrix is based on the 9-point rule
        I,J = np.where(mask) #Give row,col values of True region

        # Preload constructing sparse array
        row = np.empty( shape=(9*len(I),))
        col = np.empty( shape=(9*len(I),))
        val = np.empty( shape=(9*len(I),))

        # Define the operator
        OP = np.array([[1/6,2/3,1/6], [2/3,-10/3,2/3], [1/6,2/3,1/6]])

        j = 0

        for k in range(0,len(I)):
          for di in [-1,0,1]:
            for dj in [-1,0,1]:
              i = np.where((I==I[k]+di) & (J==J[k]+dj))
              len_i = len(i[0])
              if len_i>0:
                  row[j] = k
                  col[j] = i[0]
                  val[j] = OP[di+1,dj+1]
                  j = j+1

        # Delete unused part
        row = row[:j]
        col = col[:j]
        val = val[:j]


        # I,J,_ = find(mask)
        # n = len(I)
        # # t = time.time()
        # S_h = lil_matrix((n,n))

        # for k in range(0,n):
        #     i = I[k]
        #     j = J[k]
        #     S_h[k, (I==i+1)&(J==j+1)]=  1/6
        #     S_h[k, (I==i+1)&(J==j)]  =  2/3
        #     S_h[k, (I==i+1)&(J==j-1)]=  1/6
        #     S_h[k, (I==i)&(J==j+1)]=    2/3
        #     S_h[k, (I==i)&(J==j)]=    -10/3
        #     S_h[k, (I==i)&(J==j-1)]=    2/3
        #     S_h[k, (I==i-1)&(J==j+1)]=  1/6
        #     S_h[k, (I==i-1)&(J==j)]=    2/3
        #     S_h[k, (I==i-1)&(J==j-1)]=  1/6

        # Scale by step size
        hx = (xlim[1]-xlim[0])/(mask.shape[1]-1)
        val /= hx**2

        # Construct the pencil matrix
        S_h = coo_matrix((val, (row, col)), shape=(len(I), len(I)))

        # print(time.time()-t)

        # Solve eigenvalue problem
        mu,V = eigsh(S_h, k=m, which='LA')

        # Better approximations of the eigenvalues
        self.hlambda = np.flipud(2*mu / (np.sqrt(1 + mu*hx**2/3) + 1))
  
        # Address scaling issues
        V = V * 1./hx

        # Expand size to match mask
        Vsquare = np.zeros((mask.shape[0]*mask.shape[1],m))
        ind, = np.where(mask.flatten())
        for i in range(len(ind)):
            Vsquare[ind[i],:] = V[i,:]

        # Store eigenvectors and mask
        self.V = np.fliplr(Vsquare)
        self.mask = mask
        self.x1 = np.linspace(xlim[0],xlim[1],mask.shape[1])
        self.x2 = np.linspace(ylim[0],ylim[1],mask.shape[0])
        self.m = m
        self.S_h = S_h

    def eigenfun(self,x):
        """
        Evaluate eigenfunctions.
        """
        foo = self.V.reshape((self.mask.shape[0],self.mask.shape[1],self.m))
        U = np.zeros((x.shape[0],self.m))
        for k in range(x.shape[0]):
            i = np.abs(self.x1-x[k,0]).argmin()
            j = np.abs(self.x2-x[k,1]).argmin()
            U[k,:] = foo[j,i,:].flatten()
        return U

    def eigenval(self):
        """
        Evaluate eigenvalues.
        """
        return -self.hlambda
