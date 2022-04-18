import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm

# %% Plot train and test data for specific magnetic axis in x-y plane

def plotTrainAndTestDataIn2DPlane(traindata,testdata=[],mag_axis=0):
    (x_train, grad_y_train) = traindata
    testdataFlag = 1
    if not testdata:
        #no testdata
        testdataFlag = 0
    else:
        (x_test, grad_y_test) = testdata

    if testdataFlag:
        zs = np.concatenate([grad_y_train[:,mag_axis], grad_y_test[:,mag_axis]], axis=0)
    else:
        zs = grad_y_train[:,mag_axis]
    min_, max_ = zs.min(), zs.max()
    norm = plt.Normalize(min_, max_)

    plt.figure()
    plt.scatter(x_train[:,0],x_train[:,1],c=grad_y_train[:,mag_axis],s=5,norm=norm)
    if testdataFlag:
        plt.scatter(x_test[:,0],x_test[:,1],c=grad_y_test[:,mag_axis],s=50,norm=norm)
    plt.axis('equal')
    plt.grid()
    plt.colorbar()


# %% Plot predictions and uncertainties
def plotTimeSeriesWithUncertainty(testdata, estimatedTestdata, saveFilename=[], **kwargs):
    (_,grad_y_test) = testdata
    (Eft_test,Varft_test) = estimatedTestdata
    #colors = ('C0','M0','Y0')
    colors = ('r','g','b')#('C0','C1','C2')
    axisLabels = ('x','y','z')
    fig, axs = plt.subplots(3, figsize=(20,8))
    for axis in range(3):
        xaxis = np.arange(0,grad_y_test.shape[0])#x_test[:,axis]
        # plt.figure()#figsize=(12, 6))
        axs[axis].plot(xaxis,grad_y_test[:,axis], color=colors[axis][0], linestyle='dashed', mew=1, label=axisLabels[axis])
        axs[axis].plot(xaxis,Eft_test[:,axis], color=colors[axis][0], lw=2)
        axs[axis].fill_between(
            xaxis,
            Eft_test[:,axis] - 1.96 * np.sqrt(Varft_test[:,axis]),
            Eft_test[:,axis] + 1.96 * np.sqrt(Varft_test[:,axis]),
            color=colors[axis],
            alpha=0.2,
        )
        axs[axis].set_ylabel('Mag. Fieldstrength (a.u.)')
        axs[axis].legend()
        axs[axis].grid() 
    #axs[axis].ylabel('Magnetic fieldstrength (a.u.)')
    # plt.legend(['x_t','y_t','z_t','x_e','y_e','z_e'])  
     
    plt.xlabel('Sample (n)')
    plt.ylabel('Magnetic fieldstrength (a.u.)')
    if saveFilename:
        plt.savefig(saveFilename,optimize=True,progressive=True,bbox_inches='tight')
    plt.show(**kwargs)

# %% Plot train and test data for specific magnetic axis in x-y plane
def scatterPlotTestTrain(traindata,testdata,mag_axis=0):
    (x_train,grad_y_train) = traindata
    (x_test,grad_y_test) = testdata

    fig,ax = plt.subplots(ncols=2,nrows=1)
    if mag_axis == 3:

        magNormTrain = np.linalg.norm(grad_y_train,axis=1)
        magNormTest = np.linalg.norm(grad_y_test,axis=1)
        zs = np.concatenate([magNormTrain,magNormTest], axis=0)
        min_, max_ = zs.min(), zs.max()
        norm = plt.Normalize(min_, max_)

        pcm1 = ax[0].scatter(x_train[:,0],x_train[:,1],c=magNormTrain,s=5,norm=norm)
        
        pcm2 = ax[1].scatter(x_test[:,0],x_test[:,1],c=magNormTest,s=5,norm=norm)
    else:
        zs = np.concatenate([grad_y_train[:,mag_axis], grad_y_test[:,mag_axis]], axis=0)
        min_, max_ = zs.min(), zs.max()
        norm = plt.Normalize(min_, max_)


        pcm1 = ax[0].scatter(x_train[:,0],x_train[:,1],c=grad_y_train[:,mag_axis],s=5,norm=norm)
        pcm2 = ax[1].scatter(x_test[:,0],x_test[:,1],c=grad_y_test[:,mag_axis],s=5,norm=norm)
    ax[0].set_title('train data')
    ax[1].set_title('test data')

    for a in ax:
        a.axis('equal')
        a.grid()
    # plt.colorbar(pcm1, ax=ax[0])
    plt.colorbar(pcm2, ax=ax[1])


    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    #scatterPlotTestTrain()

# %% Plot train and test data for specific magnetic axis in x-y plane
def scatterPlotTestTrainEst(traindata,testdata,Eft_test=[],mag_axis = 0):
    (x_train,grad_y_train) = traindata
    (x_test,grad_y_test) = testdata
    # zs = np.concatenate([grad_y_train[:,mag_axis], grad_y_test[:,mag_axis], Eft_test[:,mag_axis]], axis=0)
    zs = np.concatenate([grad_y_train[:,mag_axis], grad_y_test[:,mag_axis]], axis=0)
    min_, max_ = zs.min(), zs.max()
    norm = plt.Normalize(min_, max_)

    fig,ax = plt.subplots(ncols=2,nrows=1)
    ax[0].scatter(x_train[:,0],x_train[:,1],c=grad_y_train[:,mag_axis],s=5,norm=norm)
    im1=ax[0].scatter(x_test[:,0],x_test[:,1],c=grad_y_test[:,mag_axis],s=50,norm=norm)
    ax[0].set_title('train and true data')
    ax[0].legend(('train','true'))
    plt.colorbar(im1,ax=ax[0])

    ax[1].scatter(x_train[:,0],x_train[:,1],c=grad_y_train[:,mag_axis],s=5,norm=norm)
    im2=ax[1].scatter(x_test[:,0],x_test[:,1],c=Eft_test[:,mag_axis],s=50,norm=norm)
    ax[1].set_title('train and estimated data')
    ax[1].legend(('train','estimate'))
    plt.colorbar(im2,ax=ax[1])

    for a in ax: 
        a.grid()
        a.axis('equal')

    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())


# %% Create contour plots within the specified domain at z=0 for specified magnetic axis
def contourPlots(m,mag_axis=0,contour=[],delta=3/18,plotMean=True,testdata=[],saveFilename=[]):
    # mag_axis=0

    # delta = 3/18 # spatial resolution
    boundary_ext = .5 # extend the boundary

    if not contour:
        L = m.domain.L
        x = np.arange(m.domain.L[0][0]-boundary_ext, m.domain.L[0][1]+delta+boundary_ext, delta)
        y = np.arange(m.domain.L[1][0]-boundary_ext, m.domain.L[1][1]+delta+boundary_ext, delta)
    else:
        L = contour
        x = np.arange(contour[0][0], contour[0][1]+delta, delta)
        y = np.arange(contour[1][0], contour[1][1]+delta, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    mesh = (X,Y,Z)

    x_grid = np.vstack((X.flatten(), Y.flatten(), np.zeros(np.prod(X.shape)))).T

    Eft_grid_3d,Varft_grid_3d = m.predict_f(x_grid)
    Eft_grid = np.reshape(Eft_grid_3d,[-1,3])
    Varft_grid = np.reshape(Varft_grid_3d,[-1,3])

    Eft_axis = np.reshape(Eft_grid[:,mag_axis],X.shape)
    Varft_axis = np.reshape(Varft_grid[:,mag_axis],X.shape)

    # contour plot

    # fig, ax = plt.subplots()
    # CS = ax.contour(X, Y, Eft_axis)
    # ax.clabel(CS, inline=1, fontsize=10)
    # ax.grid('grid')
    # ax.axis('equal')
    # # ax.set_title('Simplest default with labels')

    #fig, ax = plt.subplots()
    if plotMean:
        zComponent = Eft_axis
    else:
        zComponent = Varft_axis        
    contours = plt.contour(X, Y, zComponent, levels=5, colors='black')
    plt.clabel(contours, inline=True, fontsize=8, )

    if testdata:
        (x_test,grad_y_test) = testdata
        zs = np.concatenate([grad_y_test[:,mag_axis], Eft_axis[:,mag_axis]], axis=0)
        min_, max_ = zs.min(), zs.max()
        norm = plt.Normalize(min_, max_)
        plt.imshow(zComponent, extent=[L[0][0], L[0][1], L[1][0], L[1][1]], origin='lower',
            cmap='RdGy', alpha=1, norm=norm)

        plt.scatter(x_test[:,0],x_test[:,1],c=grad_y_test[:,mag_axis],s=20,cmap='RdGy',norm=norm,alpha=1,edgecolors='k',linewidths=.1,marker='P')
        plt.plot(x_test[:,0],x_test[:,1],alpha=1,lw=.2)
    else:
        plt.imshow(zComponent, extent=[L[0][0], L[0][1], L[1][0], L[1][1]], origin='lower',
            cmap='RdGy', alpha=1)

    plt.colorbar()
    plt.grid()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    if saveFilename:
        plt.savefig(saveFilename,optimize=True,progressive=True,bbox_inches='tight')

    return mesh, Eft_grid

# %% surface plot
def surfacePlot():
    from mpl_toolkits.mplot3d import Axes3D 
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Eft_axis, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# %% Quiver (vector field) plot
def quiverPlot(mesh,Eft_grid):

    (X,Y,Z) = mesh

    # Some aiding functions
    def set_axes_radius(ax, origin, radius):
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def set_axes_equal(ax):
        # '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        # cubes as cubes, etc..  This is one possible solution to Matplotlib's
        # ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        # Input
        # ax: a matplotlib axis, e.g., as output from plt.gca().
        # '''

        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        set_axes_radius(ax, origin, radius)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Eft_grid = Eft_grid/np.linalg.norm(Eft_grid,axis=1)[:,None]

    u = np.reshape(Eft_grid[:,0],X.shape)
    v = np.reshape(Eft_grid[:,1],X.shape)
    w = np.zeros_like(v)#np.reshape(Eft_grid[:,2],X.shape) #plot only x-y components

    ax.quiver(X,Y,Z,u,v,w)
    ax.view_init(elev=30, azim=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    set_axes_equal(ax)

    plt.show()

def transparencyHeatMap(minvals, maxvals, pos_xy, plotdata, fig, ax):

    n_data = pos_xy.shape[0]

    xx, yy = np.meshgrid(pos_xy[:,0], pos_xy[:,1])

    (xyzmean, xyzcov) = plotdata
    mean_mesh = xyzmean[:,0].reshape(n_data,-1)
    cov_mesh = xyzcov[:,0].reshape(n_data,-1)

    weights = mean_mesh
    alpha = 1/cov_mesh
    
    # We'll also create a grey background into which the pixels will fade
    #greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)
    # vmax = np.abs(weights).max()
    imshow_kwargs = {
    'vmax': weights.max(),
    'vmin': weights.min(),
    'cmap': 'jet',
    'extent': (minvals[0], maxvals[0], minvals[0], maxvals[1]),
}

    # Create an alpha channel based on weight values
    # Any value whose absolute value is > .0001 will have zero transparency
    # alphas = SymLogNorm(linthresh=0.03, linscale=0.03, base=10)(alpha)#0, .9, clip=True)(np.abs(alpha))
    alphas = Normalize()(alpha)
    alphas = np.clip(alphas, .001, 1)  # alpha value clipped at the bottom at .4

    # Create the figure and image
    #fig, ax = plt.subplots(figsize=(12,12))
    
    # Note that the absolute values may be slightly different

    # get rid of the frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    # ax.imshow(greys)
    # ax.pcolormesh(xx, yy, weights, alpha=alphas)#, **imshow_kwargs)
    pcm = ax.imshow(weights, alpha= alphas, origin="lower", **imshow_kwargs)
    #fig.colorbar(pcm, ax=ax)#, extend='both')#, shrink=0.5, aspect=5)

    # plt.show()