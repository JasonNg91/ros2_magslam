import numpy as np
import scipy.io as sio
# from rotationfunctions import *

class loadDataDelft:
    def __init__(self, matfile: str, resample: int = 10, testInterval: int =10, trainPercentage: int =None, trainData: str ="start"):
        dict = sio.loadmat(matfile)
        self.N: int = int(dict['N'][0][0] / resample)
        self.ts: float = dict['T'][0][0] * resample
        self.g: np.ndarray = dict['g'].flatten()
        self.p: np.ndarray = dict['p_opti'][:,::resample].T #position in global
        self.q: np.ndarray = dict['q_opti'][:,::resample].T #body to global
        self.acc: np.ndarray = dict['u'][:3,::resample].T
        self.gyr: np.ndarray = dict['u'][3:6,::resample].T
        self.mag: np.ndarray = dict['u'][6:,::resample].T

        self.mag_g: np.ndarray = rotateVec_vmap(self.mag, self.q)
        self.mag_g_norm: np.ndarray = normalizeVec(self.mag_g)

def loadData(
    filename_training,
    filename_testing=None,
    testInterval=10,
    trainPercentage=None,
    trainData="start",
):

    data_training = np.loadtxt(filename_training, delimiter=",")
    Ndata_training = data_training.shape[0]
    divide_sample = 0

    if (filename_testing == None) or (type(filename_testing) == int):
        if type(filename_testing) == int:
            divide_sample = filename_testing
        if (trainPercentage != 0) or (trainPercentage != None):
            divide_sample = int(np.ceil(Ndata_training * trainPercentage / 100))
        filename_testing = filename_training

        if trainData == "end":
            trainIndices = range(Ndata_training - divide_sample, Ndata_training)
            testIndices = range(0, (Ndata_training - divide_sample), testInterval)
        else:
            trainIndices = range(0, divide_sample)
            testIndices = range(divide_sample, Ndata_training, testInterval)

    # print(filename_testing)
    # data_training = np.loadtxt(filename_training, delimiter=',')
    data_testing = np.loadtxt(filename_testing, delimiter=",")

    if divide_sample != 0:
        data_training = data_training[trainIndices, :]
        data_testing = data_testing[testIndices, :]

    x_train = data_training[:, :3]
    x_test = data_testing[:, :3]

    # # %offset position
    # x_train_center = (np.amin(x_train,axis=0) + np.amax(x_train,axis=0) )/2
    # x_train = x_train - x_train_center
    # x_test = x_test-x_train_center

    grad_y_train = data_training[:, 3:]
    grad_y_test = data_testing[:, 3:]

    traindata = (x_train, grad_y_train)
    testdata = (x_test, grad_y_test)

    return traindata, testdata


# %% Define domain


def determineBoundary(trainPositions, margin, testPositions=None):
    if testPositions is None:
        pos_max = trainPositions.max(axis=0) + margin
        pos_min = np.abs(trainPositions.min(axis=0)) + margin
    else:
        pos_max = np.vstack((testPositions, trainPositions)).max(axis=0) + margin
        pos_min = (
            np.abs(np.vstack((testPositions, trainPositions)).min(axis=0)) + margin
        )

    return np.vstack((pos_max, pos_min)).max(axis=0)


def evaluateTestPoints(m, testdata):
    # Predict test points
    (x_test, grad_y_test) = testdata
    Eft_test, Varft_test = m.predict_f(x_test)
    # Error metric
    N_test = x_test.shape[0]
    # grad_y_test_vec_3d = np.reshape(grad_y_test,[-1,1])

    e_rms = np.sqrt(
        np.sum((grad_y_test.ravel() - Eft_test.ravel()) ** 2) / (3 * N_test)
    )
    print("E_rms: {}".format(e_rms))


