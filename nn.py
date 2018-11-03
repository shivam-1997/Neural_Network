import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from aux import *

m =  24         # Number of features
n = 32000       # Number of training data points
alpha = 0.001    # learning rate
lmbda = 0   # regularisation parameter
# thresh = 0.5
p = 2           # p-norm will be used
nIterations = 20000
nHiddenUnits = 100
batchSize = 100
nOutputUnits = 3
################################################################################

train_file = "train.csv" #sys.argv[1]
test_file = "test.csv" #sys.argv[2]
out_file = 'submission.csv'


day2num = {                                 # Dictionary to map days to numbers
    'monday': 1, 'tuesday': 2,
    'wednesday': 3, 'thursday': 4,
    'friday': 5, 'saturday': 6,
    'sunday': 7
}
################################################################################

class ActivationFun_Grad:
    def sigmoid(self, s):
        # print(-s[0])
        # print()
        # print(np.exp(-s[0][0]))
        # input()
        return 1.0 / ( 1.0 + np.exp( -s ) )
    def sigmoid_der(self, x):
        # print(x)
        # input()
        return x * (1-x)
    def softmax(self, s):
        sum = np.sum(np.exp(s), axis = 1)
        num = np.exp(s)
        res = num / sum[:,None]
        # print('num',num[:5,:],sep='\n')
        # print('sum',sum[:5],sep='\n')
        # print('res',res[:5,:],sep='\n')
        # print(np.sum(res, axis = 1))
        # input()
        return res
    def softmax_1d_array(self, a):
        return np.exp(a)/np.sum(np.exp(a))
    def cross_Entropy_Error( self, trainigLabel, yHat):
        error = np.dot( trainingLabel, np.log(yHat)) + np.dot( ( 1 - trainingLabel ), np.log( 1 - yHat ) )
        error = error.sum()
        return -error
    def grad_Output_2_Hidden( self, outputHidden, outputFinal_minus_trainingLabel):
        return np.dot( outputHidden.T, outputFinal_minus_trainingLabel )
    def grad_Hidden_2_input( self, outputHidden, outputFinal_minus_trainingLabel, oldWtsHidden2Output, X):
    #     print('outputHidden',outputHidden.shape)
    #     print('outputFinal_minus_trainingLabel',outputFinal_minus_trainingLabel.shape)
    #     print('oldWtsHidden2Output',oldWtsHidden2Output.shape)
    #     print('X',X.shape)                                    """batchSize x (m+1)"""
        term1 = np.dot( outputFinal_minus_trainingLabel, oldWtsHidden2Output.T)   #"""batchSize x nHiddenUnits"""
        term2 = ActivationFun_Grad().sigmoid_der( outputHidden )                  #"""batchSize x nHiddenUnits"""
        finalTerm = np.dot( ( term1 * term2).T , X)                               #"""nHiddenUnits x (m+1)"""
        # term3 = np.dot( term2.T, X )
        # finalTerm = np.dot( term1, term3)
        # # print('grad_Hidden_2_input shape: ', finalTerm.shape)
        # input()
        # return finalTerm.sum()
        return finalTerm.T

class Train:

    def __init__(self):
        self.feat_scal_avg, self.feat_scal_diff, self.feat_min, self.feat_std, self.wtsInput2Hidden, self.wtsHidden2Output = self.train_network()

    def initialise(self):
        X, y = load_data()
        feat_scal_avg = np.zeros(m+1)
        feat_scal_diff = np.ones(m+1)
        feat_min = np.zeros(m+1)
        feat_std = np.ones(m+1)
        i = 1
        while i <= m:
            X[i], feat_scal_avg[i], feat_scal_diff[i], feat_min[i], feat_std[i]  = normalisation(X[i])
            i += 1

        trainingLabel = np.empty((0, nOutputUnits))
        i = 0
        while i<len(y):
            if y[i] == 1:
                trainingLabel = np.concatenate((trainingLabel, [[ 1, 0, 0 ]]), axis = 0)
            elif y[i] == 2:
                trainingLabel = np.concatenate((trainingLabel, [[ 0, 1, 0 ]]), axis = 0)
            else:
                trainingLabel = np.concatenate((trainingLabel, [[ 0, 0, 1 ]]), axis = 0)
            i = i+1

        return X, y, trainingLabel, feat_scal_avg, feat_scal_diff, feat_min, feat_std

    def train_network(self):
        print('total number of iterations =', nIterations)
        X, y, trainingLabel, feat_scal_avg, feat_scal_diff, feat_min, feat_std = self.initialise()
        # plot_all(X, y)
        X = X.transpose()
        X = X[1:]
        print('number of observations: ', X.shape[0])
        wtsInput2Hidden, wtsHidden2Output = self.training(X, y, trainingLabel)
        return feat_scal_avg, feat_scal_diff, feat_min, feat_std, wtsInput2Hidden, wtsHidden2Output

    def forward_pass( self, X, wtsInput2Hidden, wtsHidden2Output):
        outputHidden = ActivationFun_Grad().sigmoid( np.dot( X, wtsInput2Hidden) )
        outputFinal = ActivationFun_Grad().softmax( np.dot( outputHidden, wtsHidden2Output ) )
        return outputHidden, outputFinal

    def backward_pass( self, X, trainingLabel, wtsInput2Hidden, outputHidden, wtsHidden2Output, outputFinal):
        outputFinal_minus_trainingLabel = outputFinal - trainingLabel
        # print('outputFinal_minus_trainingLabel: ', outputFinal_minus_trainingLabel.shape)
        tempWtsHidden2Output = wtsHidden2Output - alpha * ActivationFun_Grad().grad_Output_2_Hidden( outputHidden,
                                                                               outputFinal_minus_trainingLabel)
        grad = ActivationFun_Grad().grad_Hidden_2_input(outputHidden, outputFinal_minus_trainingLabel,
                                                        wtsHidden2Output, X)
        # grad = grad.T
        # print(grad.shape)
        # input()
        tempWtsInput2Hidden = wtsInput2Hidden - alpha * grad
        return tempWtsInput2Hidden, tempWtsHidden2Output

    def training(self, X, y, trainingLabel):
        np.random.seed(1)
        wtsInput2Hidden = 2 * np.random.random( (m+1, nHiddenUnits) ) - 1
        # print('wtsInput2Hidden', wtsInput2Hidden[:,:], sep='\n' )
        print('dimension of wtsInput2Hidden: ', wtsInput2Hidden.shape)
        wtsHidden2Output =  2 * np.random.random( (nHiddenUnits, nOutputUnits) ) - 1
        print( 'dimension of wtsHidden2Output: ', wtsHidden2Output.shape)

        for iterations in range(nIterations):
            if iterations % 100 == 0:
                print(iterations)
                # print('wtsInput2Hidden',wtsInput2Hidden.shape, wtsInput2Hidden, sep='\n')
                # print('wtsHidden2Output',wtsHidden2Output.shape, wtsHidden2Output, sep='\n')

            allLabels = np.array([])
            i = 0
            while i < len(y):
                # if iterations % 100 == 0:
                #     print(iterations, i )
                trainingBatch = X[i:i+batchSize,:]
                # print('trainingBatch', trainingBatch, sep='\n')
                # print('shape of training batch: ',trainingBatch.shape)
                # input()
                outputHidden, outputFinal = self.forward_pass( trainingBatch,
                                                         wtsInput2Hidden,
                                                          wtsHidden2Output)

                allLabels = np.append(allLabels, np.argmax(outputFinal, axis=1)+1)
                # print('dimension of outputHidden: ', outputHidden.shape)
                # print('dimension of outputFinal: ', outputFinal.shape)
                wtsInput2Hidden, wtsHidden2Output = self.backward_pass( trainingBatch, trainingLabel[i:i+batchSize,:],
                                                                  wtsInput2Hidden, outputHidden,
                                                                  wtsHidden2Output, outputFinal)
                i = i + batchSize

                # print('final output')
                # print(np.round(outputFinal,decimals=4))
                # print( np.sum(outputFinal, axis = 1))
                # print(outputFinal.shape)
    #     print('hidden output',outputs)
    #     print(outputs.shape)
        # print('error')
        # print(outputFinal_minus_trainingLabel)

        np.save('i2h', wtsInput2Hidden)
        np.save('h2o', wtsHidden2Output)
        # print('final output', allLabels.shape)
        # print(np.round(outputFinal,decimals=2))
        # print( np.sum(outputFinal, axis = 1))
        # print(allLabels)
        correct = 0
        for label in (allLabels - y):
            if label == 0:
                correct = correct + 1

        print('accuracy is: ',(correct / n)*100,'%')
        return wtsInput2Hidden, wtsHidden2Output

class Test:
    def __init__(self, feat_scal_avg, feat_scal_diff, feat_min, feat_std, wtsInput2Hidden, wtsHidden2Output):
        self.test_network(feat_scal_avg, feat_scal_diff, feat_min, feat_std, wtsInput2Hidden, wtsHidden2Output)
        print('Submission file successfully written.')

    def output_fun( self, X, wtsInput2Hidden, wtsHidden2Output ):
        # wtsInput2Hidden = np.load('i2h.npy')
        # wtsHidden2Output = np.load('h2o.npy')
        outputHidden = ActivationFun_Grad().sigmoid( np.dot( X, wtsInput2Hidden) )
        outputFinal = ActivationFun_Grad().softmax_1d_array( np.dot( outputHidden, wtsHidden2Output ) )
        label = np.argmax(outputFinal) + 1
        return label

    def test_network(self, feat_scal_avg, feat_scal_diff, feat_min, feat_std, wtsInput2Hidden, wtsHidden2Output ):

        with open(out_file,'w') as f:
            print('Id,predicted_class', file = f)
            i = 1
            first = 0
            with open(test_file, 'r') as tst:
                for line in tst:
                    if first == 0:
                        first =1
                        continue
                    X = line.split(',')
                    X[-1] = day2num[X[-1].rstrip('\n')]
                    X[-2] = day2num[X[-2]]
                    X = [1] + X
                    X = np.array(X).astype(float)
                    X = X - feat_min
                    X = X / feat_scal_diff
                    print(i,self.output_fun(X, wtsInput2Hidden, wtsHidden2Output ), file = f,sep=',')
                    i = i+1

def main():
    global nIterations
    nIterations = int(input('enter the number of iterations: '))
    tr =  Train()
    Test( tr.feat_scal_avg, tr.feat_scal_diff, tr.feat_min, tr.feat_std, tr.wtsInput2Hidden, tr.wtsHidden2Output )
if __name__ == '__main__':
    main()
