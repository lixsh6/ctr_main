from __future__ import print_function
import time

import numpy as np
import numpy.linalg
import scipy.optimize
from six.moves import xrange

from simplex_projection import euclidean_proj_simplex
from formatted_logger import formatted_logger

logger = formatted_logger('CollaborativeTopicModel', 'info')

e = 1e-10
error_diff = 10
pre_rmse = 1E5
Done = 0

class CollaborativeTopicModel:  
    """
    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.

    Attributes
    ----------
    n_item: int
        number of items
    n_user: int
        number of users
    R: ndarray, shape (n_user, n_item)
        user x item rating matrix
    """

    def __init__(self, n_topic, n_voca, n_user, n_item, doc_ids, doc_cnt, ratings,dataName = 'imdb'):
        self.lambda_u = 0.01
        self.lambda_v = 0.01
        self.alpha = 1
        self.eta = 0.01
        self.a = 1 
        self.b = 0.01

        self.n_topic = n_topic
        self.n_voca = n_voca
        self.n_user = n_user
        self.n_item = n_item

        self.testUser = []
        self.testItem = []
        self.testAns = []

        self.dataName = dataName
        # U = user_topic matrix, U x K
        #self.U = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_u),
        #                                       size=self.n_user)
        # V = item(doc)_topic matrix, V x K
        #self.V = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_u),
        #                                       size=self.n_item)
        
        self.U = np.random.random([n_user, n_topic])
        self.V = np.random.random([n_item, n_topic])
        self.theta = np.random.random([n_item, n_topic])
        self.theta = self.theta / self.theta.sum(1)[:, np.newaxis]  # normalize
        self.beta = np.random.random([n_voca, n_topic])
        self.beta = self.beta / self.beta.sum(0)  # normalize

        self.doc_ids = doc_ids
        self.doc_cnt = doc_cnt

        self.C = np.zeros([n_user, n_item]) + self.b
        self.R = np.zeros([n_user, n_item])  # user_size x item_size

        self.verbose = True

        #di is item
        #ratings : items * users
        '''
        for di in xrange(len(ratings)):
            rate = ratings[di]
            for user in rate:
                self.C[user, di] += self.a - self.b
                self.R[user, di] = 1
        '''
        for (u,v,r) in ratings:
            self.C[u, v] += self.a - self.b
            self.R[u, v] = r

        self.phi_sum = np.zeros([n_voca, n_topic]) + self.eta

    #def fit(self, doc_ids, doc_cnt, rating_matrix, max_iter=100):
    def fit(self, max_iter=100,testFileName = None):
        old_err = 0
        print ('Start training...')
        print (self.verbose)
        for iteration in xrange(max_iter):
            tic = time.clock()
            self.do_e_step()
            self.do_m_step()
            err = self.sqr_error()
            #print ('Iteration:' + str(iteration)+ ' Error: ' + str(err))
            if self.verbose:
                logger.info('[ITER] %3d,\tElapsed time:%.2f,\tReconstruction error:%.3f', iteration,
                            time.clock() - tic, err)
            self.predictInLoop()
            if abs(old_err - err) < error_diff:
                break

    # reconstructing matrix for prediction
    def predict_item(self):
        return np.dot(self.U, self.V.T)

    # reconstruction error
    def sqr_error(self):
        err = (self.R - self.predict_item()) ** 2
        err = err.sum()

        return err

    def do_e_step(self):
        self.update_u()
        self.update_v()
        self.update_theta()

    def update_theta(self):
        def func(x, v, phi, beta, lambda_v):
            return 0.5 * lambda_v * np.dot((v - x).T, v - x) - np.sum(np.sum(phi * (np.log(x * beta + 1e-4) - np.log(phi + 1e-4))))
       
        for vi in xrange(self.n_item):
            W = np.array(self.doc_ids[vi])
            word_beta = self.beta[W, :]
            phi = self.theta[vi, :] * word_beta + e  # W x K
            phi = phi / phi.sum(1)[:, np.newaxis]
            result = scipy.optimize.minimize(func, self.theta[vi, :], method='nelder-mead',
                                             args=(self.V[vi, :], phi, word_beta, self.lambda_v))
            self.theta[vi, :] = euclidean_proj_simplex(result.x)            
            self.phi_sum[W, :] += np.array(self.doc_cnt[vi])[:, np.newaxis] * phi

    def update_u(self):
        for ui in xrange(self.n_user):
            left = np.dot(self.V.T * self.C[ui, :], self.V) + self.lambda_u * np.identity(self.n_topic)
            self.U[ui, :] = numpy.linalg.solve(left, np.dot(self.V.T * self.C[ui, :], self.R[ui, :]))

    def update_v(self):
        for vi in xrange(self.n_item):
            left = np.dot(self.U.T * self.C[:, vi], self.U) + self.lambda_v * np.identity(self.n_topic)
            self.V[vi, :] = numpy.linalg.solve(left, np.dot(self.U.T * self.C[:, vi],
                                                            self.R[:, vi]) + self.lambda_v * self.theta[vi, :])

    def do_m_step(self):
        self.beta = self.phi_sum / self.phi_sum.sum(0)
        self.phi_sum = np.zeros([self.n_voca, self.n_topic]) + self.eta

    def importTestFile(self,fileName):
        fin = open(fileName,'r')
        for row in fin:
            elements = row.split('\t\t')
            elements = map(eval,elements)
            self.testUser.append(elements[0])
            self.testItem.append(elements[1])
            self.testAns.append(elements[2])

        fin.close()

    def predictInLoop(self):
        size = len(self.testAns)
        RMSE = 0.0
        MAE = 0.0
        for i in xrange(size):
            pred = self.U[self.testUser[i],:].dot(self.V[self.testItem[i],:].T)
            RMSE += (pred - self.testAns[i]) ** 2
            MAE += abs(pred - self.testAns[i])
        RMSE = np.sqrt(RMSE / size)
        MAE = MAE / size
   
        if RMSE > pre_rmse and Done is 0:
            Done = 1
            self.saveMatrix('./log/'+self.dataName+'_matrix.txt')
        pre_rmse = RMSE
        print ('RMSE: ', RMSE,' MAE: ',MAE)
        logger.info('RMSE Error:%.3f MAE Error:%.3f', RMSE,MAE)

    def saveMatrix(self,fileName):
        result = self.U.dot(self.V)
        np.savetxt(fileName,result)

    def test(self,fileName):
        fin = open(filename,'r')
        RMSE = 0.0
        count = 0
        for row in fin:
            elements = row.split('\t\t')
            elements = map(eval,elements)

            pred = self.U[elements[0],:].dot(self.V[elements[1],:].T)
            RMSE += (pred - elements[2]) ** 2

        err = np.sqrt(RMSE / count)
        print ('RMSE: ', err)
        logger.info('RMSE Error:%.3f', err)
        fin.close()

