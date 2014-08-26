#!/usr/bin/env python
#
''' Logistic regression with USPS digit data

Trains logistic regression classifier with tr_x.txt and tr_y.txt as training set. 
tr_x is the set of binary 16x16 images (flattened into vectors of size 256)
tr_y is the set of labels valued 1-10 for tr_x data
te_x and te_y are testing data set
'''
import numpy as np
import matplotlib.pyplot as plt

def p_y_giv_x(x, y, w, k):
    ''' Calculates probability of y=k given x and w according to logistic regression 
    '''
    return np.exp((x).dot(w[:,k])) / np.sum(np.exp((x).dot(w)))

def grad_ascent(w, eta, lam, tr_x, tr_y):
    ''' Update step of gradient ascent algorithm 
    '''
    for k in range(w.shape[1] - 1):
        sbt = np.where(tr_y == k, 1., 0.) - np.array([p_y_giv_x(x, y, w, k) for (x, y) in zip(tr_x, tr_y)]) 
        gradient = (sbt).dot(tr_x) - lam * w[:, k]
        w[:, k] = w[:, k] + gradient * eta
    return w

def log_reg_classify(w, tr_x):
    ''' Implements classification given the coefficients w 
    '''
    return np.argmax((tr_x).dot(w), axis = 1)

def get_class_acc(cur_y, y):
    ''' Calculates classification accuracy 
    '''
    return np.sum(np.where(cur_y == y, 1., 0.)) / y.size

if __name__ == '__main__':
    eta = 2e-4
    lam = 100
    K = 10
    d = 256
    w = np.zeros((d, K))
    n_step = 0
    n_step_max = 1000
    convergence = 'not satisfied'
    tr_x = np.loadtxt('tr_x.txt', delimiter = ',')
    tr_y = np.loadtxt('tr_y.txt', delimiter = ',')
    tr_y = tr_y - 1
    te_x = np.loadtxt('te_x.txt', delimiter = ',')
    te_y = np.loadtxt('te_y.txt', delimiter = ',')
    te_y = te_y - 1
    tr_acc = np.empty((n_step_max,))
    te_acc = np.empty((n_step_max,))
    while convergence is 'not satisfied':
        w = grad_ascent(w, eta, lam, tr_x, tr_y)
        cur_tr_y = log_reg_classify(w, tr_x)
        tr_acc[n_step] = get_class_acc(cur_tr_y, tr_y)
        cur_te_y = log_reg_classify(w, te_x)
        te_acc[n_step] = get_class_acc(cur_te_y, te_y)
        n_step += 1
        if n_step == n_step_max:
            convergence = 'satisfied'
    print tr_acc[-1]
    print te_acc[-1]
    time_steps = np.arange(1, n_step + 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    p1 = ax1.plot(time_steps, tr_acc[0 : n_step], lw = 2)
    ax1.grid(True)
    ax1.set_xlabel('Step #')
    ax1.set_ylabel('Training Accuracy')
    ax2 = fig.add_subplot(212)
    p2 = ax2.plot(time_steps, te_acc[0 : n_step], lw = 2)
    ax2.grid(True)
    ax2.set_xlabel('Step #')
    ax2.set_ylabel('Testing Accuracy')
    plt.show()
