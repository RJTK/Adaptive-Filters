'''
Test file for LMS.py
'''
import numpy as np
import math

from matplotlib import pyplot as plt
from scipy.signal import lfilter
from scipy.stats import norm as gaussian

from collections import deque #A queue

from LMS import LMS, LMS_Normalized

#This method defines a function closure that uses the LMS
#algorithm as a 1 step predictor.
def one_step_pred_setup(F, m = 0):
  '''
  F should specify an LMS instance and
  m is the mean of the process
  '''
  def ff_fb(x):
    F.fb(x - ff_fb.x_hat_prev)
    x_hat = F.ff(x)
    ff_fb.x_hat_prev = x_hat
    return x_hat
  ff_fb.x_hat_prev = m
  return ff_fb

def n_step_pred_setup(F, n, m = 0):
  '''
  F should specify an LMS instance and
  m is the mean of the process
  '''
  def ff_fb(x):
    F.fb(x - ff_fb.x_hat_prevs.pop())
    x_hat = F.ff(x)
    ff_fb.x_hat_prevs.appendleft(x_hat)
    return x_hat
  ff_fb.x_hat_prevs = deque([m]*n)
  return ff_fb

#--------------------------------------------

def test1():
  '''
  Runs an example of LMS filtering for 1 step prediction on a WSS
  process.  We plot the actual result, the errors, as well as the
  convergence to the "correct" parameters.
  '''
  np.random.seed(2718)

  N = 500 #Length of data
  mu = .02 #Step size
  p = 1 #Filter order

  #Filter for generating d(n)
  b = [1.]
  a = [1, -0.1, -0.8]
  sv2 = .25 #Innovations noise variance

  #scale specifies standard deviation sqrt(sv2)
  v = gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  d = lfilter(b, a, v) #Desired process

  def test1_setup(F, m = 0):
    '''Return the 1step prediction and the coefficients'''
    def ff_fb(x):
      F.fb(x - ff_fb.x_hat_prev)
      x_hat = F.ff(x)
      ff_fb.x_hat_prev = x_hat
      return (x_hat, F.w)
    ff_fb.x_hat_prev = m
    return ff_fb

  #Initialize LMS filter and then
  #Get function closure implementing 1 step prediction
  F = LMS(mu = mu, p = p)
  ff_fb = test1_setup(F)

  #Run it through the filter and get the error
  d_hat, w = zip(*[ff_fb(di) for di in d])
  d_hat = np.array(d_hat)
  w = np.array(w)

  err = (d - d_hat)
  
  plt.subplot(2,1,1)
  plt.plot(range(N), d, linewidth = 2, linestyle = ':',
           label = 'True Process')
  plt.plot(range(N), d_hat, linewidth = 2, label = 'Prediction')
  plt.legend()
  plt.xlabel('$n$')
  plt.ylabel('Process Value')
  plt.title('LMS $1$ Step Prediction, $\mu = %s$, $p = %d$' % (mu, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()

  plt.plot(range(N), w[:,0], linewidth = 2, label = '$w[0]$')
  plt.plot(range(N), w[:,1], linewidth = 2, label = '$w[1]$')
  plt.hlines(-a[1], 0, N, linestyle = '--', label = '$-a[1]$')
  plt.hlines(-a[2], 0, N, linestyle = ':', label = '$-a[2]$')
  plt.legend()
  plt.ylim((-.5, 1))
  plt.xlabel('$n$')
  plt.ylabel('Coefficients')
  plt.title('Convergence to true coefficients')
  plt.show()
  return

def test2():
  '''
  Shows the normalized LMS algorithm on a more complicated process
  doing multi-step step prediction
  '''
  np.random.seed(314)

  N = 5000 #Length of data
  beta = 0.3 #Step size modifier
  p = 5 #Filter order
  num_steps = 2 #Number of steps to predict ahead

  #Filter for generating d(n)
  b = [2., -.3, 0.8, 1.1, 0.4]
  a = [1, -0.1, -0.8]
  sv2 = .5 #Innovations noise variance

  #scale specifies standard deviation sqrt(sv2)
  v = gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  d = lfilter(b, a, v) #Desired process

  #Initialize LMS filter and then
  #Get function closure implementing 1 step prediction
  F = LMS_Normalized(p = p)
  ff_fb = n_step_pred_setup(F, num_steps)

  #Run it through the filter and get the error
  d_hat = np.array([ff_fb(di) for di in d])
  err = (d - d_hat)
  
  plt.subplot(2,1,1)
  plt.plot(range(N), d, linewidth = 2, linestyle = ':',
           label = 'True Process')
  plt.plot(range(N), d_hat, linewidth = 2, label = 'Prediction')
  plt.legend()
  plt.xlabel('$n$')
  plt.ylabel('Process Value')
  plt.title('LMS_Normalized %d Step Prediction, '\
            '$\\beta = %s$, $p = %d$' % (num_steps, beta, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

def test3():
  '''
  1 step prediction of brownian motion
  '''
  from scipy.stats import multivariate_normal as norm
  #Brownian motion kernel
  def K_brownian(tx, ty, sigma2):
    return (sigma2)*np.minimum(tx, ty)

  def sample_gp(t, cov_func):
    '''
    Draws samples from a gaussian process with covariance given by cov_func.
    cov_func should be a function of 2 variables e.g. cov_func(tx, ty).  For
    the x,y coordinates of the matrix.  If the underlying covariance function
    requires more than 2 arguments, then they should be passed via a lambda
    function.
    '''
    tx, ty = np.meshgrid(t, t)
    cov = cov_func(tx, ty)
    return norm.rvs(cov = np.array(cov))

  np.random.seed(4)

  N = 800 #Length of data
  beta = 0.3 #Step size modifier
  p = 3 #Filter order
  num_steps = 2 #Number of steps ahead to predict

  d = sample_gp(range(N), lambda tx, ty: K_brownian(tx, ty, 3))

  #Initialize LMS filter and then
  #Get function closure implementing 1 step prediction
  F = LMS_Normalized(p = p)
  ff_fb = n_step_pred_setup(F, num_steps)

  #Run it through the filter and get the error
  d_hat = np.array([ff_fb(di) for di in d])
  err = (d - d_hat)
  
  plt.subplot(2,1,1)
  plt.plot(range(N), d, linewidth = 2, linestyle = ':',
           label = 'True Process')
  plt.plot(range(N), d_hat, linewidth = 2, label = 'Prediction')
  plt.legend()
  plt.xlabel('$n$')
  plt.ylabel('Process Value')
  plt.title('LMS_Normalized %d Step Prediction, '\
            '$\\beta = %s$, $p = %d$' % (num_steps, beta, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

if __name__ == '__main__':
  test1()

  #TEST 2 AND 3 NEED WORK
  #test2()
  #test3()
