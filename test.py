'''
Test file for LMS.py
'''
import numpy as np
import math

from matplotlib import pyplot as plt
from scipy.signal import lfilter
from scipy.stats import norm as gaussian

from LMS import LMS, LMS_Normalized

#This method defines a function closure that uses the LMS
#algorithm as a 1 step predictor.
def one_step_pred_setup(F):
  def ff_fb(x):
    F.fb(x - ff_fb.x_hat_prev)
    x_hat = F.ff(x)
    ff_fb.x_hat_prev = x_hat
    return x_hat
  ff_fb.x_hat_prev = 0
  return ff_fb

#--------------------------------------------

def test1():
  '''
  Runs an example of LMS filtering for 1 step prediction on a WSS process.
  '''
  np.random.seed(2718)

  N = 500 #Length of data
  mu = .01 #Step size
  p = 2 #Filter order

  #Filter for generating d(n)
  b = [1.]
  a = [1, -0.1, -0.8]
  sv2 = .25 #Innovations noise variance

  #scale specifies standard deviation sqrt(sv2)
  v = gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  d = lfilter(b, a, v) #Desired process

  #Initialize LMS filter and then
  #Get function closure implementing 1 step prediction
  F = LMS(mu = mu, p = p)
  ff_fb = one_step_pred_setup(F)

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
  plt.title('LMS $1$ Step Prediction, $\mu = %s$, $p = %d$' % (mu, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

def test2():
  '''
  Shows the normalized LMS algorithm on a more complicated process.
  '''
  np.random.seed(314)

  N = 500 #Length of data
  beta = 0.3 #Step size modifier
  p = 4 #Filter order

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
  ff_fb = one_step_pred_setup(F)

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
  plt.title('LMS_Normalized $1$ Step Prediction, '\
            '$\\beta = %s$, $p = %d$' % (beta, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

if __name__ == '__main__':
  test1()
  test2()
