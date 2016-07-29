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

def system_identification_setup(F):
  '''
  F should specify an LMS instance.  We return
  the filter parameters of the LMS algorithm w, which
  can be used to identify a model of a system.
  '''
  def ff_fb(x):
    F.fb(x - ff_fb.x_hat_prev)
    x_hat = F.ff(x)
    ff_fb.x_hat_prev = x_hat
    return F.w
  ff_fb.x_hat_prev = 0
  return ff_fb

def one_step_pred_setup(F):
  '''
  F should specify an LMS instance.  This function uses
  LMS to make a 1 step prediction of a process.  This
  is basically a method to track the process over time.

  Note that there is an implicit prediction of 0 for x(0)
  '''
  def ff_fb(x):
    F.fb(x - ff_fb.x_hat_prev)
    x_hat = F.ff(x)
    ff_fb.x_hat_prev = x_hat
    return x_hat
  ff_fb.x_hat_prev = 0
  return ff_fb

def equalizer_setup(F, rx_delay = 0):
  '''
  Sets up a function to use the LMS instane F as an adaptive
  equalizer.  The argument x to ff_fb is the filter's observation, and
  d is the true value of what it should estimate.  Hence if we have x
  = Hd + v where Hd is a filtered data sequence and v is noise, then
  passing in x and d to ff_fb will train an equalizer for H.

  If d is None, we use decision feedback.

  The rx_delay parameter specifies the delay that will be added at the
  receiver.  This is to offset the delay of the channel.  It should be
  long enough that all the "useful" information for predicting d(n -
  rx_delay) from the input sequence x has made it's way into the
  filter We then delay the d inputs by this amount so that the filter
  does not need to be non-causal.

  Note that the filter must have enough taps to accomodate this delay.
  '''
  assert F.p >= rx_delay, 'Filter does not have enough taps'
  def ff_fb(x, d = None):
    d_hat = F.ff(x)
    if d: #Training mode
      ff_fb.D.appendleft(d)
    else:
      ff_fb.D.appendleft(0)


    if ff_fb.D[-1]: #Check for training data
      F.fb(ff_fb.D.pop() - d_hat)
    else: #Else, decision directed feedback
      d = 1 if d_hat >= 0 else -1
      F.fb(d - d_hat)

    return d

  ff_fb.D = deque([0]*(rx_delay + 1),
                  maxlen = rx_delay + 1)
  return ff_fb

#--------------------------------------------

def system_identification():
  '''
  Runs an example of LMS filtering for 1 step prediction on a WSS
  process.  We plot the actual result, the errors, as well as the
  convergence to the "correct" parameters.  This is essentially
  doint system identification.
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

  #Initialize LMS filter and then
  F = LMS(mu = mu, p = p)
  ff_fb = system_identification_setup(F)

  #Run it through the filter and get the error
  #Pay attention to the offsets.  d_hat[0] is a prediction of d[1].
  #We implicitly predict d[0] = 0
  w = np.array([ff_fb(di) for di in d])
  w = np.array(w)
  
  plt.plot(range(N), w[:,0], linewidth = 2, label = '$w[0]$')
  plt.plot(range(N), w[:,1], linewidth = 2, label = '$w[1]$')
  plt.hlines(-a[1], 0, N, linestyle = '--', label = '$-a[1]$')
  plt.hlines(-a[2], 0, N, linestyle = ':', label = '$-a[2]$')
  plt.legend()
  plt.ylim((-.5, 1))
  plt.xlabel('$n$')
  plt.ylabel('$w$')
  plt.title('System Identification')
  plt.show()
  return

def tracking_example1():
  '''
  Shows the LMS algorithm tracking a time varying process.
  '''
  np.random.seed(314)

  N = 500 #Length of data
  beta = 0.4 #Step size modifier
  p = 6 #Filter order

  #Filter for generating d(n)
  b = [1, -0.5, .3]
  a = [1, 0.2, 0.16, -0.21, -0.0225]
  sv2 = .25 #Innovations noise variance

  #Track a time varying process
  t = np.linspace(0, 1, N)
  f = 2
  v = 4*np.sin(2*np.pi*f*t) + \
      gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  d = lfilter(b, a, v) #Desired process

  #Initialize LMS filter and then
  #Get function closure implementing 1 step prediction
  F = LMS_Normalized(p = p, beta = beta)
  ff_fb = one_step_pred_setup(F)

  #Run it through the filter and get the error
  d_hat = np.array([0] + [ff_fb(di) for di in d])[:-1]
  err = (d - d_hat)
  
  plt.subplot(2,1,1)
  plt.plot(range(N), d, linewidth = 2, linestyle = ':',
           label = 'True Process')
  plt.plot(range(N), d_hat, linewidth = 2, label = 'Prediction')
  plt.legend()
  plt.xlabel('$n$')
  plt.ylabel('Process Value')
  plt.title('LMS_Normalized tracking a process' \
            '$\\beta = %s$, $p = %d$' % (beta, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

def tracking_example2():
  '''
  Tracking a brownian motion process
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
  beta = 0.4 #Step size modifier
  p = 6 #Filter order
  sd2 = 2

  d = sample_gp(range(N), lambda tx, ty: K_brownian(tx, ty, sd2))

  #Initialize LMS filter and then
  #Get function closure implementing 1 step prediction
  F = LMS_Normalized(p = p)
  ff_fb = one_step_pred_setup(F)

  #Run it through the filter and get the error
  d_hat = np.array([0] + [ff_fb(di) for di in d])[:-1]
  err = (d - d_hat)
  
  plt.subplot(2,1,1)
  plt.plot(range(N), d, linewidth = 2, linestyle = ':',
           label = 'True Process')
  plt.plot(range(N), d_hat, linewidth = 2, label = 'Prediction')
  plt.legend()
  plt.xlabel('$n$')
  plt.ylabel('Process Value')
  plt.title('LMS_Normalized tracking a process, '\
            '$\\beta = %s$, $p = %d$' % (beta, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

def channel_equalization():
  '''
  Shows an example of channel equalization.  We train the LMS
  algorithm with an aprior known sequence, then use decision feedback
  equalization.
  '''
  from scipy.stats import bernoulli
  np.random.seed(13)

  #Channel impulse response
  h = [0, .05, 0.15, 0.5, 0.15, .05]
  a = [1.]

  rx_delay = 10
  beta = 0.5
  p = 15 #Filter order

  N = 1000 #Length of all data
  t_N = N/8 #Length of training sequence
  d_N = N - t_N #Length of "real data" sequence

  sv2 = 0.01 #noise variance

  k = -1 + 2*bernoulli.rvs(0.5, size = N) #All data
  v = gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Noise
  x = lfilter(h, a, k) + v #Signal after the channel

  t = k[:t_N] #Training sequence
  rx_t = x[:t_N] #Received training sequence
  d = k[t_N:] #Data sequence
  rx_d = x[t_N:] #Received data sequence

  plt.plot(range(t_N), rx_t, label = 'noisy rx')
  plt.plot(range(t_N), rx_t - v[:t_N], label = 'non noisy rx')
  plt.plot(range(t_N), t, label = 'training sequence')
  plt.ylim((-1.5, 1.5))
  plt.legend()
  plt.title('Training Phase')
  plt.xlabel('$n$')
  plt.show()

  #Setup an equalization LMS filter
  F = LMS_Normalized(beta = 0.6, p = p)
  ff_fb = equalizer_setup(F, rx_delay)
  #Train the equalizer
  W = np.array([ff_fb(rx_ti, ti) for (rx_ti, ti) in zip(rx_t, t)])

  eq_h = np.convolve(F.w, h) #Equalized channel response

  plt.subplot(2,1,1)
  plt.stem(range(len(h)), h)
  plt.title('Channel Response')
  plt.xlim((0, len(h) + 2))
  plt.xlabel('$n$')
  plt.ylabel('$h[n]$')

  plt.subplot(2,1,2)
  plt.stem(range(len(eq_h)), eq_h)
  plt.title('Equalized Response')
  plt.xlabel('$n$')
  plt.ylabel('$(h*w)[n]$')

  plt.show()

  #now use the equalizer with decision directed feedback
  d_hat = np.array([ff_fb(rx_di) for rx_di in rx_d])

  #Need to synchronize
  err = d[:-rx_delay] != d_hat[rx_delay:]
  plt.stem(range(d_N - rx_delay), err)
  plt.title('Errors')
  plt.xlabel('$n$')
  plt.ylabel('$e[n]$')
  plt.show()

  return

if __name__ == '__main__':
  #system_identification()
  #tracking_example1()
  #tracking_example2()
  channel_equalization()
