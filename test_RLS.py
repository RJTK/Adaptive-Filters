'''
Test file for RLS.py
'''
import numpy as np
import math

from matplotlib import pyplot as plt
from scipy.signal import lfilter
from scipy.stats import norm as gaussian
from scipy.stats import pareto
from collections import deque

from tvfilt import tvfilt
from RLS import RLS
from setup_helpers import \
  system_identification_setup,\
  one_step_pred_setup,\
  equalizer_setup

def system_identification1():
  '''
  Runs an example of RLS filtering for 1 step prediction on a WSS
  process.  We plot the actual result, the errors, as well as the
  convergence to the "correct" parameters.  This is essentially
  doint system identification.
  '''
  np.random.seed(2718)

  N = 5000 #Length of data
  lmbda = 0.99 #Forgetting factor
  p = 2 #Filter order

  #Filter for generating d(n)
  b = [1.]
  a = [1, -0.1, -0.8, 0.2]
  sv2 = .25 #Innovations noise variance

  #scale specifies standard deviation sqrt(sv2)
  v = gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  d = lfilter(b, a, v) #Desired process

  #Initialize RLS filter and then
  F = RLS(p = p, lmbda = lmbda) #Vanilla
  ff_fb = system_identification_setup(F)

  #Run it through the filter and get the error
  #Pay attention to the offsets.  d_hat[0] is a prediction of d[1].
  #We implicitly predict d[0] = 0
  w = np.array([ff_fb(di) for di in d])
  w = np.array(w)
  
  plt.plot(range(N), w[:,0], linewidth = 2, label = '$w[0]$')
  plt.plot(range(N), w[:,1], linewidth = 2, label = '$w[1]$')
  plt.plot(range(N), w[:,2], linewidth = 2, label = '$w[2]$')
  plt.hlines(-a[1], 0, N, linestyle = ':', label = '$-a[1]$')
  plt.hlines(-a[2], 0, N, linestyle = ':', label = '$-a[2]$')
  plt.hlines(-a[3], 0, N, linestyle = ':', label = '$-a[3]$')
  plt.legend()
  plt.ylim((-.5, 1))
  plt.xlabel('$n$')
  plt.ylabel('$w$')
  plt.title('System Identification')
  plt.show()
  return

def system_identification2():
  '''
  Runs an example of Sparse RLS filtering for 1 step prediction on a
  WSS process.  We plot the actual result, the errors, as well as the
  convergence to the "correct" parameters.  This is essentially doing
  system identification.

  The point of this is to compare the sparse vs non sparse RLS
  '''
  np.random.seed(2718)

  N = 5000 #Length of data
  lmbda = 0.99 #Forgetting factor
  p = 9 #Filter order

  #Filter for generating d(n)
  b = [1.]
  a = [1, -0.1, 0., 0., 0.3, 0., 0.2, 0., 0., 0., -0.3]
  sv2 = .25 #Innovations noise variance

  #scale specifies standard deviation sqrt(sv2)
  v = gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  d = lfilter(b, a, v) #Desired process

  #Initialize RLS filter and then
  F = RLS(p = p, lmbda = lmbda)
  ff_fb = system_identification_setup(F)

  #Run it through the filter and get the error
  #Pay attention to the offsets.  d_hat[0] is a prediction of d[1].
  #We implicitly predict d[0] = 0
  w = np.array([ff_fb(di) for di in d])
  w = np.array(w)
  
  for i in range(p):
    plt.plot(range(N), w[:, i], linewidth = 2)
    plt.hlines(-a[i + 1], 0, N, linestyle = ':')

  plt.ylim((-.5, 1))
  plt.xlabel('$n$')
  plt.ylabel('$w$')
  plt.title('System Identification')
  plt.show()
  return

def tracking_example1():
  '''
  Shows the RLS algorithm tracking a time varying process.
  '''
  np.random.seed(314)

  N = 500 #Length of data
  lmbda = 0.99 #Forgetting factor
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

  #Initialize RLS filter and then
  #Get function closure implementing 1 step prediction
  F = RLS(p = p, lmbda = lmbda)
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
  plt.title('RLS tracking a process ' \
            '$\\lambda = %s$, $p = %d$' % (lmbda, p))

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
  lmbda = 0.99 #Forgetting factor
  p = 6 #Filter order
  sd2 = 2

  d = sample_gp(range(N), lambda tx, ty: K_brownian(tx, ty, sd2))

  #Initialize RLS filter and then
  #Get function closure implementing 1 step prediction
  F = RLS(p = p, lmbda = lmbda)
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
  plt.title('RLS tracking a process, '\
            '$\\lambda = %s$, $p = %d$' % (lmbda, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

def tracking_example3():
  '''
  Shows the RLS algorithm tracking a process with time varying statistics.
  '''
  np.random.seed(314)

  N = 5000 #Length of data
  lmbda = .98 #Forgetting factor
  p_RLS = 4 #RLS Filter order

  #Filter for generating d(n)
  p1_f = 1.35 #Frequency of pole 1
  p2_fA = 1.1 #Frequency of pole 2 amplitude
  p2_fP = 1.5 #Frequency of pole 2 phase
  def A(tau, t):
    a = [1, -0.1, -0.8, 0.2]
    p1 = 0.25 + 0.75*np.sin(2*np.pi*t*p1_f) #pole 1 position
    p2A = 0.15 + abs(0.75*np.sin(2*np.pi*t*p2_fA)) #pole 2 amplitude
    p2P = np.exp(2*np.pi*1j*t*p2_fP) #pole 2 phase
    p2 = p2A*p2P
    p3 = p2.conj() #pole 3
    a = np.poly([p1, p2, p3])
    return -a[tau + 1]

  b = [1., -.2, 0.8]

  B = lambda tau, t : b[tau]
  p = 3
  q = 1

  sv2 = 0.25 #Scale paremeter

  #Track a time varying process
  t = np.linspace(0, 1, N)
  f = 2
#  v = 4*np.sin(2*np.pi*f*t) + \
#      gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  v = gaussian.rvs(size = N, scale = math.sqrt(sv2))
#  v = pareto.rvs(beta, size = N, scale = math.sqrt(sv2))
#  d = lfilter(b, a, v) #Desired process
  d = tvfilt(B, A, v, p, q, t)

  #Initialize RLS filter and then
  #Get function closure implementing 1 step prediction
  F = RLS(p = p_RLS, lmbda = lmbda)
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
  plt.title('RLS tracking a process ' \
            '$\\lambda = %s$, $p = %d$' % (lmbda, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2)
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()

  F = RLS(p = p_RLS, lmbda = lmbda)
  ff_fb = system_identification_setup(F)
  w_hat = np.array([ff_fb(di) for di in d])

  a1 = np.array([A(0, ti) for ti in t])
  a2 = np.array([A(1, ti) for ti in t])
  a3 = np.array([A(2, ti) for ti in t])
  plt.plot(t, a1, linestyle = '--', label = '$a[1]$')
  plt.plot(t, a2, linestyle = '--', label = '$a[2]$')
  plt.plot(t, a3, linestyle = '--', label = '$a[3]$')
  plt.plot(t, w_hat[:, 0], label = '$w[0]$')
  plt.plot(t, w_hat[:, 1], label = '$w[1]$')
  plt.plot(t, w_hat[:, 2], label = '$w[2]$')
  plt.plot(t, w_hat[:, 3], label = '$w[3]$')
  plt.legend()
  plt.title('RLS Tracking ARMA Process, misspecified $p$')
  plt.show()
  return

def tracking_example4():
  '''
  Shows the RLS algorithm tracking a process with fat tailed noise.
  We compare performance with and without a clamped input range.

  Obviously, simply clipping the input range is a pretty naive method
  for dealing with fat tailed noise.
  '''
  np.random.seed(2718)

  N = 1000 #Length of data
  lmbda = 0.98 #Forgetting factor
  p = 6 #Filter order

  #Filter for generating d(n)
  b = [1, -0.5, .3]
  a = [1, 0.2, 0.16, -0.21, -0.0225]
  sv2 = .25 #Innovations noise variance
  beta = 1.25
  psv = 0.025

  t = np.linspace(0, 1, N)
  f = 2
  v = 2*np.sin(2*np.pi*f*t) + \
      gaussian.rvs(size = N, scale = math.sqrt(sv2)) #Innovations
  d = lfilter(b, a, v) #Desired process
  d = d + pareto.rvs(beta, size = N, scale = math.sqrt(psv)) #fat tailed noise

  def clamp(x, M, m):
    return M*(x >= M) + m*(x <= m) + x*(x < M and x > m)

  M = 4.
  m = -4.
  d_clamp = np.array([clamp(di, M, m) for di in d])

  #Initialize RLS filter and then
  #Get function closure implementing 1 step prediction

  #-------CLAMPED INPUT-----------
  F = RLS(p = p, lmbda = lmbda)
  ff_fb = one_step_pred_setup(F)

  d_hat_clamp = np.array([0] + [ff_fb(di) for di in d_clamp])[:-1]
  err_clamp = (d - d_hat_clamp)
  MSE_avg_clamp = np.average(abs(err_clamp)**2)

  #--------UNCLAMPED INPUT---------
  F = RLS(p = p, lmbda = lmbda)
  ff_fb = one_step_pred_setup(F)

  #Run it through the filter and get the error
  d_hat = np.array([0] + [ff_fb(di) for di in d])[:-1]
  err = (d - d_hat)
  MSE_avg = np.average(abs(err)**2)

  plt.subplot(2,1,1)
  plt.plot(range(N), d, linewidth = 1, linestyle = ':',
           label = 'True Process')
  plt.plot(range(N), d_clamp, linewidth = 1, linestyle = '--',
           label = 'Clamped Process')
  plt.plot(range(N), d_hat, linewidth = 1, label = 'Prediction')
  plt.plot(range(N), d_hat_clamp, linewidth = 1, label = 'Prediction (clamped)')
  plt.legend()
  plt.xlabel('$n$')
  plt.ylabel('Process Value')
  plt.title('RLS tracking a process ' \
            '$\\lambda = %s$, $p = %d$' % (lmbda, p))

  plt.subplot(2,1,2)
  plt.plot(range(N), err, linewidth = 2, label = 'err')
  plt.plot(range(N), err_clamp, linewidth = 2, label = 'err (clamped)')
  plt.hlines(MSE_avg, 0, N, linestyle = '--', label = 'MSE', linewidth = 3,
             color = 'r')
  plt.hlines(MSE_avg_clamp, 0, N, linestyle = '--', label = 'MSE (clamped)',
             linewidth = 3, color = 'y')
  plt.legend()
  plt.xlabel('$n$')
  plt.ylabel('Error')
  plt.title('Prediction Error')

  plt.show()
  return

def channel_equalization():
  '''
  Shows an example of channel equalization.  We train the RLS
  algorithm with an aprior known sequence, then use decision feedback
  equalization.
  '''
  from scipy.stats import bernoulli
  np.random.seed(13)

  #Channel impulse response
  h = [0, .05, 0.15, 0.5, 0.15, .05]
  a = [1.]

  rx_delay = 10
  lmbda = 0.99 #Forgetting factor
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

  #Setup an equalization RLS filter
  F = RLS(p = p, lmbda = lmbda)
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
  system_identification1()
  system_identification2()
  tracking_example1()
  tracking_example2()
  tracking_example3()
  tracking_example4()
  channel_equalization()
