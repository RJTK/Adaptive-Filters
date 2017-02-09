'''
Time varying filter
'''

import numba
import numpy as np

def tvfilt(b, a, x, p, q, time = None):
  '''
  Applies a time varying filter specified by functions b and a to x.
  There are p poles and q zeros in the filter.  e.g. for s static filter
  we may have b = [1, 1.1, -.2] in which case q = 2. And, a = [1., 0.2],
  in which case p = 1.  The 'time' input specifies a time vector.  If none,
  we will use integers between 0 and len(x) - 1
  
  The function b(tau, t) is a function of time lag tau and current
  time t specifying the FIR coefficient.
  
  b: [0, 1, ..., q] x [0, 1, ...] --> R

  Similarly for a

  a: [1, ..., p] x [0, 1, ...] --> R

  x is the input to the filter.  Time starts at 0.

  Roughly:

  T = len(x) - 1 #Total time
  y(< 0) = 0 #Initialize the output
  for t = 0, 1, ..., T:
    y(t) = sum_tau{0 ... p}[b(tau, t)*x(t - tau)] + \
      sum_tau{0 ... q}[a(tau, t)*y(t - tau)]

  NOTE OF WARNING: scipy.signal.lfilter specifies a, b in as
  they would appear in the system function H(z).  This means that
  the a coefficients are the negatives of what they would be
  for this function.
  '''
  #Many errors trying to use numba :/
  #@numba.jit(cache = True, nopython = False)
  def tvfilt_numba(b, a, x, p, q, time):
    z = [0]*p #Internal state
    x = [0]*q + x #Extend x to the left
    y = [] #Output vector
    for i, t in enumerate(time):
      #Numba does not support list comprehensions?
      # yt_FIR, yt_IIR = 0, 0
      # for tau in range(q + 1):
      #   yt_FIR += b(tau, t)*x[q + t - tau]
      # for tau in range(p):
      #   yt_IIR += a(tau, t)*z[tau]

      yt_FIR = sum(b(tau, t)*x[q + i - tau] for tau in range(q + 1))
      yt_IIR = sum(a(tau, t)*z[tau] for tau in range(p))
      yt = yt_FIR + yt_IIR
      z = [yt] + z[:-1] #Append left internal state
      y.append(yt) #Append output sample
    return y
  
  x = list(x)
  p = int(p)
  q = int(q)

  if time is None:
    time = range(len(x))

  y = tvfilt_numba(b, a, x, p, q, time)
  y = np.array(y)
  return y
