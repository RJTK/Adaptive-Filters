import numpy as np
import scipy as sp

from collections import deque #A queue

class RLS(object):
  '''Basic Recursive Least Squares adaptive filter. Suppose we wish to
  estimate a signal d(n) from a related signal x(n).  (for example, we
  could have noisy measurements x(n) = d(n) + v(n)) We make a linear
  estimate d^(n) (d hat of n).

  d^(n) = sum(w(k)*x(n - k) for k in range(p + 1))
  
  where x(n) denotes the measurements, and w(k) are the coefficients
  of our filter.

  Assuming that we can observe the true value of d(n), at least for
  some n, then we can update our filter coefficents w(k) to attempt to
  more accurately track d(n).  The RLS algorithm minimizes the
  exponentially weighted squared error over the last p samles.  In
  order to update the filter coefficients pass in the error on the
  previous prediction to LMS.fb.

  w(k) <- w(k) + mu*e(n)*x(n - k)^*
  '''
  def __init__(self, p, lmbda = 0.9, delta = 1e-3):
    '''p denotes the order of the filter.  A zero order filter makes a
    prediction of d(n) based only on x(n), and a p order filter uses
    x(n) ... x(n - p).

    The 0 <= lmbda <= 1 paramter specifies the exponential weighting
    of the errors.  As lmbda decreases, the data is "forgotten" more quickly.

    E(n) = sum_i[lmbda**(n - i) * (d(i) - d^(i))**2]

    delta specifies the parameter for the initial covariance matrix
    P(0) = delta*I
    '''
    assert p >= 0, 'Filter order must be non-negative'
    assert (lmbda > 0) and (lmbda <= 1), 'Forgetting factor must be in (0, 1]'
    assert delta > 0, 'Initial covariance must be positive definite'
    self.p = p #Filter order
    self.lmbda = float(lmbda) #Forgetting factor
    delta = float(delta)

    self.Rt = delta * np.eye(p + 1) #Empirical covariance
    self.Rt_inv = (1 / delta) * np.eye(p + 1) #Inverse of Rt
    self.x = deque([0]*(p + 1), maxlen = p + 1) #Saved data vector

    self.w = np.array([0]*(p + 1))
    return

  def ff(self, x_n):
    '''
    Feedforward.  Make a new prediction of d(n) based on the new input
    x(n)
    '''
    self.x.appendleft(x_n) #Update data history
    return sum([wi*xi for (wi, xi) in zip(self.w, self.x)])

  def fb(self, e):
    '''
    Feedback.  Updates the coefficient vector w based on an error
    feedback e.  Note that e(n) = d(n) - d^(n) must be the error on
    the previous prediction.
    '''
    l = (1. / self.lmbda)
    x = np.array(self.x).reshape(self.p + 1, 1) #Make a column vector
    w = self.w.reshape(self.p + 1, 1) #Make a column vector

    #---------------------------
    #This is more stable
    #But, it completely defeats the purpose of updating Rt_inv...
    #self.Rt = self.lmbda*self.Rt + np.dot(x, x.T)
    #g = sp.linalg.solve(self.Rt, x, sym_pos = True, check_finite = False)

    #---------------------------
    u = np.dot(self.Rt_inv, x) #Intermediate value
    g = u / (self.lmbda + np.dot(x.T, u)) #Gain vector
    self.Rt_inv = l*(self.Rt_inv - np.dot(g, u.T))

    self.w = (w + e*g).reshape(self.p + 1) #Update the filter
    self.Rt_inv = l*np.dot((np.eye(self.p + 1) - np.dot(g, x.T)), 
                           self.Rt_inv)

    self.Rt_inv = 0.5*(self.Rt_inv + self.Rt_inv.T) #Ensure symmetry
    return

def RRLS(RLS):
  '''Regularized Recursive Least Squares adaptive filter.

  d^(n) = sum(w(k)*x(n - k) for k in range(p + 1))

  We use the cost function:

  sum_n^t l^(n - t)(d(n) - d^(n))**2 + mu||w_t||_2^2

  Which is just an L2 regularization term added to the regular LS cost
  function.

  See RLS for more details
  '''
  def __init__(self, p, lmbda, delta, mu):
    '''p denotes the order of the filter.  A zero order filter makes a
    prediction of d(n) based only on x(n), and a p order filter uses
    x(n) ... x(n - p).

    The 0 <= lmbda <= 1 paramter specifies the exponential weighting
    of the errors.  As lmbda decreases, the data is "forgotten" more quickly.

    delta specifies the parameter for the initial covariance
    matrix P(0) = delta*I

    mu is the L2 regularization parameter
    '''
    raise NotImplementedError
    RLS.__init__(self, p, lmbda, delta)
    assert mu >= 0, 'we must have mu >= 0'
    self.mu = mu
    del self.Rt_inv #We don't use this
    return

  def fb(self, e):
    '''
    Feedback.  Updates the coefficient vector w based on an error
    feedback e.  Note that e(n) = d(n) - d^(n) must be the error on
    the previous prediction.
    '''
    return
