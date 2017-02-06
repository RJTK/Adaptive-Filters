from collections import deque #A queue

def sgn(x):
  '''
  sgn function.

  sgn(x) = {x / |x| if x != 0, 0 if x = 0
  '''
  return (1 - 2*(x < 0))*(x != 0)

class LMS(object):
  '''
  Basic Least Mean Squares adaptive filter.  Suppose we wish to
  estimate a signal d(n) from a related signal x(n).  (for example, we
  could have noisy measurements x(n) = d(n) + v(n)) We make a linear
  estimate d^(n) (d hat of n).

  d^(n) = sum(w(k)*x(n - k) for k in range(p + 1))

  where x(n) denotes the measurements, and w(k) are the coefficients
  of our filter.

  Assuming that we can observe the true value of d(n), at least for
  some n, then we can update our filter coefficents w(k) to attempt to
  more accurately track d(n).  The LMS algorithm is an approximation
  of a steepest descent adaptive algorithm, very similar to stochastic
  gradient descent.  In order to update the filter coefficients pass
  in the error on the previous prediction to LMS.

  w(k) <- w(k) + mu*e(n)*x(n - k)^*

  '''
  def __init__(self, p, mu = 0.001, w0 = None):
    '''
    p denotes the order of the filter.  A zero order filter makes a
    prediction of d(n) based only on x(n).  A first order filter uses
    both x(n) and x(n - 1).  In general, x(n) ... x(n - p) is used to
    estimate d(n).  p should be moderate, the error does NOT decrease
    monotonically with p.  The error will be roughly parabolic-ish in p.

    The mu paramter specifies the descent rate.  Small values are
    safe, but larger values can provide faster tracking if the
    signal's statistics are rapidly time varying.  If mu is too large
    the filter will be unstable.

    w0 specifies an initial set of paramter weaights.  0 is default
    '''
    assert mu > 0, 'Step size must be positive'
    assert p >= 0, 'Filter order must be non-negative'

    self.mu = mu
    self.p = p 

    #The saved data queue.  Front is x[0]
    self.x = deque([0]*(p + 1), maxlen = p + 1)

    if w0:
      assert len(w0) == p + 1, 'w0 must be same order as filter'
      self.w = w0
    else:
      self.w = [0]*(p + 1)
    
    return

  def ff(self, x_n):
    '''
    Feedforward.  Make a new prediction of d(n) based on the new
    input x(n)
    '''
    #Update the data history
    self.x.appendleft(x_n) #oldest data automatically removed
    return sum([wi*xi for (wi, xi) in zip(self.w, self.x)])

  def fb(self, e):
    '''
    Feedback.  Updates the coefficient vector w based on an error
    feedback e.  Note that e(n) = d(n) - d^(n) must be the error on
    the previous prediction.
    '''
    u = self.mu*e
    self.w = [wi + u*xi.conjugate() for (wi, xi) in zip(self.w, self.x)]
    return

class LMS_Normalized(LMS):
  '''
  Normalized LMS algorithm.  This method uses a dynamically varying
  step size.  Specifically we use mu(n) = 2*beta / (eps + ||x(n)||^2)
  beta must be in (0, 1) for the filter to be stable.  The epsilon
  parameter ensures that the coefficients don't blow up if ||x(n)||^2
  is small
  '''
  def __init__(self, p, beta = 0.2, w0 = None, eps = 1e-3):
    '''
    p is the filter order (See LMS.__init__)

    beta is the stepsize parameter.  We need beta in (0, 1)
    
    epsilon is a non-negative number ensuring that w doesn't blow up
    if ||x(n)|| is small.
    '''
    assert beta > 0 and beta < 1, 'Beta must be in (0, 1)'
    assert eps >= 0, 'epsilon must be non-negative'
    LMS.__init__(self, p = p, w0 = w0)
    self.beta = beta
    self.eps = eps
    return
    
  def fb(self, e):
    '''
    Feedback.  Updates the coefficient vector w based on an error
    feedback e.  Note that e(n) = d(n) - d^(n) must be the error on
    the previous output.
    '''
    x_norm_sqrd = sum([xi*xi.conjugate() for xi in self.x])
    u = self.beta*e/(x_norm_sqrd + self.eps)
    self.w = [wi + u*xi.conjugate() for (wi, xi) in zip(self.w, self.x)]
    return

class LMS_ZA(LMS):
  '''
  A sparse LMS algorithm, "Zero-Attracting LMS".  We simply add a
  g||w||_1 penalty term to the cost function.  This yields essentially
  just stochastic subgradient descent.

  Chen, Yilun, Yuantao Gu, and Alfred
  O. Hero. "Sparse LMS for system identification." Acoustics, Speech
  and Signal Processing, 2009. ICASSP 2009. IEEE International
  Conference on. IEEE, 2009.

  '''
  def __init__(self, p, mu, g, w0 = None):
    '''
    p is the filter order, and mu is the descent rate (See LMS.__init__)
    
    The 'g' parameter (for gamma in Chen et. al.) specifies the size
    of the L1 penalty.  Larger values of g will induce greater
    sparsity.
    '''
    LMS.__init__(self, p = p, mu = mu, w0 = w0)
    assert g >= 0, 'g must be non-negative'
    self.g = float(g)
    return
    
  def fb(self, e):
    '''
    Feedback.  Updates the coefficient vector based on an error
    feedback e.  Note that e(n) = d(n) - d^(n) must be the error on
    the previous output.
    '''
    u = self.mu*e
    rho = self.mu*self.g
    self.w = [wi - rho*sgn(wi) + u*xi.conjugate() for (wi, xi) in
              zip(self.w, self.x)]
    return

class LMS_RZA(LMS):
  '''
  A reweighted sparse LMS algorithm, "Reweighted Zero-Attracting
  LMS".  We add a g*sum_i[log(1 + wi/eps)] penalty term to the cost
  function.  This will reduce the bias of the LMS_ZA
  algorithm. subgradient descent.

  Chen, Yilun, Yuantao Gu, and Alfred O. Hero. "Sparse LMS for system
  identification." Acoustics, Speech and Signal Processing,
  2009. ICASSP 2009. IEEE International Conference on. IEEE, 2009.

  '''
  def __init__(self, p, mu, g, eps, w0 = None):
    '''
    p is the filter order, and mu is the descent rate (See LMS.__init__)
    
    The 'g' parameter (for gamma in Chen et. al.) specifies the size
    of the penalty.  Larger values of g will induce greater
    sparsity.

    The eps term is the bias reduction mechanism.  For taps with size
    comparable to 1 / eps there is little shrinkage, and for taps with
    size much less than 1 / eps the shrinkage is greater.

    '''
    LMS.__init__(self, p = p, mu = mu, w0 = w0)
    assert g >= 0, 'g must be non-negative'
    assert eps >= 0, 'eps must be non-negative'
    self.g = float(g)
    self.eps = float(eps)
    return
    
  def fb(self, e):
    '''
    Feedback.  Updates the coefficient vector based on an error
    feedback e.  Note that e(n) = d(n) - d^(n) must be the error on
    the previous output.
    '''
    u = self.mu*e
    rho = self.mu*self.g / self.eps
    f = lambda wi: sgn(wi) / (1 + abs(wi) / self.eps)
    self.w = [wi - rho*f(wi) + u*xi.conjugate() for (wi, xi) in
              zip(self.w, self.x)]
    return
