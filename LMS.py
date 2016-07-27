from collections import deque #A queue
from scipy import linalg as spla
from matplotlib import pyplot as plt
import numpy as np
import math

from C7_1 import fir_wiener

class LMS(object):
  '''
  '''
  def __init__(self, mu = 0.001, p = 1, w0 = None):
    '''
    '''
    assert mu > 0, 'Step size cannot be negative!'
    assert p >= 0, 'Cannot have a negative order filter!'
    #Initialize learning rate, filter order, and initial taps
    self.mu = mu
    self.p = p 

    #The saved data queue.  Front is x[0]
    self.x = deque([0]*(p + 1))

    if w0:
      assert len(w0) == p + 1, 'w0 must be same order as filter'
      self.w = w0
    else:
      self.w = [0]*(p + 1)
    
    return

  def ff(self, x_n):
    '''
    Feedforward.  Make a new prediction of d(n) based on the new
    input x_n
    '''
    #Update the data history
    self.x.pop() #Remove the oldest data element
    self.x.appendleft(x_n)
    return sum([w*x for (w, x) in zip(self.w, self.x)])

  def fb(self, e):
    '''
    Feedback.  Updates the coefficient vector w based on an error
    feedback e.  Note that e = d - d_hat
    '''
    u = self.mu*e
    self.w = [w + u*x.conjugate() for (w, x) in zip(self.w, self.x)]
    return

class LMS_normalized(LMS):
  '''
  Normalized LMS algorithm
  '''
  def __init__(self, beta = 0.5, p = 1, w0 = 0, eps = 0):
    '''
    '''
    assert beta > 0 and beta < 1, 'Beta must be in (0, 1)'
    LMS.__init__(0, p, w0)
    self.beta = beta
    self.eps = eps
    return
    
  def fb(self, e):
    '''
    Feedback.  Similar to LMS.fb, but mu = beta / ||x||^2.
    e = d - d_hat
    '''
    x_norm = sum([x*x.conjugate() for x in self.x]) + self.eps
    u = self.beta*e/x_norm
    self.w = [w + u*x.conjugate() for (w, x) in zip(self.w, self.x)]
    return
