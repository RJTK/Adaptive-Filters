from collections import deque #A queue

def system_identification_setup(F):
  '''
  F should specify an instance of a recursive filter implementing ff
  and fb.  We return the filter parameters of the filter w, which can
  be used to identify a model of a system.
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
  F should specify an instance of a recursive filter implementing ff
  and fb.  This function uses F to make a 1 step prediction of a
  process.  This is basically a method to track the process over time.

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
  Sets up a function to use F as an adaptive equalizer.  The
  argument x to ff_fb is the filter's observation, and d is the true
  value of what it should estimate.  Hence if we have x = Hd + v where
  Hd is a filtered data sequence and v is noise, then passing in x and
  d to ff_fb will train an equalizer for H.

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
