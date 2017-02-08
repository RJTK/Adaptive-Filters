# Adaptive Filters
This repo provides some information and implementations of adaptive filtering algorithms.  This is mostly for my own interest, and the implementations are not rigorously tested.  However, if anyone takes an interest in using this code I would be happy to receive requests for information or assistance.

__Background Info__:
If we have two random processes x(n) and d(n), we can make some inference about d(n) if we know the statistics of the two processes.  In particular, we could design a [Wiener filter](https://en.wikipedia.org/wiki/Wiener_filter) if we knew the cross correlation between x and d.

In practice however these statistics are usually not known, and may be time varying.  Another option is to use an adaptive filter which updates it's method of inferring d from x over time. The Least Mean Squares (LMS), Recursive Least Squares (RLS), and their variants make linear predictions with FIR filters: d ~= w^T * x with w being FIR filter taps.  The Least Mean Squares (LMS) filter [see here](https://en.wikipedia.org/wiki/Least_mean_squares_filter) is one of the simplest possible adaptive filters, and the RLS filter has been known since the time of Gauss.  However, these filters and variations thereof are widely applied in practice, and papers are still being published about them.  In particular, it is at the height of fashion to refer to these methods as techniques for Online Learning.

This repo currently contains implementations of:

     -LMS (Classic LMS algorithm)
     -LMS_Normalized (Dynamically varying step size)
     -LMS_ZA (Zero Attracting LMS.  Incorporates sparsity assumptions on w)
     -LMS_RZA (Rewighted LMS_ZA.  Reduces bias of LMS_ZA)
     -RLS (Classic RLS algorithm)

Example uses include tracking a time varying process, system identification, and channel identification.  These examples are given in test.py

An example of using the LMS filter to track a brownian motion process:

![alt tag](https://raw.githubusercontent.com/RJTK/LMS/master/tracking.png)

Obviously, the filter can perform much better if the processes are Jointly WSS, or approximately so.

RLS filter tracking a signal with time varying statistics:

![alt tag](https://raw.githubusercontent.com/RJTK/LMS/master/RLS_tracking.png)

References:

__LMS and LMS_Normalized__: Statistical Digital Signal Processing and Modeling - Monson H. Hayes

__RLS__: ibid.

__LMS_ZA and LMS_RZA__: Chen, Yilun, Yuantao Gu, and Alfred O. Hero. "Sparse LMS for system identification." Acoustics, Speech and Signal Processing, 2009. ICASSP 2009. IEEE International Conference on. IEEE, 2009. 

