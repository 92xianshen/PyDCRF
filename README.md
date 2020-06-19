# PyDCRF

This is a Python-based Dense CRF implementation for [Philipp Krähenbühl's Fully-Connected CRFs](http://web.archive.org/web/20161023180357/http://www.philkr.net/home/densecrf).

This code is inspired by PyDenseCRF (https://github.com/lucasb-eyer/pydensecrf) and CRFasRNN-keras (https://github.com/sadeepj/crfasrnn_keras).

The mean-field approximation is implemeneted in Python, but the high-dim filtering for efficient message passing is implemented in C using Cython-based wrapper.

Please cite  

```
Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
Philipp Krähenbühl and Vladlen Koltun
NIPS 2011
```

and this repo if the code is of help for your research.