# Liquid State Machine (LSM)

The Liquid State Machine (LSM) ([Maass et. al. 2002][1]) is a computational model
which uses the high-dimensional, complex dynamics of recurrent neural circuits to
conduct memory-dependent readout operations on continuous input streams.

[1]: http://dx.doi.org/10.1162/089976602760407955

This package provides a convenience wrapper for network construction, as well as typical
operations on the reservoir.

Note that this code does not implement the exact model in [Maass et. al. 2002][1], but rather a slightly simplified model. This code was used in ([Kaiser et al. 2017][2]).

[2]: https://iopscience.iop.org/article/10.1088/1748-3190/aa7663/meta

## Usage

After cloning this repository locally, run `pip install .` in the working copy. Requires a working installation of the [NEST simulator](http://www.nest-initiative.org).
