## What does this document describe?
This Document describes how to use the Python bindings for PCM inversion library. 

## Prerequisites
To build Python bindings library, one needs to install [Pybind11](https://pybind11.readthedocs.io/en/latest/index.html) library. 

On Mac it is installed with 
```
brew install pybind11
```

You should have Python installed in your system, but just in case, on Mac it is installed with brew:

```
brew install python3
```
## Installation

Once you have the Pybind11 and Python installed, the rest is just the same steps that are needed to compile and install all the rest of the library, these are descrived in `Installation` section of [Main Readme](../README.md)

Assuming you have ran `sudo make install`, your bindings library gets copied to `/usr/local/lib/`

## Usage

The exposed functions have in-code documentation. Each function has extensive explanation of the types expected for each argument. Market share functions are detailed in [pcm_market_share.py](pcm_market_share.py) and inversion function is detailed in [pcm_inversion_algorithm.py](pcm_inversion_algorithm.py).

An exmple of usage is given in [market_share_python_bindings_example.py](market_share_python_bindings_example.py). 
