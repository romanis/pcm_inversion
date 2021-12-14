# What does this code do?
This code implements methods that compute and invert [Pure Charactetistics Model](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-2354.2007.00459.x). 

# Prerequisites
## Software that is essential for the library to compile
- cmake
- [Boost](https://www.boost.org/users/download/)
- [Eigen](https://eigen.tuxfamily.org/) at least 3.4 version is required. Beware that as of December 2021 Ubuntu `apt` is only distributing 3.3, so, you might want to build form source.
- [NLopt](https://nlopt.readthedocs.io/) This is the only solver that is supported so far, in the future I might add additional ones
## Software that is highly recommented 
- [Tasmanian](https://tasmanian.ornl.gov/) This library is not required for the inversion to work, but is useful for examples and in general, for 

# Installation 

This is a software designed to be compiled an run on a Unix machine only.

To install: checkout this code to your computer

```
git checkout https://github.com/romanis/pcm_inversion
```
Enter the directory and create build directory
```
cd pcm_inversion
mkdir build
cd build
```
Run cmake and then make
```
cmake ..
make
```
At this point the libraries are built and are located 
inside `build/market_share` and `build/inversion_algorithm` directories. 
You can leave them there and add these paths to your link path and add the 
`pcm_inversion/market_share` and `pcm_inversion/inversion_algorithm` to your 
include path. 

But it is advisable that you install
the required files in system path (if you have administrative privilages) by running 
```
sudo make install
```

If you had Tasmanian library installed, you also have built the example folder `pcm_inversion/example/test_pcm.cpp`. Executable of it is in `pcm_inversion/build/examples/test_pcm`. 

# Quick user guide
There are two essential libraries that are built by this code: one that computes market shares (along with jecobians)
conditional on structural parameters, the other uses these computations in conjunction with NLopt solver
to solve MPEC style feasibility problem trying to equate observed market shares and the predicted ones.
## Market share computation library
The source is located at `pcm_inversion/market_share`. All functions are put in `pcm_share::` scope and the main function to call is `pcm_share::unc_share` that computes PCM shares conditional on structural parameters. Here is the signature:

```
Eigen::ArrayXd 
unc_share(const Eigen::ArrayXd& delta_bar, const Eigen::MatrixXd& x, const Eigen::ArrayXd& p, 
        double sigma_p, const Eigen::ArrayXd& sigma_x, const Eigen::ArrayXXd& grid, 
        const Eigen::ArrayXd & weights, Eigen::MatrixXd & jacobian);
```
(there is also same names function with a signature without the last argument, which would skip Jacobian computation)

The inputs are:
- delta_bar - Eigen Array of average over population vertical qualities of each product. Has size `$N_{products}$`

The output of the function is an Eigen Array that corresponds to the predicted shares of every product.

# Author
Roman Istomin

- [github/romanis](https://github.com/romanis)

# License
Copyright © 2021, Roman Istomin. 


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software") FOR NON-COMMERCIAL PURPOSES, to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright does not allow the use of the Software for profit, and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

# Citing 
If you use this Libraries in work that leads to a publication, I would appreciate it if you would kindly cite Me in your manuscript. Cite Library as something like:

Roman Istomin, The PCM Inversion library, https://github.com/romanis/pcm_inversion

