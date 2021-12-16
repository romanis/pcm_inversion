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
the required files in system path (if you have administrative privilege) by running 
```
sudo make install
```

If you had Tasmanian library installed, you also have built the example folder `pcm_inversion/example/test_pcm.cpp`. Executable of it is in `pcm_inversion/build/examples/test_pcm`. 

# Quick user guide
There are two essential libraries that are built by this code: one that computes market shares (along with Jacobian)
conditional on structural parameters, the other uses these computations in conjunction with NLopt solver
to solve MPEC style feasibility problem trying to equate observed market shares and the predicted ones.
## Market share computation library
The source is located at `pcm_inversion/market_share`. All functions are put in `pcm_share::` namespace and the main function to call is `pcm_share::unc_share` that computes PCM shares in the market with
`N` products each having `K` characteristics of horizontal differentiation (except for price) 
conditional on structural parameters. Here is the signature:

```
Eigen::ArrayXd 
unc_share(
    const Eigen::ArrayXd& delta_bar, 
    const Eigen::MatrixXd& x, 
    const Eigen::ArrayXd& p, 
    double sigma_p,
    const Eigen::ArrayXd& sigma_x, 
    const Eigen::ArrayXXd& grid, 
    const Eigen::ArrayXd & weights, 
    Eigen::MatrixXd & jacobian
);
```
(there is also same names function with a signature without the last argument, which would skip Jacobian computation)

The inputs are:
- `delta_bar` - Eigen array of average over population vertical qualities of each product. Has size `N`
- `x` - Eigen matrix that has `N` rows, each row containing values of `K` numerical horizontal 
characteristics of the products (e.g. volume, horsepower, CPU clock, 
battery of dummies for the manufacturer etc.)
- `p` - Eigen array of prices of each product. Has size `N`. It is important that products are sorted 
in ascending order by their price. It is also important that 
products with the same price do have horizontal differences.
- `sigma_p` - standard deviation of (log of) price sensitivity. Log of price sensitivity is assumed to be 
distributed normally with zero mean and standard deviation of sigma_p
- `sigma_x` - Eigen array of standard deviations of the idiosyncratic preferences for each 
horizontal characteristic in population. Has size `K`. Marginal indirect utility (or preference) 
for horizontal characteristic `i` is assumed to have some distribution parameterized 
by only scale parameter `sigma_x[i]`. E.g. each horizontal characteristic's preference 
in population may have normal distribution with mean zero and std  `sigma_x[i]`.
- `grid` - Eigen 2 dimensional array that contains draws that numerically 
integrate out the distribution of idiosyncratic preferences. 
Has the size `[D; K]` where `D` is the number of draws. One can use any grid generation technique, but
I recommend Tasmanian Sparse Grids.
- `weights` - Eigen array of size `D` that contains the weights of draws in the grid
- `jacobian` - Eigen matrix containing the jacobian of predicted market shares with respect to delta_bar

The output of the function is an Eigen Array that corresponds to the predicted shares of every product.

Next function is important because it produces an initial guess, which is a solution to "unperturbed"
demand without any idiosyncratic preferences, in other word, vertical qualities that would rationalize 
observed shares in a pure vertical model [Bresnahan 1987 style](http://homes.chass.utoronto.ca/~jovb/ECO2901/Bresnahan_JIndE87.pdf)

```
Eigen::ArrayXd 
initial_guess(
    const Eigen::ArrayXd& shares_data, 
    const Eigen::ArrayXd& p, 
    double sigma
)
```
The inputs are:

- `shared_data` - Eigen array of shares that each product has in the data
- `p` - Eigen array of prices of every product. It is important that prices are sorted in ascending order
- `sigma` - standard deviation of log of price sensitivity

## Share inversion library
The source is located at `pcm_inversion/market_inversion`. All functions are put in `share_inversion::` namespace.  
The main function to call is `share_inversion::invert_shares` here is the signature:

```
Eigen::ArrayXd 
invert_shares(
    pcm_parameters & params
)
```

The function inverts market shares to find pure vertical product qualities *in this market*, i.e. 
qualities that all consumers of this market can agree on. 
It does not do any cross market comparisons, it does 
not further process the vertical qualities to produce moment conditions etc. All of it should be done outside of these libraries.

The only parameter is the `share_inversion::pcm_parameters` class object that contains all the information about this particular market. Here is the definition of it:

```
struct pcm_parameters{
        Eigen::MatrixXd x; //!< Matrix of features of the products. Size [n_products x n_features]
        Eigen::ArrayXd p; //!< Array of prices of products. Size n_products
        double sigma_p; //!< Standard deviation of the distribution of (log of) price sensitivity
        Eigen::ArrayXd sigma_x; //!< Array of standard deviations of random taste for every feature. Size n_features
        Eigen::ArrayXXd grid; //!< Grid that integrates out the distribution for the random taste for features. Size [n_draws x n_features]. The more draws the more exact is the integration
        Eigen::ArrayXd weights; //!< Array of weights of every point in a grid. Size n_draws
        Eigen::ArrayXd data_shares; //!< Observed shares of products. Size n_products. Should sum up to (strictly) less than 1
        int func_evals = 0, jacobian_evals = 0; //!< counters of the time we have called the function with and without jacobian
        nlopt::algorithm nlopt_algo; //!< one of the algorithms from NLOPT library. default is nlopt::LD_SLSQP
        double share_equality_tolerances; //!< tolerance with withich we expect the inversion to come to observed shares. default is 1e-8
        double delta_step_tolerance; //!< tolerance to step in delta space. default is 1e-5
        std::vector<double> delta_initial; //!< initial step for the inversion. 
                                            //!< Highly recommended NOT to provide it because the algorithm has a very good initial guess of its own. Only provide it if you really know what you are doing.
                                            //!< If not all shares are positive at this initial guess, it will be ignored.
        unsigned max_number_of_function_calls; //!< maximum number of times share predicting function will be called. default 1000
        unsigned number_of_times_function_called = 0;
        


        pcm_parameters(Eigen::MatrixXd& x, Eigen::ArrayXd &p, double sigma_p, Eigen::ArrayXd &sigma_x, Eigen::ArrayXXd& grid, 
                       Eigen::ArrayXd& weights, Eigen::ArrayXd& data_shares, std::vector<double> delta_initial = std::vector<double>(), 
                       nlopt::algorithm nlopt_algo = nlopt::LD_SLSQP, 
                       double share_equality_tolerances = 1e-8, double delta_step_tolerance = 1e-5, unsigned max_number_of_function_calls = 1000) :
                            x(x), p(p), sigma_x(sigma_x), sigma_p(sigma_p), grid(grid), weights(weights), data_shares(data_shares), 
                            nlopt_algo(nlopt_algo), share_equality_tolerances(share_equality_tolerances), delta_step_tolerance(delta_step_tolerance), delta_initial(delta_initial),
                            max_number_of_function_calls(max_number_of_function_calls) {}; //!< constructor of the struct
    } ;
```

One important constant that is part of `share_inversion::` namespace is `MIN_ADMISSIBLE_SHARE = 1e-5` which 
defines smallest share of any product (or outside option) such that current algorithms still reliably invert 
the shares to get vertical qualities.



# Author
Roman Istomin

- [github/romanis](https://github.com/romanis)
- [https://www.linkedin.com/in/roman-istomin/](https://www.linkedin.com/in/roman-istomin/)

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

