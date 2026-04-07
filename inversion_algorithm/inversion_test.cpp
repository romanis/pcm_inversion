#include "gtest/gtest.h"
#include "market_inversion.hpp"
#include "Eigen/Dense"
#include "Eigen/Core"

#include <boost/random.hpp>
#include <chrono>
#include <iostream>

TEST(PCMInversion, Basic_calculation) {
    int num_prod = 50;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd shares_data(num_prod);

    
    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        shares_data[i-1] = 1.0/(num_prod+1);
    }
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;
    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares_data);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);
    ASSERT_TRUE((inverted_deltas > 0).all());
    for(int i=0; i<num_prod-1; ++i){
        ASSERT_LE(inverted_deltas[i], inverted_deltas[i+1]);
    }
}

TEST(PCMInversion, Basic_parameter_back_out) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);

    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);
    
    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

TEST(PCMInversion, Two_deltas_the_same_at_low_end) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[0] = 2;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);

    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);
    
    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

TEST(PCMInversion, Two_deltas_the_same_at_in_the_middle) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.3);
        deltas[i-1] = i;
    }
    deltas[4] = deltas[5];

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);
    
    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

TEST(PCMInversion, Two_deltas_the_same_at_high_end) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.3);
        deltas[i-1] = i;
    }
    deltas[num_prod-1] = deltas[num_prod-2];

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);

    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);
    
    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

TEST(PCMInversion, zero_share_at_low_end) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[0] = -100;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    std::vector<int> positive;
    for(int i=0; i< num_prod; ++i){
        if(shares[i] > 0){
            positive.push_back(i);
        }
    }
    Eigen::MatrixXd x_positive = x(positive, Eigen::seq(0, x.cols()-1));
    Eigen::ArrayXd p_positive = p(positive);
    Eigen::ArrayXd shares_positive = shares(positive);

    share_inversion::pcm_parameters param(x_positive , p_positive, 1.0, sigma_x, grid, weights, shares_positive);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);

    Eigen::MatrixXd tmp(num_prod, 3);
    
    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

TEST(PCMInversion, zero_share_middle) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[6] = -100;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    std::vector<int> positive;
    for(int i=0; i< num_prod; ++i){
        if(shares[i] > 0){
            positive.push_back(i);
        }
    }
    Eigen::MatrixXd x_positive = x(positive, Eigen::seq(0, x.cols()-1));
    Eigen::ArrayXd p_positive = p(positive);
    Eigen::ArrayXd shares_positive = shares(positive);

    share_inversion::pcm_parameters param(x_positive , p_positive, 1.0, sigma_x, grid, weights, shares_positive);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);

    Eigen::MatrixXd tmp(num_prod, 3);
    
    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Verifies that CONE_AWARE_NEWTON and NLOPT_SLSQP produce identical delta
 *        vectors to within 1e-6 absolute tolerance on a 3-product PCM market with
 *        sigma_x > 0, confirming both inversion methods reach the same solution.
 *
 * @details data_shares are forward-computed from known_deltas = {1, 2, 3} so that
 *          all shares are strictly positive and both solvers have a valid solution.
 *          sigma_x = 0.5 ensures the unc_share Jacobian is dense and the cone-aware
 *          Newton solver is exercised (not the trivial sigma_x = 0 closed-form path).
 */
TEST(PCMInversion, ConeAwareNewton_agrees_with_NLOPT_SLSQP) {
    const int num_prod  = 25;
    const int num_dim   = 5;
    const int num_draws = 10000;

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);

    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd known_deltas(num_prod);
    
    for(int i = 1; i <= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        known_deltas[i-1] = i;
    }

    const double sigma_p = 1.0;

    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);

    Eigen::ArrayXXd grid    = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd  weights = Eigen::ArrayXd::Ones(num_draws) / num_draws;

    Eigen::ArrayXd data_shares = pcm_share::unc_share(
        known_deltas, x, p, sigma_p, sigma_x, grid, weights);

    ASSERT_TRUE((data_shares > 0).all());

    share_inversion::pcm_parameters param_slsqp(
        x, p, sigma_p, sigma_x, grid, weights, data_shares);

    share_inversion::pcm_parameters param_newton(
        x, p, sigma_p, sigma_x, grid, weights, data_shares);

    Eigen::ArrayXd delta_slsqp  = share_inversion::invert_shares(
        param_slsqp, share_inversion::InversionMethod::NLOPT_SLSQP);
    Eigen::ArrayXd delta_newton = share_inversion::invert_shares(
        param_newton, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    ASSERT_LE((delta_slsqp - delta_newton).abs().maxCoeff(), 1e-6);
}

TEST(PCMInversion, zero_share_at_high_end) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[num_prod-1] = -100;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    std::vector<int> positive;
    for(int i=0; i< num_prod; ++i){
        if(shares[i] > 0){
            positive.push_back(i);
        }
    }
    Eigen::MatrixXd x_positive = x(positive, Eigen::seq(0, x.cols()-1));
    Eigen::ArrayXd p_positive = p(positive);
    Eigen::ArrayXd shares_positive = shares(positive);

    share_inversion::pcm_parameters param(x_positive , p_positive, 1.0, sigma_x, grid, weights, shares_positive);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::NLOPT_SLSQP);

    Eigen::MatrixXd tmp(num_prod, 3);
    
    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Mirrors Basic_calculation using CONE_AWARE_NEWTON: verifies that
 *        inversion returns positive deltas in ascending order for uniform shares
 *        on a 50-product, 5-dimensional market.
 */
TEST(PCMInversion, Basic_calculation_ConeNewton) {
    int num_prod = 50;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd shares_data(num_prod);

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        shares_data[i-1] = 1.0/(num_prod+1);
    }
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;
    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares_data);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);
    ASSERT_TRUE((inverted_deltas > 0).all());
    for(int i=0; i<num_prod-1; ++i){
        ASSERT_LE(inverted_deltas[i], inverted_deltas[i+1]);
    }
}

/**
 * @brief Mirrors Basic_parameter_back_out using CONE_AWARE_NEWTON: verifies that
 *        forward-computed shares are inverted back to the known deltas within tolerance
 *        on a 25-product, 5-dimensional market.
 */
TEST(PCMInversion, Basic_parameter_back_out_ConeNewton) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);

    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Mirrors Two_deltas_the_same_at_low_end using CONE_AWARE_NEWTON: verifies
 *        inversion accuracy when the two lowest-quality products have identical deltas.
 */
TEST(PCMInversion, Two_deltas_the_same_at_low_end_ConeNewton) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[0] = 2;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);

    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Mirrors Two_deltas_the_same_at_in_the_middle using CONE_AWARE_NEWTON: verifies
 *        inversion accuracy when two mid-range products share the same delta value.
 */
TEST(PCMInversion, Two_deltas_the_same_at_in_the_middle_ConeNewton) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.3);
        deltas[i-1] = i;
    }
    deltas[4] = deltas[5];

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Mirrors Two_deltas_the_same_at_high_end using CONE_AWARE_NEWTON: verifies
 *        inversion accuracy when the two highest-quality products have identical deltas.
 */
TEST(PCMInversion, Two_deltas_the_same_at_high_end_ConeNewton) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.3);
        deltas[i-1] = i;
    }
    deltas[num_prod-1] = deltas[num_prod-2];

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);

    ASSERT_TRUE((shares > 0).all());

    share_inversion::pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, shares);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    ASSERT_LE((inverted_deltas - deltas).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Mirrors zero_share_at_low_end using CONE_AWARE_NEWTON: verifies inversion
 *        accuracy when the lowest-quality product is excluded due to near-zero share,
 *        using the reduced positive-share product set.
 */
TEST(PCMInversion, zero_share_at_low_end_ConeNewton) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[0] = -100;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    std::vector<int> positive;
    for(int i=0; i< num_prod; ++i){
        if(shares[i] > 0){
            positive.push_back(i);
        }
    }
    Eigen::MatrixXd x_positive = x(positive, Eigen::seq(0, x.cols()-1));
    Eigen::ArrayXd p_positive = p(positive);
    Eigen::ArrayXd shares_positive = shares(positive);

    share_inversion::pcm_parameters param(x_positive, p_positive, 1.0, sigma_x, grid, weights, shares_positive);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    Eigen::MatrixXd tmp(num_prod, 3);

    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Mirrors zero_share_middle using CONE_AWARE_NEWTON: verifies inversion
 *        accuracy when a mid-range product is excluded due to near-zero share,
 *        using the reduced positive-share product set.
 */
TEST(PCMInversion, zero_share_middle_ConeNewton) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[6] = -100;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    std::vector<int> positive;
    for(int i=0; i< num_prod; ++i){
        if(shares[i] > 0){
            positive.push_back(i);
        }
    }
    Eigen::MatrixXd x_positive = x(positive, Eigen::seq(0, x.cols()-1));
    Eigen::ArrayXd p_positive = p(positive);
    Eigen::ArrayXd shares_positive = shares(positive);

    share_inversion::pcm_parameters param(x_positive, p_positive, 1.0, sigma_x, grid, weights, shares_positive);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    Eigen::MatrixXd tmp(num_prod, 3);

    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Mirrors zero_share_at_high_end using CONE_AWARE_NEWTON: verifies inversion
 *        accuracy when the highest-quality product is excluded due to near-zero share,
 *        using the reduced positive-share product set.
 */
TEST(PCMInversion, zero_share_at_high_end_ConeNewton) {
    int num_prod = 25;
    int num_dim = 5;
    int num_draws = 10000;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_dim);
    Eigen::ArrayXXd grid = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(num_draws)/num_draws;

    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        deltas[i-1] = i;
    }
    deltas[num_prod-1] = -100;

    auto shares = pcm_share::unc_share(deltas, x, p, 1.0, sigma_x, grid, weights);
    std::vector<int> positive;
    for(int i=0; i< num_prod; ++i){
        if(shares[i] > 0){
            positive.push_back(i);
        }
    }
    Eigen::MatrixXd x_positive = x(positive, Eigen::seq(0, x.cols()-1));
    Eigen::ArrayXd p_positive = p(positive);
    Eigen::ArrayXd shares_positive = shares(positive);

    share_inversion::pcm_parameters param(x_positive, p_positive, 1.0, sigma_x, grid, weights, shares_positive);

    auto inverted_deltas = share_inversion::invert_shares(param, share_inversion::InversionMethod::CONE_AWARE_NEWTON);

    Eigen::MatrixXd tmp(num_prod, 3);

    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}

/**
 * @brief Benchmarks NLOPT_SLSQP against CONE_AWARE_NEWTON by running each algorithm
 *        N = 1000 times on a 5-product, 1-dimensional PCM market with sigma_x = 0.5
 *        and 500 integration draws, measuring total and per-call wall-clock time via
 *        std::chrono::high_resolution_clock.  Timing results are printed to std::cout;
 *        the test also asserts that both algorithms produce results within 1e-4 of each
 *        other to confirm correctness under the benchmark fixture.
 */
TEST(PCMInversion, Benchmark_NLOPT_vs_ConeNewton) {
    const int num_prod  = 5;
    const int num_dim   = 1;
    const int num_draws = 500;
    const int N         = 1000;

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_dim);

    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd known_deltas(num_prod);

    for (int i = 1; i <= num_prod; ++i) {
        p[i-1]            = pow(i, 1.5);
        known_deltas[i-1] = i;
    }

    const double sigma_p = 1.0;

    Eigen::ArrayXd sigma_x(num_dim);
    sigma_x(0) = 0.5;

    Eigen::ArrayXXd grid    = Eigen::ArrayXXd::Random(num_draws, num_dim);
    Eigen::ArrayXd  weights = Eigen::ArrayXd::Ones(num_draws) / num_draws;

    Eigen::ArrayXd data_shares = pcm_share::unc_share(
        known_deltas, x, p, sigma_p, sigma_x, grid, weights);

    share_inversion::pcm_parameters param_slsqp(
        x, p, sigma_p, sigma_x, grid, weights, data_shares);
    share_inversion::pcm_parameters param_newton(
        x, p, sigma_p, sigma_x, grid, weights, data_shares);

    Eigen::ArrayXd delta_slsqp_final;
    Eigen::ArrayXd delta_newton_final;

    auto t0_slsqp = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N; ++iter) {
        delta_slsqp_final = share_inversion::invert_shares(
            param_slsqp, share_inversion::InversionMethod::NLOPT_SLSQP);
    }
    auto t1_slsqp = std::chrono::high_resolution_clock::now();

    auto t0_newton = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < N; ++iter) {
        delta_newton_final = share_inversion::invert_shares(
            param_newton, share_inversion::InversionMethod::CONE_AWARE_NEWTON);
    }
    auto t1_newton = std::chrono::high_resolution_clock::now();

    double ms_slsqp  = std::chrono::duration<double, std::milli>(t1_slsqp  - t0_slsqp).count();
    double ms_newton = std::chrono::duration<double, std::milli>(t1_newton - t0_newton).count();
    double speedup   = ms_slsqp / ms_newton;

    std::cout << "\n[BENCHMARK] NLOPT_SLSQP:       "
              << ms_slsqp  << " ms total, "
              << ms_slsqp  / N << " ms/call (" << N << " calls)\n"
              << "[BENCHMARK] CONE_AWARE_NEWTON: "
              << ms_newton << " ms total, "
              << ms_newton / N << " ms/call (" << N << " calls)\n"
              << "[BENCHMARK] Speedup: " << speedup << "x\n";

    ASSERT_LE((delta_slsqp_final - delta_newton_final).abs().maxCoeff(), 1e-4);
}
