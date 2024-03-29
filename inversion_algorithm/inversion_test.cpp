#include "gtest/gtest.h"
#include "market_inversion.hpp"
#include "Eigen/Dense"
#include "Eigen/Core"

#include <boost/random.hpp>
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

    auto inverted_deltas = share_inversion::invert_shares(param);
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

    auto inverted_deltas = share_inversion::invert_shares(param);
    
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

    auto inverted_deltas = share_inversion::invert_shares(param);
    
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

    auto inverted_deltas = share_inversion::invert_shares(param);
    
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

    auto inverted_deltas = share_inversion::invert_shares(param);
    
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

    auto inverted_deltas = share_inversion::invert_shares(param);

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

    auto inverted_deltas = share_inversion::invert_shares(param);

    Eigen::MatrixXd tmp(num_prod, 3);
    
    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
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

    auto inverted_deltas = share_inversion::invert_shares(param);

    Eigen::MatrixXd tmp(num_prod, 3);
    
    ASSERT_LE((inverted_deltas - deltas(positive)).abs().sum(), param.delta_step_tolerance * num_prod * 10);
}