#include <gtest/gtest.h>
#include "pcm_market_share.hpp"
#include "Eigen/Dense"
#include <boost/random.hpp>
#include <iostream>

TEST(InitialGuessTest, Basic_calculation) {
    int num_prod = 5;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd shares_data(num_prod);

    
    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.0);
        shares_data[i-1] = 1.0/(num_prod+1);
    }

    auto initial_guess = pcm_share::initial_guess(shares_data, p, 1.0);
    ASSERT_TRUE((initial_guess > 0).all());
    for(int i=0; i<num_prod-1; ++i){
        ASSERT_LE(initial_guess[i], initial_guess[i+1]);
    }

}

TEST(InitialGuessTest, Equal_prices_in_the_middle) {
    int num_prod = 5;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd shares_data(num_prod);

    
    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        shares_data[i-1] = 1.0/(num_prod+1);
    }
    p[2] = p[3];

    auto initial_guess = pcm_share::initial_guess(shares_data, p, 1.0);
    ASSERT_TRUE((initial_guess > 0).all());
    for(int i=0; i<num_prod-1; ++i){
        ASSERT_LE(initial_guess[i], initial_guess[i+1]);
    }

}

TEST(InitialGuessTest, Equal_prices_in_the_beginning) {
    int num_prod = 5;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd shares_data(num_prod);

    
    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        shares_data[i-1] = 1.0/(num_prod+1);
    }
    p[0] = p[1];

    auto initial_guess = pcm_share::initial_guess(shares_data, p, 1.0);
    ASSERT_TRUE((initial_guess > 0).all());
    for(int i=0; i<num_prod-1; ++i){
        ASSERT_LE(initial_guess[i], initial_guess[i+1]);
    }

}

TEST(InitialGuessTest, Equal_prices_in_the_end) {
    int num_prod = 5;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd shares_data(num_prod);

    
    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        shares_data[i-1] = 1.0/(num_prod+1);
    }
    p[num_prod-1] = p[num_prod-2];

    auto initial_guess = pcm_share::initial_guess(shares_data, p, 1.0);
    ASSERT_TRUE((initial_guess > 0).all());
    for(int i=0; i<num_prod-1; ++i){
        ASSERT_LE(initial_guess[i], initial_guess[i+1]);
    }

}

TEST(InitialGuessTest, Exponentially_Distributed_shares) {
    int num_prod = 50;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd shares_data(num_prod);

    // create random number generator for shares
    boost::mt19937 seed(5u); 
    boost::variate_generator<boost::mt19937&, boost::exponential_distribution<>> random_n(seed, boost::exponential_distribution<>()) ;

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.5);
        shares_data[i-1] = random_n();
    }
    // make shares sum to 1/1.4
    shares_data /= 1.4*shares_data.sum();

    auto initial_guess = pcm_share::initial_guess(shares_data, p, 1.0);
    ASSERT_TRUE((initial_guess > 0).all());
    for(int i=0; i<num_prod-1; ++i){
        ASSERT_LE(initial_guess[i], initial_guess[i+1]);
    }
}

TEST(CondShareTest, Simple_Computation) {
    int num_prod = 50;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.1);
        deltas[i-1] = i;
    }

    auto shares = pcm_share::cond_share(deltas, p, 1.0);
    ASSERT_TRUE((shares > 0).all());
}

TEST(CondShareTest, One_product_with_equal_quality) {
    int num_prod = 50;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.1);
        deltas[i-1] = i;
    }
    deltas[4] = deltas[3]-1;
    Eigen::MatrixXd  jacobian;
    auto shares = pcm_share::cond_share(deltas, p, 1.0, jacobian);
    ASSERT_EQ(shares[4], 0);
    for(int i = 0; i< num_prod; ++i){
        if(i != 4){
            ASSERT_GE(shares[i], 0);
        }
    }
}
TEST(CondShareTest, One_product_with_equal_quality_no_jacobian) {
    int num_prod = 50;
    Eigen::ArrayXd p(num_prod);
    Eigen::ArrayXd deltas(num_prod);

    // fill prices and shares
    for(int i = 1; i<= num_prod; ++i){
        p[i-1] = pow(i, 1.1);
        deltas[i-1] = i;
    }
    deltas[4] = deltas[3]-1;
    auto shares = pcm_share::cond_share(deltas, p, 1.0);
    ASSERT_EQ(shares[4], 0);
    for(int i = 0; i< num_prod; ++i){
        if(i != 4){
            ASSERT_GE(shares[i], 0);
        }
    }
}