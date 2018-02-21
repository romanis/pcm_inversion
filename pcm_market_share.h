/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   pcm_market_share.h
 * Author: Roman
 *
 * Created on February 13, 2018, 5:07 PM
 */

#ifndef PCM_MARKET_SHARE_H
#define PCM_MARKET_SHARE_H
#include <vector>
#include <map>
#include "/storage/home/rji5040/work/Tasmanian_run/include/TasmanianSparseGrid.hpp"
#include <omp.h>
#include "/opt/aci/sw/knitro/10.2.1/include/knitro.h"
#include "KTRSolver.h"
#include "KTRProblem.h"

class pcm_market_share : public knitro::KTRProblem{
private:
    int dimension;
//    dimension - the dimension of heterogeneity. size D
    std::vector<std::vector<double> > grid;
//    grid - the grid to integrate with respect to draws of heterogeneity. size KxD
    std::vector<double> weights;
//    weights of each point in  grid size K
    std::vector<double> shares_data;
//    the market shares of real data that need to be fitted. Size N
    std::vector<std::vector<double>> x;
//    x is observable heterogeneity. size NxD 
    std::vector<double> sigmax;
//    vector of standard deviations of each heterogeneity parameter
    std::vector<double> p;
//    vector of prices size N
    double sigma_p;
//     standard deviation of log of price elsaticity
    
    
public:
    pcm_market_share ();
//    default constructor
    pcm_market_share (std::vector<double> sch, std::vector<std::vector<double>> x, std::vector<double> sigmax, std::vector<double> p, double sigma_p):  
        KTRProblem(sch.size(), 0), dimension(1), shares_data(sch), x(x), sigmax(sigmax), p(p), sigma_p(sigma_p)  { 
    //    set up grid and weights
        std::vector<double> point;
        point.push_back(0);
        grid.push_back(point);
        weights.push_back(1);
        setObjectiveProperties();
        setVariableProperties();
        setConstraintProperties(0);
    }
    
    std::vector<double> get_sigma_x(){
        return sigmax;
    }
    
    std::vector<double> initial_guess(); // calculates initial guess based on undisturbed model.
    void get_traction(); // pushes initial point up so that each market share if greater than 1e-4;
    bool set_grid(std::vector<std::vector<double> > grid1, std::vector<double> weights1);
    bool set_grid(int dim, int n); // sets up a grid to integrate with respect to standard normal measure gauss hermite quadrature with dimension and number of points in each dimension
    void set_shares(std::vector<double> sch){
        shares_data = sch;
    }
    void decrease_sigma_x(double factor = 2){
        for(int it = 0; it<sigmax.size(); ++it){
            sigmax[it] /= factor;
        }
    }
    void increase_sigma_x(double factor = 2){
        for(int it = 0; it<sigmax.size(); ++it){
            sigmax[it] *= factor;
        }
    }
    
    
    double relax_til_solved(std::vector<double> & solution, std::vector<double> starting_point);
//    this function reduces variance of sigma_x until problem solves, reports ratio of final delta to original one
    
    std::vector<double> unc_share(std::vector<double> delta_bar, std::vector<std::vector<double>> x, std::vector<double> p, double sigma_p, std::vector<double> sigma_x ); //does the same but does not calculate jacobian
    std::vector<double> unc_share(std::vector<double> delta_bar, std::vector<std::vector<double>> x, std::vector<double> p, double sigma_p, std::vector<double> sigma_x, std::vector<std::vector<double> > & jacobian ); //does the same but does not calculate jacobian
    
//    knitro initialization
    void setObjectiveProperties() {
        setObjType(knitro::KTREnums::ObjectiveType::ObjGeneral);
        setObjGoal(knitro::KTREnums::ObjectiveGoal::Minimize);
    }

    // variable bounds. All variables 0 <= x.
    void setVariableProperties() {
        setVarLoBnds(-KTR_INFBOUND);
        setVarUpBnds( KTR_INFBOUND);
    }

    // constraint properties
    void setConstraintProperties() {
        // set constraint types
//        setConTypes(0, knitro::KTREnums::ConstraintType::ConGeneral);
//        setConTypes(1, knitro::KTREnums::ConstraintType::ConGeneral);

        // set constraint lower bounds to zero for all variables
        setConLoBnds(0.0);

        // set constraint upper bounds
//        setConUpBnds(0, 0.0);
//        setConUpBnds(1, KTR_INFBOUND);
    }
    
    // constraint properties when know how many constrains
    void setConstraintProperties(int num_constr) {
        // set constraint types
        for(int i = 0; i< num_constr;++i){
            setConTypes(i, knitro::KTREnums::ConstraintType::ConGeneral);
        }

        // set constraint lower bounds to zero for all variables
        setConLoBnds(0.0);

        // set constraint upper bounds
        setConUpBnds(0.0);
    }
    
    double evaluateFC(const std::vector<double>& delta,  std::vector<double>& c,  std::vector<double>& objGrad, std::vector<double>& jac){//, std::vector<double> & p, std::vector<std::vector<double>> x, double sigma_p, std::vector<double> sigma_x) {
        std::vector<std::vector<double>> jacobian;

//        compute market share predicted by model
        std::vector<double> share_predict = this->unc_share(delta, x, p, sigma_p, sigmax, jacobian);
//        std::cout<<"here"<<std::endl;
        double obj=0;
          // constraints
        jac.clear();
        c.clear();
        for(int i=0; i<shares_data.size(); ++i){
//            c.push_back(share_predict[i] - shares_data[i]);
            obj += (share_predict[i] - shares_data[i])*(share_predict[i] - shares_data[i]);
        }
          // return objective function value
          return obj;
      }
    
    int evaluateGA(const std::vector<double>& delta, std::vector<double>& objGrad, std::vector<double>& jac) {
        std::vector<std::vector<double>> jacobian;
        
        std::vector<double> share_predict = this->unc_share(delta, x, p, sigma_p, sigmax, jacobian);
//        jac.clear();
        std::vector<double> obj_tmp(delta.size(),0.0);
        objGrad = obj_tmp;
        for(int i=0; i< shares_data.size(); ++i){
//            std::cout<<"here\n";
            for(int j=0; j<shares_data.size(); ++j){
                objGrad[j] += 2*jacobian[i][j]*(share_predict[i] - shares_data[i]);
            }
            
//            for(int j = 0; j< jacobian[i].size(); ++j){
//                jac.push_back(jacobian[i][j]);
//            }
        }
	return 0;
    }
 
};


std::vector<double> cond_share(std::vector<double> delta, std::vector<double> p, double sigma_p, std::vector<std::vector<double> > & jacobian ); // computes market share of vertical model (conditional on draw of heterogeneity. 
std::vector<double> cond_share(std::vector<double> delta, std::vector<double> p, double sigma_p ); //does the same but does not calculate jacobian

inline void print_jacobian(std::vector<std::vector<double> > jac){
    for(auto p : jac){
        for(auto b : p){
            std::cout<<b<<"\t";
        }
        std::cout<<std::endl;
    }
}
inline pcm_market_share::pcm_market_share() : KTRProblem(1, 0), dimension(1), shares_data({})  { 
//    set up grid and weights
    std::vector<double> point;
    point.push_back(0);
    grid.push_back(point);
    weights.push_back(1);
    setObjectiveProperties();
    setVariableProperties();
    setConstraintProperties();
}

#endif /* PCM_MARKET_SHARE_H */
