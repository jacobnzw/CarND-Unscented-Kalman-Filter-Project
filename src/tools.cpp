#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
     * Calculates RMSE of the estimated trajectory.
     */
    unsigned int dim = estimations[0].size();
    unsigned int num_steps = estimations.size();
    VectorXd sum, diff;
    sum.setZero(dim);
    for (unsigned int i = 0; i < estimations.size(); i++) {
        diff = estimations[i] - ground_truth[i];
        sum += diff.cwiseProduct(diff);
    }
    return (sum / num_steps).cwiseSqrt();
}