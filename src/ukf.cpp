#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  // true if the filter was initialized with the first measurement
  is_initialized_ = false;

  // state dimension
  n_x_ = 5;
  // augmented state dimension
  n_aug_ = 7;

  // weights for sigma-points in augmented state-space
  lambda_ = sqrt(3);
  weights_ = VectorXd::Ones(2*n_aug_) / (2*(n_aug_ + lambda_));
  weights_[0] = lambda_ / (n_aug_ + lambda_);

  // storage for predicted sigma-points
  Xsig_pred_  = MatrixXd(n_x_, 2*n_aug_ + 1);
  
  // define augmented unit sigma-points
  Xsig_aug_ = MatrixXd(n_aug_, 2*n_aug_ + 1);
  MatrixXd I = MatrixXd::Identity(n_aug_, n_aug_);
  Xsig_aug_.col(0) = VectorXd::Zero(n_aug_);
  Xsig_aug_.block(0, 1, n_aug_, n_aug_) = lambda_*I;
  Xsig_aug_.block(0, n_aug_+1, n_aug_, n_aug_) = -lambda_*I;

  // initial state vector
  x_ = VectorXd(n_x_);
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // TODO: set these values meaningfully! Use NIS plots to verify.
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // augmented state mean
  x_aug = VectorXd::Zero(n_aug_);
  // augmented state covariance
  P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug(n_x_, n_x_) = std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
   * Push sigma-points through the CTRV model.
  */

  // create augmented state mean and covariance
  x_aug.head(n_x_) = x_;
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  
  // create augmented sigma-points
  MatrixXd X = x_aug.rowwise().replicate(2*n_aug_ + 1) + P_aug*Xsig_aug_;
  
  //predict sigma points
  for (int i = 0; i <= 2*n_aug_; ++i) {
    // pull out state dimensions for convenience
    double v = X(2, i);
    double psi = X(3, i);
    double psid = X(4, i);
    double nu_a = X(5, i);
    double nu_p = X(6, i);

    // handle zero yaw rate
    if (psid != 0) {
        Xsig_pred_(0, i) = (v/psid) * (sin(psi + psid*delta_t) - sin(psi));
        Xsig_pred_(1, i) = (v/psid) * (-cos(psi + psid*delta_t) + cos(psi));
    } else {
        Xsig_pred_(0, i) = v*delta_t*cos(psi);
        Xsig_pred_(1, i) = v*delta_t*sin(psi);
    }

    // add the process noise contribution
    Xsig_pred_(0, i) += 0.5*pow(delta_t, 2)*cos(psi)*nu_a;
    Xsig_pred_(1, i) += 0.5*pow(delta_t, 2)*sin(psi)*nu_a;
    Xsig_pred_(2, i) = delta_t*nu_a;
    Xsig_pred_(3, i) = psid*delta_t + 0.5*pow(delta_t, 2)*nu_p;
    Xsig_pred_(4, i) = delta_t*nu_p;
    //
    Xsig_pred_.col(i) += X.col(i).head(5);
  }

  // TODO: use predicted SPs to calculate predicted mean and covariance of the state
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  VectorXd mz = VectorXd(3U);
  // compute predicted measurement and safeguard against division by zero
  double norm = sqrt(pow(x_[0], 2) + pow(x_[1], 2)) + DBL_EPSILON;
  mz << norm,
        atan2(x_[1], x_[0]),
        (x_[0]*x_[2] + x_[1]*x_[3])/norm;

  VectorXd e = meas_package.raw_measurements_ - mz;
  // keep difference in angles between -pi and pi
  double temp = e[1] / (2*M_PI);
  if (abs(temp) > 1) {
      unsigned int pi_count = floor(temp);
      e[1] -= 2*pi_count*M_PI;
  }
  MatrixXd Pz = H_*P_*H_.transpose() + R_;
  MatrixXd Pzx = H_*P_;
  MatrixXd K = Pz.ldlt().solve(Pzx).transpose();
  x_ = x_ + K*e;
  P_ = (I_ - K*H_)*P_;

}
