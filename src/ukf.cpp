#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

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

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
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
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  lambda_ = sqrt(3);
  n_x_ = 5;
  n_aug_ = 7;
  weights_ = VectorXd::Ones(2*n_aug_) / (2*(n_aug_ + lambda_));
  weights_[0] = lambda_ / (n_x_ + lambda_);
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
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //predict sigma points
  for (int i = 0; i <= 2*n_aug; ++i) {
    double v = Xsig_aug(2, i);
    double psi = Xsig_aug(3, i);
    double psid = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_p = Xsig_aug(6, i);
    if (psid != 0) {
        Xsig_pred(0, i) = (v/psid) * (sin(psi + psid*delta_t) - sin(psi));
        Xsig_pred(1, i) = (v/psid) * (-cos(psi + psid*delta_t) + cos(psi));
    } else {
        Xsig_pred(0, i) = v*delta_t*cos(psi);
        Xsig_pred(1, i) = v*delta_t*sin(psi);
    }
    Xsig_pred(0, i) += 0.5*pow(delta_t, 2)*cos(psi)*nu_a;
    Xsig_pred(1, i) += 0.5*pow(delta_t, 2)*sin(psi)*nu_a;
    // predict the rest of the state vector
    Xsig_pred(2, i) = delta_t*nu_a;
    Xsig_pred(3, i) = psid*delta_t + 0.5*pow(delta_t, 2)*nu_p;
    Xsig_pred(4, i) = delta_t*nu_p;

    Xsig_pred.col(i) += Xsig_aug.col(i).head(5);
  }
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
}
