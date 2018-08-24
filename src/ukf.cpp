#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>
#include <cfloat>
#include <iomanip>

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
  // weights for sigma-points in augmented state-space
  lambda_ = max(3 - n_x_, 0); // 3 - n_x_;
  weights_ = VectorXd::Ones(2*n_x_ + 1) / (2*(n_x_ + lambda_));
  weights_[0] = lambda_ / (n_x_ + lambda_);
  // define unit sigma-points
  Xsig_ = MatrixXd(n_x_, 2*n_x_ + 1);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  Xsig_.col(0) = VectorXd::Zero(n_x_);
  double c = sqrt(n_x_ + lambda_);
  Xsig_.block(0, 1, n_x_, n_x_) = c*I;
  Xsig_.block(0, n_x_+1, n_x_, n_x_) = -c*I;

  // augmented state dimension
  n_aug_ = 7;
  // weights for sigma-points in augmented state-space
  lambda_ = max(3 - n_aug_, 0); // 3 - n_aug_;
  weights_aug_ = VectorXd::Ones(2*n_aug_ + 1) / (2*(n_aug_ + lambda_));
  weights_aug_[0] = lambda_ / (n_aug_ + lambda_);
  // define augmented unit sigma-points
  Xsig_aug_ = MatrixXd(n_aug_, 2*n_aug_ + 1);
  I = MatrixXd::Identity(n_aug_, n_aug_);
  Xsig_aug_.col(0) = VectorXd::Zero(n_aug_);
  c = sqrt(n_aug_ + lambda_);
  Xsig_aug_.block(0, 1, n_aug_, n_aug_) = c*I;
  Xsig_aug_.block(0, n_aug_+1, n_aug_, n_aug_) = -c*I;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);
  x_ << 0.6, 0.6, 5.5, 0, 0;  // init from dataset
  // initial covariance matrix
  P_ = 0.01*MatrixXd::Identity(n_x_, n_x_);  // low variance => I'm pretty sure by the init values

  // TODO: set these values meaningfully! Use NIS plots to verify.
  // Process noise standard deviation longitudinal acceleration in m/s^2
  // 3.5 is max bicycle acceleration according to: https://www.analyticcycling.com/DiffEqMotionFunctions_Page.html
  std_a_ =  1.75;
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4; // 0.2;

  // augmented state mean
  x_aug_ = VectorXd::Zero(n_aug_);
  // augmented state covariance
  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_x_, n_x_) = pow(std_a_, 2);
  P_aug_(n_x_+1, n_x_+1) = pow(std_yawdd_, 2);
  
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
  
  R_lidar_ = MatrixXd::Zero(2, 2);
  R_lidar_(0, 0) = pow(std_laspx_, 2);
  R_lidar_(1, 1) = pow(std_laspy_, 2);
  
  R_radar_ = MatrixXd::Zero(3, 3);
  R_radar_(0, 0) = pow(std_radr_, 2);
  R_radar_(1, 1) = pow(std_radphi_, 2);
  R_radar_(2, 2) = pow(std_radrd_, 2);

  radar_nis_count_ = 0;
  lidar_nis_count_ = 0;
  radar_total_ = 0;
  lidar_total_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    // first measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.

      This seemingly ad-hocy conversion happens because we are initializing state with measurements.
      Hence the need for inverse transform from polar (measurement space) to cartesian (state space).
      */
      double rho = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      x_[0] = rho * cos(theta);
      x_[1] = rho * sin(theta);

      radar_total_++;  // radar measurement counter
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_[0] = meas_package.raw_measurements_[0];
      x_[1] = meas_package.raw_measurements_[1];

      lidar_total_++;  // lidar measurement counter
    }
    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    cout << "Percent of measurements greater than the chi^2 value (p=0.05)" << endl;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double dt = (meas_package.timestamp_ - previous_timestamp_) / 1e6;
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) 
  {
    // Radar updates
    UpdateRadar(meas_package);
  } 
  else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) 
  {
    // Laser updates
    UpdateLidar(meas_package);
  }

  // 5% of measurements should be above the threshold after processing the whole trajectory
  // cout << "Percent of measurement above threshold:"
  if ((lidar_total_ + radar_total_) % 20 == 0)
  {
    cout << "LIDAR: " << setprecision(2) 
         << 100 * float(lidar_nis_count_) / lidar_total_ << " \t"
         << "RADAR: " << setprecision(2) 
         << 100 * float(radar_nis_count_) / radar_total_ << endl;
  }
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
  x_aug_.head(n_x_) = x_;
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  
  // create augmented sigma-points
  unsigned int num_points = 2*n_aug_ + 1;
  MatrixXd L_aug = P_aug_.llt().matrixL();
  MatrixXd X = x_aug_.rowwise().replicate(num_points) + L_aug*Xsig_aug_;
  
  //predict sigma points
  MatrixXd Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);
  for (unsigned int i = 0; i < num_points; ++i) {
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
    Xsig_pred_.col(i) += X.col(i).head(n_x_);
  }

  // predicted mean
  x_ = Xsig_pred_ * weights_aug_;

  // predicted covariance
  // adding process covariance not necessary for non-additive case (already reflected in the transform)
  MatrixXd df = Xsig_pred_ - x_.rowwise().replicate(num_points);
  // TODO: normalize difference between angles to [-PI, PI]
  P_.fill(0.0);  // important so we don't add on top of the old values from previous time step
  for (unsigned int i = 0; i < num_points; ++i)
  {
    P_ += weights_aug_[i] * (df.col(i) * df.col(i).transpose());
  }
  
  // cout << "x_" << endl << x_ << endl;
  // cout << "P_" << endl << P_ << endl;  
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
    You'll also need to calculate the lidar NIS.
  */

  // push sigma-points through the LIDAR measurement model
  unsigned int num_points = 2*n_x_ + 1;
  MatrixXd H = MatrixXd::Identity(2, n_x_);
  MatrixXd L = P_.llt().matrixL();
  MatrixXd X = x_.rowwise().replicate(num_points) + L*Xsig_;
  MatrixXd Xsig_lidar = H * X;

  // compute measurement moments
  VectorXd mz = Xsig_lidar * weights_;
  MatrixXd Pz = MatrixXd::Zero(2, 2);
  MatrixXd Pzx = MatrixXd::Zero(2, n_x_);
  MatrixXd dh = Xsig_lidar - mz.rowwise().replicate(num_points);
  MatrixXd dx = X - x_.rowwise().replicate(num_points);
  for (unsigned int i = 0; i < num_points; ++i) {
     Pz  += weights_(i) * (dh.col(i) * dh.col(i).transpose());
     Pzx += weights_(i) * (dh.col(i) * dx.col(i).transpose());
  }
  // add lidar measurement covariance
  Pz += R_lidar_;

  // Cholesky factor for semi-definite matrices (because covariances are semi-definite)
  Eigen::LDLT<MatrixXd> cholPz(Pz);
  // Kalman gain
  MatrixXd K = cholPz.solve(Pzx).transpose();
  VectorXd dz = meas_package.raw_measurements_ - mz;
  // measurement update
  x_ = x_ + K * dz;
  P_ = P_ - K * Pz * K.transpose();

  // NIS computation
  lidar_total_++;  // lidar measuremnt counter
  // if NIS higher than threshold, count it
  if (dz.transpose() * (cholPz.solve(dz)) > NIS_LIDAR_) lidar_nis_count_++;

  // cout << "x_ " << endl << x_ << endl;
  // cout << "P_ " << endl << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
    You'll also need to calculate the radar NIS.
  */
  
  // push sigma-points through RADAR measurement model
  unsigned int num_points = 2*n_x_ + 1;
  MatrixXd Xsig_radar = MatrixXd(3, num_points);
  
  // compute sigma-points from unit sigma-points
  MatrixXd L = P_.llt().matrixL();
  MatrixXd X = x_.rowwise().replicate(num_points) + L*Xsig_;
  for (unsigned int i = 0; i < num_points; ++i) 
  {
    VectorXd x_i = X.col(i);
    double norm = sqrt(pow(x_i[0], 2) + pow(x_i[1], 2)) + DBL_EPSILON;
    Xsig_radar.col(i) << norm, atan2(x_i[1], x_i[0]), x_i[2]*(x_i[0]*cos(x_i[3]) + x_i[1]*sin(x_i[3]))/norm;
  }

  // predicted measurement moments
  MatrixXd Pz = MatrixXd::Zero(3, 3);
  MatrixXd Pzx = MatrixXd::Zero(3, n_x_);
  VectorXd mz = Xsig_radar * weights_;
  MatrixXd dh = Xsig_radar - mz.rowwise().replicate(num_points);
  MatrixXd dx = X - x_.rowwise().replicate(num_points);

  // TODO: normalize difference between angles to [-PI, PI]
  // if ((dx.array().row(3) > M_PI).any() || (dx.array().row(3) < -M_PI).any())
  // {
  //   cout << "Yaw = " << dx.row(3).maxCoeff() << endl;
  //   for (unsigned int i = 0; i < dx.cols(); ++i)
  //   {
  //     // if (dx.row(3)(i) > M_PI || dx.row(3)(i) < -M_PI)
  //     double temp = dx.row(3)(i) / (2*M_PI);
  //     if (abs(temp) > 1) 
  //     {
  //         unsigned int pi_count = floor(temp);
  //         dx.row(3)(i) -= 2*pi_count*M_PI;
  //     }
  //   }
  // }

  // compute predicted measurement moments
  for (unsigned int i = 0; i < num_points; ++i) 
  {
    Pz  += weights_(i) * (dh.col(i) * dh.col(i).transpose());
    Pzx += weights_(i) * (dh.col(i) * dx.col(i).transpose());
  }
  // add radar measurement covariance
  Pz += R_radar_;

  // innovation with normalization of angles
  VectorXd dz = meas_package.raw_measurements_ - mz;
  // keep difference in angles between -pi and pi
  double temp = dz[1] / (2*M_PI);
  if (abs(temp) > 1) 
  {
      unsigned int pi_count = floor(temp);
      dz[1] -= 2*pi_count*M_PI;
  }
  // Cholesky factor for semi-definite matrices (because covariances are semi-definite)
  Eigen::LDLT<MatrixXd> cholPz(Pz);
  // Kalman gain
  MatrixXd K = cholPz.solve(Pzx).transpose();
  // measurement update
  x_ = x_ + K * dz;
  P_ = P_ - K * Pz * K.transpose();

  // NIS computation
  radar_total_++;  // radar measuremnt counter
  // if NIS higher than threshold, count it
  if (dz.transpose() * (cholPz.solve(dz)) > NIS_RADAR_) radar_nis_count_++;

  // cout << "lz^2: " << dz.transpose() * (cholPz.solve(dz)) 
  //      << " LDLT: " << dz.transpose() * (Pz.ldlt().solve(dz)) 
  //      << " LLT: " << dz.transpose() * (Pz.llt().solve(dz)) 
  //      << " inv: " << dz.transpose() * (Pz.inverse() * dz) << endl;

  // cout << "radar NIS: " << lz.transpose() * lz 
  //      << " radar_nis_count: " << radar_nis_count_ 
  //      << " radar_total: " << radar_total_ << endl;

  // cout << "x_ " << endl << x_ << endl;
  // cout << "P_ " << endl << P_ << endl;
}
