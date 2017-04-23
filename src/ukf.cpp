#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
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
    std_a_      = 1.0; // orig 30; tested with 0.6 - 3

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_  = 0.59;  // orig 30;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_  = 0.15; // from EKF project

    // Laser measurement noise standard deviation position2 in m
    std_laspy_  = 0.15; // from EKF project

    // Radar measurement noise standard deviation radius in m
    std_radr_   = 0.3;  // from EKF project

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03; // from EKF project

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_  = 0.3;  // from EKF project

    is_initialized_ = false;

    previous_timestamp_ = 0;

    // Sigma points
    n_x_    = 5;
    n_aug_  = n_x_ + 2;
    lambda_ = 3 - n_aug_;
    n_sig_  = 2 * n_aug_ + 1;   // 15 sigma points

    // init state vector
    x_      = VectorXd(n_x_);       // size 5
    x_aug_   = VectorXd(n_aug_);     // size 7

    // Noise covar matrix
    Q_      = MatrixXd(2, 2);       // 2x2
    Q_  <<  std_a_ * std_a_, 0,
            0, std_yawdd_ * std_yawdd_;

    // init covar matrix
    P_      = MatrixXd(n_x_, n_x_);   // 5x5
    P_ <<   1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1000, 0, 0,
            0, 0, 0, 100, 0,
            0, 0, 0, 0, 1;

    P_aug_   = MatrixXd(n_aug_, n_aug_);  // 7x7
    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(n_x_, n_x_) = P_;
    P_aug_.bottomRightCorner(2, 2)   = Q_;

    // init weights_
    weights_    = VectorXd(n_sig_);   // 15 sigma points
    weights_(0) = double(lambda_ / (lambda_ + n_aug_));
    for (int i=1; i < 2*n_aug_ + 1; i++)
    {
        weights_(i) = double(0.5 / (lambda_ + n_aug_));
    }

    // sigma points matrix
    Xsig_aug_   = MatrixXd(n_aug_, n_sig_); // 7x15
    Xsig_pred_  = MatrixXd(n_x_, n_sig_);   // 5x15

    // For Radar Updates
    n_z_radar_  = 3;

    R_radar_    = MatrixXd(n_z_radar_, n_z_radar_);
    R_radar_  <<  std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;

    // Laser Updates
    n_z_laser_ = 2;


    R_laser_    = MatrixXd(2, 2);
    R_laser_  <<  std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;

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
    if (!is_initialized_) {
        double px = 0;
        double py = 0;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            double rhodot = meas_package.raw_measurements_(2);

            px = rho * cos(phi);
            py = rho * sin(phi);

            double vx = rhodot * cos(phi);
            double vy = rhodot * sin(phi);

            if(fabs(px) < 0.0001) {
                px        = 1;
                P_(0, 0)  = 100;
            }
            if(fabs(py) < 0.0001) {
                py        = 1;
                P_(1,1)   = 100;
            }

            x_ << px, py, rhodot, 0.0, 0.0;
        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            if (meas_package.raw_measurements_[0] == 0 or meas_package.raw_measurements_[1] == 0) {
                return;
            }
            px = meas_package.raw_measurements_(0);
            py = meas_package.raw_measurements_(1);

            x_ << px, py, 0.0, 0.0, 0.0;
        }
        previous_timestamp_ = meas_package.timestamp_;
        x_aug_ << x_.array(), 0.0, 0.0;
        is_initialized_ = true;
        std::cout << "Initialized" << std::endl;
        return;
    }

    float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = meas_package.timestamp_;

    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        // 4a -
        UpdateRadar(meas_package);

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        // Laser updates
        // 4b -
        UpdateLidar(meas_package);
    }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    x_aug_.head(5) = x_;
    x_aug_(5) = 0.0;
    x_aug_(6) = 0.0;

    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(n_x_, n_x_) = P_;
    P_aug_.bottomRightCorner(2, 2) = Q_;

    MatrixXd A = P_aug_.llt().matrixL();

    Xsig_aug_.col(0) = x_aug_;

    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug_.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i);
        Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i);
    }

    for (int i = 0; i < n_sig_; i++) {
        double p_x = Xsig_aug_(0, i);
        double p_y = Xsig_aug_(1, i);
        double v = Xsig_aug_(2, i);
        double yaw = Xsig_aug_(3, i);
        double yawd = Xsig_aug_(4, i);
        double nu_a = Xsig_aug_(5, i);
        double nu_yawdd = Xsig_aug_(6, i);

        double px_p, py_p;

        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }
        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        // 2f - write predicted sigma points in the columns
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
    x_.fill(0.0);
    x_      = Xsig_pred_ * weights_;

    P_.fill(0.0);
    for (int i = 0; i < n_sig_; i++)  // over 2n+1 sigma points
    {
        // 3d - state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // 3e - angle norm
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        // 3f - update P_
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    MatrixXd Zsig = MatrixXd(n_z_laser_, n_sig_);
    for (int i = 0; i < n_sig_; i++) {
        Zsig(0, i) = Xsig_pred_(0, i); // px
        Zsig(1, i) = Xsig_pred_(1, i); // py
    }

    VectorXd z_pred = VectorXd(n_z_laser_);
    z_pred.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    MatrixXd S = MatrixXd(n_z_laser_, n_z_laser_);
    S.fill(0.0);

    MatrixXd Tc = MatrixXd(n_x_, n_z_laser_);   // dim 5x2
    Tc.fill(0.0);

    for (int i = 0; i < n_sig_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights_(i) * z_diff * z_diff.transpose();

        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    S         = S + R_laser_;

    MatrixXd K = Tc * S.inverse();

    VectorXd z_real = meas_package.raw_measurements_; // size 3
    VectorXd z_diff = z_real - z_pred;

    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    MatrixXd Zsig = MatrixXd(n_z_radar_, n_sig_);
    for (int i = 0; i < n_sig_; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double cosv = cos(yaw) * v;
        double sinv = sin(yaw) * v;

        double rho1   = sqrt( pow(p_x, 2) + pow(p_y, 2) );
        double phi    = 0.0;

        if (fabs(p_x) > 0.001) {
            phi = atan2(p_y, p_x);
        }
        double rhodot = 0.0;
        // rhodot = (p_x * cosv + p_y * sinv) / rho1;
        if (fabs(rho1) > 0.001) {
            rhodot = (p_x * cosv + p_y * sinv) / rho1;
        }

        Zsig(0, i)   = rho1;   // rho1;               // rho
        Zsig(1, i)   = phi;    // atan2(p_y, p_x);    // phi
        Zsig(2, i)   = rhodot; // (p_x * cosv + p_y * sinv) / rho1; // rhodot
    }

    VectorXd z_pred = VectorXd(n_z_radar_);
    z_pred.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_); // dim 3x3
    S.fill(0.0);

    MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);   // dim 5x3
    Tc.fill(0.0);

    for (int i = 0; i < n_sig_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();

        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    S = S + R_radar_;

    MatrixXd K = Tc * S.inverse();

    VectorXd z_real = meas_package.raw_measurements_; // size 3
    VectorXd z_diff = z_real - z_pred;

    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
