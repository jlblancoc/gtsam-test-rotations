#include <cmath>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <iostream>

static double deg2rad(double x) { return x * M_PI / 180.0; }

void testRotUncertainty(bool R0_identity) {
  using gtsam::symbol_shorthand::P; // Position
  using gtsam::symbol_shorthand::R; // Rotation

  const double std_large = 1e+2;
  const double std_small = 1e-3;

  gtsam::Rot3 R0;
  if (R0_identity)
    R0 = gtsam::Rot3::identity();
  else
    R0 = gtsam::Rot3::Ypr(deg2rad(0.0), deg2rad(0.0), deg2rad(30.0));

  const auto relRot =
      gtsam::Rot3::Ypr(deg2rad(1.0), deg2rad(2.0), deg2rad(3.0));

  const auto R1 = R0 * relRot;

  auto noiseRelRotLLL = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector3() << std_large, std_large, std_large).finished());
  auto noiseRelRotSSS = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector3() << std_small, std_small, std_small).finished());

  for (int axis = 0; axis < 3; axis++) {
    std::cout << "\n================= AXIS " << axis << "\n";

    gtsam::Matrix33 cov = gtsam::Matrix33::Zero();
    for (int i = 0; i < 3; i++)
      cov(i, i) = (i == axis) ? 1e-6 * std_small : std_large;

    // IMPORTANT: GTSAM rot uncertainty is in the TARGET FRAME reference
    // (Lie group base)
    // cov = relRot.matrix() * cov * relRot.matrix().transpose();
    // cov = relRot.matrix().transpose() * cov * relRot.matrix();

    gtsam::NonlinearFactorGraph fg;
    fg.emplace_shared<gtsam::BetweenFactor<gtsam::Rot3>>(
        R(0), R(1), relRot, gtsam::noiseModel::Gaussian::Covariance(cov));

    fg.emplace_shared<gtsam::PriorFactor<gtsam::Rot3>>(
        R(0), R0, gtsam::noiseModel::Constrained::All(3));
    fg.emplace_shared<gtsam::PriorFactor<gtsam::Rot3>>(
        R(1), gtsam::Rot3::identity(), noiseRelRotSSS);

    gtsam::Values initValues;
    initValues.insert(R(0), gtsam::Rot3::identity());
    initValues.insert(R(1), gtsam::Rot3::identity());

    gtsam::LevenbergMarquardtOptimizer optim(fg, initValues);
    const auto optimValues = optim.optimize();

    fg.print();
    optimValues.print("Final values: ");
    fg.printErrors(optimValues);

    const auto estR0 = optimValues.at<gtsam::Rot3>(R(0));
    const auto estR1 = optimValues.at<gtsam::Rot3>(R(1));

    std::cout << std::fixed << std::setprecision(5);

    std::cout << "r0 (rpy): " << estR0.rpy().transpose() << " localCoords: "
              << gtsam::Rot3::LocalCoordinates(estR0).transpose() << "\n";
    std::cout << "r1      : " << estR1.rpy().transpose() << " localCoords: "
              << gtsam::Rot3::LocalCoordinates(estR1).transpose() << "\n";
    std::cout << "r01 obs : " << relRot.rpy().transpose() << " localCoords: "
              << gtsam::Rot3::LocalCoordinates(relRot).transpose() << "\n";
    std::cout
        << "r01 est : " << (estR0.inverse() * estR1).rpy().transpose()
        << " localCoords: "
        << gtsam::Rot3::LocalCoordinates(estR0.inverse() * estR1).transpose()
        << "\n";
    std::cout << "r01 est rel local coords : "
              << estR0.localCoordinates(estR1).transpose() << "\n";
  }
}

int main() {
  try {
    std::cout << " R0: identity"
                 "\n======================\n";
    testRotUncertainty(true);
    std::cout << " R0: not identity"
                 "\n======================\n";
    testRotUncertainty(false);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
