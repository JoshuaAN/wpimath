// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/EigenCore.h"
#include "frc/optimization/Problem.h"

#include <iostream>
#include <typeinfo>

//std::vector<double> Range(double start, double end, double step) {
//    std::vector<double> ret;
//
//    for (double i = start; i < end; i += step) {
//        ret.emplace_back(i);
//    }
//
//    return ret;
//}
//
//int main() {
//    frc::Problem problem;
//
//    // Scalar zero init
//    auto x = problem.Var();
//    EXPECT_DOUBLE_EQ(0.0, x.Value());
//
//    // Scalar assignment
//    x = 2.0;
//    EXPECT_DOUBLE_EQ(2.0, x.Value());
//
//    // Vector zero init
//    auto y = problem.Var<2>();
//    EXPECT_DOUBLE_EQ(0.0, y.Value(0));
//    EXPECT_DOUBLE_EQ(0.0, y.Value(1));
//
//    // Vector assignment
//    y(0) = 1.0;
//    y(1) = 2.0;
//    EXPECT_DOUBLE_EQ(1.0, y.Value(0));
//    EXPECT_DOUBLE_EQ(2.0, y.Value(1));
//
//    // Matrix zero init
//    auto z = problem.Var<3, 2>();
//    EXPECT_DOUBLE_EQ(0.0, z.Value(0, 0));
//    EXPECT_DOUBLE_EQ(0.0, z.Value(0, 1));
//    EXPECT_DOUBLE_EQ(0.0, z.Value(1, 0));
//    EXPECT_DOUBLE_EQ(0.0, z.Value(1, 1));
//    EXPECT_DOUBLE_EQ(0.0, z.Value(2, 0));
//    EXPECT_DOUBLE_EQ(0.0, z.Value(2, 1));
//
//    // Matrix assignment; element comparison
//    z = frc::Matrixd<3, 2>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
//    EXPECT_DOUBLE_EQ(1.0, z.Value(0, 0));
//    EXPECT_DOUBLE_EQ(2.0, z.Value(0, 1));
//    EXPECT_DOUBLE_EQ(3.0, z.Value(1, 0));
//    EXPECT_DOUBLE_EQ(4.0, z.Value(1, 1));
//    EXPECT_DOUBLE_EQ(5.0, z.Value(2, 0));
//    EXPECT_DOUBLE_EQ(6.0, z.Value(2, 1));
//
//    // Matrix assignment; matrix comparison
//    {
//    frc::Matrixd<3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
//    z = expected;
//    EXPECT_EQ(expected, z.Value());
//    }
//
//    // Block assignment
//    {
//    frc::Matrixd<3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
//
//    frc::Matrixd<2, 1> expectedBlock{{1.0}, {1.0}};
//    z.Block<2, 1>(0, 0) = expectedBlock;
//    expected.block<2, 1>(0, 0) = expectedBlock;
//
//    EXPECT_EQ(expected, z.Value());
//
//    frc::Matrixd<3, 2> expectedResult{{1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
//    EXPECT_EQ(expectedResult, z.Value());
//    }
//}

int main() {
    // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    frc::Problem problem;
//
//    autodiff::var z = 2.0;
//    autodiff::var x = 3.0;
//    z = 0.5;
//    auto y = problem.Var();
//    y = 3.30;

//    std::cout << z << std::endl;
//    std::cout << x << std::endl;

//    problem.Minimize(frc::pow(1 - x, 2) + 100 * frc::pow(y - frc::pow(x, 2), 2));
//    problem.Minimize(z * (z - 1) + y * (y - 1));
//    problem.SubjectTo(z == 5);

//    problem.SubjectTo(frc::pow(x - 1, 3) - y + 1 <= 0);
//    problem.SubjectTo(x + y - 2 <= 0);

//    problem.Solve();

    Eigen::MatrixXd x(4,4);

    x(0, 0) = 2;
    x(1, 0) = 4;
    x(2, 0) = -2;
    x(3, 0) = 2;
    x(0, 1) = 4;
    x(1, 1) = 9;
    x(2, 1) = -1;
    x(3, 1) = 6;
    x(0, 2) = -2;
    x(1, 2) = -1;
    x(2, 2) = 14;
    x(3, 2) = 13;
    x(0, 3) = 2;
    x(1, 3) = 6;
    x(2, 3) = 13;
    x(3, 3) = 35;

    std::cout << problem.ModifiedCholeskyFactorization(x);

}


//TEST(ProblemTest, QuadraticUnconstrained2) {
//    {
//    frc::Problem problem;
//
//    auto x = problem.Var();
//    x = 1.0;
//    auto y = problem.Var();
//    y = 2.0;
//
//    problem.Minimize(x * x + y * y);
//
//    problem.Solve();
//
//    EXPECT_NEAR(0.0, x.Value(), 1e-2);
//    EXPECT_NEAR(0.0, y.Value(), 1e-2);
//    }
//    {
//    frc::Problem problem;
//
//    auto x = problem.Var<2>();
//    x(0) = 1.0;
//    x(1) = 2.0;
//
//    problem.Minimize(x.transpose() * x);
//
//    problem.Solve();
//
//    EXPECT_NEAR(0.0, x.Value(0), 1e-2);
//    EXPECT_NEAR(0.0, x.Value(1), 1e-2);
//    }
//}
//
//TEST(ProblemTest, QuadraticEqualityConstrained) {
//    // Maximize xy subject to x + 3y = 36.
//    //
//    // Maximize f(x,y) = xy
//    // subject to g(x,y) = x + 3y - 36 = 0
//    //
//    //         value func  constraint
//    //              |          |
//    //              v          v
//    // L(x,y,λ) = f(x,y) - λg(x,y)
//    // L(x,y,λ) = xy - λ(x + 3y - 36)
//    // L(x,y,λ) = xy - xλ - 3yλ + 36λ
//    //
//    // ∇_x,y,λ L(x,y,λ) = 0
//    //
//    // ∂L/∂x = y - λ
//    // ∂L/∂y = x - 3λ
//    // ∂L/∂λ = -x - 3y + 36
//    //
//    //  0x + 1y - 1λ = 0
//    //  1x + 0y - 3λ = 0
//    // -1x - 3y + 0λ + 36 = 0
//    //
//    // [ 0  1 -1][x]   [  0]
//    // [ 1  0 -3][y] = [  0]
//    // [-1 -3  0][λ]   [-36]
//    //
//    // Solve with:
//    // ```python
//    //   np.linalg.solve(
//    //     np.array([[0,1,-1],
//    //               [1,0,-3],
//    //               [-1,-3,0]]),
//    //     np.array([[0], [0], [-36]]))
//    // ```
//    //
//    // [x]   [18]
//    // [y] = [ 6]
//    // [λ]   [ 6]
//    {
//    frc::Problem problem;
//
//    auto x = problem.Var();
//    auto y = problem.Var();
//
//    // Maximize xy
//    problem.Minimize(-x * y);
//
//    problem.SubjectTo(x + 3 * y == 36);
//
//    problem.Solve();
//
//    EXPECT_DOUBLE_EQ(18.0, x.Value());
//    EXPECT_DOUBLE_EQ(6.0, y.Value());
//    }
//
//{
//frc::Problem problem;
//
//auto x = problem.Var<2>();
//x(0) = 1.0;
//x(1) = 2.0;
//
//problem.Minimize(x.transpose() * x);
//
//problem.SubjectTo(x == frc::Matrixd<2, 1>{{3.0, 3.0}});
//
//problem.Solve();
//
//EXPECT_NEAR(3.0, x.Value(0), 1e-2);
//EXPECT_NEAR(3.0, x.Value(1), 1e-2);
//}
//}
//
//TEST(ProblemTest, RosenbrockConstrainedWithCubicAndLine) {
//// https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
//for (auto x0 : Range(-1.5, 1.5, 0.1)) {
//for (auto y0 : Range(-0.5, 2.5, 0.1)) {
//frc::Problem problem;
//
//auto x = problem.Var();
//x = x0;
//auto y = problem.Var();
//y = y0;
//
//problem.Minimize(frc::pow(1 - x, 2) +
//100 * frc::pow(y - frc::pow(x, 2), 2));
//
//problem.SubjectTo(frc::pow(x - 1, 3) - y + 1 <= 0);
//problem.SubjectTo(x + y - 2 <= 0);
//
//problem.Solve();
//
//EXPECT_NEAR(1.0, x.Value(), 1e-2);
//EXPECT_NEAR(1.0, y.Value(), 1e-2);
//}
//}
//}
//
//TEST(ProblemTest, RosenbrockConstrainedToDisk) {
//// https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
//for (auto x0 : Range(-1.5, 1.5, 0.1)) {
//for (auto y0 : Range(-1.5, 1.5, 0.1)) {
//frc::Problem problem;
//
//auto x = problem.Var();
//x = x0;
//auto y = problem.Var();
//y = y0;
//
//problem.Minimize(frc::pow(1 - x, 2) +
//100 * frc::pow(y - frc::pow(x, 2), 2));
//
//problem.SubjectTo(frc::pow(x, 2) + frc::pow(y, 2) <= 2);
//
//problem.Solve();
//
//EXPECT_NEAR(1.0, x.Value(), 1e-2);
//EXPECT_NEAR(1.0, y.Value(), 1e-2);
//}
//}
//}
//
//TEST(ProblemTest, FlywheelDirectTranscription) {
//constexpr double T = 5.0;
//constexpr auto dt = 5_ms;
//constexpr int N = T / dt.value();
//
//frc::Matrixd<1, 1> x0{0.0};
//frc::Matrixd<1, 1> r{10.0};
//
//// Flywheel model:
//// States: [velocity]
//// Inputs: [voltage]
//auto system = frc::LinearSystemId::IdentifyVelocitySystem<units::radians>(
//        1_V / 1_rad_per_s, 1_V / 1_rad_per_s_sq);
//Eigen::Matrix<double, 1, 1> A;
//Eigen::Matrix<double, 1, 1> B;
//frc::DiscretizeAB<1, 1>(system.A(), system.B(), dt, &A, &B);
//
//frc::Problem problem;
//auto X = problem.Var<1, N + 1>();
//auto U = problem.Var<1, N>();
//
//// Dynamics constraint
//for (int k = 0; k < N; ++k) {
//problem.SubjectTo(X.Col(k + 1) == A * X.Col(k) + B * U.Col(k));
//}
//
//// State and input constraints
//problem.SubjectTo(X.Col(0) == x0);
//problem.SubjectTo(U >= -12);
//problem.SubjectTo(U <= 12);
//
//// Cost function - minimize error
//frc::Variable<1, 1> J = 0.0;
//for (int k = 0; k < N + 1; ++k) {
//J += ((r - X.Col(k)).transpose() * (r - X.Col(k)));
//}
//problem.Minimize(J);
//
//problem.Solve();
//
//// TODO: Verify solution
//}