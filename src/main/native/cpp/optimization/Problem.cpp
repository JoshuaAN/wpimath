// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/optimization/Problem.h"

#include <stdexcept>
#include <string>

#include "Eigen/QR"
#include "wpimath/MathShared.h"

using namespace frc;

void Problem::Minimize(const Variable<1, 1>& cost) {
  m_f = cost.GetAutodiffWrapper(0, 0).GetAutodiff();
}

void Problem::Minimize(Variable<1, 1>&& cost) {
  m_f = std::move(cost.GetAutodiffWrapper(0, 0).GetAutodiff());
}

void Problem::Solve() {
  // Create the initial value column vector
  Eigen::VectorXd x{m_leaves.size(), 1};
  for (size_t i = 0; i < m_leaves.size(); ++i) {
    x(i) = val(m_leaves[i]);
  }

  // Solve the optimization problem
  Eigen::VectorXd solution;
  if (m_inequalityConstraints.empty() && m_equalityConstraints.empty()) {
    // TODO: Implement faster unconstrained solver than gradient descent. SQP
    // beats gradient descent at the moment.
    solution = SQP(x);
  } else {
    solution = SQP(x);
  }

  // Assign solution to the original AutodiffWrapper instances
  SetLeaves(solution);
}

void Problem::SetLeaves(const Eigen::Ref<const Eigen::VectorXd>& x) {
  for (size_t i = 0; i < m_leaves.size(); ++i) {
    m_leaves[i].update(x(i));
  }
}

double Problem::f(const Eigen::Ref<const Eigen::VectorXd>& x) {
  SetLeaves(x);
  return val(m_f);
}

Eigen::Map<autodiff::VectorXvar> Problem::MakeVectorAutodiff(
    wpi::SmallVector<autodiff::var>& vec) {
  return Eigen::Map<autodiff::VectorXvar>{vec.data(), Eigen::Index(vec.size()),
                                          1};
}

double Problem::BacktrackingLineSearch(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& gradient) {
  // [1] https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm

  double m = gradient.dot(gradient);  // gradiental derivative

  constexpr double c = 0.1;    // [0, 1]
  constexpr double tau = 0.9;  // [0, 1]
  double alpha = 0.01;         // > 0

  double t = -c * m;
  double f_x = f(x);
  while (f_x - f(x + alpha * gradient) < alpha * t) {
    alpha *= tau;
  }

  // Perform gradient descent with the step size alpha
  return alpha;
}

Eigen::VectorXd Problem::GradientDescent(
    const Eigen::Ref<const Eigen::VectorXd>& x) {
  constexpr double kConvergenceTolerance = 1E-4;

  auto xAD = MakeVectorAutodiff(m_leaves);

  Eigen::VectorXd lastX = x;
  Eigen::VectorXd currentX = x;
  while (true) {
    SetLeaves(lastX);
    m_f.update();
    Eigen::VectorXd g = gradient(m_f, xAD);

    currentX = lastX - g * BacktrackingLineSearch(lastX, g);

    if ((currentX - lastX).norm() < kConvergenceTolerance) {
      return currentX;
    }
    lastX = currentX;
  }
}

Eigen::VectorXd Problem::SQP(const Eigen::Ref<const Eigen::VectorXd>& x) {
  // The equality-constrained quadratic programming problem is defined as
  //
  //        min f(x)
  //         x
  // subject to c(x) = 0
  //
  // The Lagrangian for this problem is
  //
  // L(x, λ) = f(x) − λᵀc(x)
  //
  // The Jacobian of the equality constraints is
  //
  //         [∇ᵀc₁(x)ₖ]
  // A(x)ₖ = [∇ᵀc₂(x)ₖ]
  //         [   ⋮    ]
  //         [∇ᵀcₘ(x)ₖ]
  //
  // The first-order KKT conditions of the equality-constrained problem are
  //
  // F(x, λ) = [∇f(x) − A(x)ᵀλ] = 0
  //           [     c(x)     ]
  //
  // The Jacobian of the KKT conditions with respect to x and λ is given by
  //
  // F'(x, λ) = [∇²ₓₓL(x, λ)  −A(x)ᵀ]
  //            [   A(x)        0   ]
  //
  // Let H(x) = ∇²ₓₓL(x, λ).
  //
  // F'(x, λ) = [H(x)  −A(x)ᵀ]
  //            [A(x)    0   ]
  //
  // The Newton step from the iterate (xₖ, λₖ) is given by
  //
  // [xₖ₊₁] = [xₖ] + [p_k]
  // [λₖ₊₁]   [λₖ]   [p_λ]
  //
  // where p_k and p_λ solve the Newton-KKT system
  //
  // [H(x)ₖ  −A(x)ₖᵀ][pₖ ] = [−∇f(x)ₖ + A(x)ₖᵀλₖ]
  // [A(x)ₖ     0   ][p_λ]   [      −c(x)ₖ      ]
  //
  // Subtracting A(x)ₖᵀλₖ from both sides of the first equation, we get
  //
  // [H(x)ₖ  −A(x)ₖᵀ][pₖ  ] = [−∇f(x)ₖ]
  // [A(x)ₖ     0   ][λₖ₊₁]   [ −c(x)ₖ]
  //
  // [1] Nocedal, J. and Wright, S. Numerical Optimization, 2nd. ed., Ch. 18.
  //     Springer, 2006.

  using namespace autodiff;

  constexpr double kConvergenceTolerance = 1E-4;

  // autodiff vector of the equality constraints c(x)
  auto c = MakeVectorAutodiff(m_equalityConstraints);

  // Lagrange multipliers for the equality constraints
  autodiff::VectorXvar lambda{m_equalityConstraints.size(), 1};

  // L(x, λ)ₖ = f(x)ₖ − λₖᵀc(x)ₖ
  autodiff::var L = m_f - (lambda.transpose() * c)(0);

  auto xAD = MakeVectorAutodiff(m_leaves);
  Eigen::VectorXd lastX = x;
  Eigen::VectorXd currentX = x;
  while (true) {
    SetLeaves(lastX);
    L.update();

    // Hₖ = ∇²ₓₓL(x, λ)ₖ
    Eigen::MatrixXd H = hessian(L, xAD);

    //         [∇ᵀc₁(x)ₖ]
    // A(x)ₖ = [∇ᵀc₂(x)ₖ]
    //         [   ⋮    ]
    //         [∇ᵀcₘ(x)ₖ]
    Eigen::MatrixXd A{m_equalityConstraints.size(), x.rows()};
    for (int row = 0; row < A.rows(); ++row) {
      A.block(row, 0, 1, x.rows()) = gradient(c(row), xAD).transpose();
    }

    // Confirm the equality constraint Jacobian A(x) has full row rank
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{A};
    if (qr.rank() < A.rows()) {
      std::string msg =
          "The equality constraints are not linearly independent!";
      wpi::math::MathSharedStore::ReportError(msg);
      throw std::invalid_argument(msg);
    }

    // F'(x, λ) = [H(x)ₖ  −A(x)ₖᵀ]
    //            [A(x)ₖ     0   ]
    Eigen::MatrixXd Fprime{H.rows() + A.rows(), H.cols() + A.rows()};
    Fprime.block(0, 0, H.rows(), H.cols()) = H;
    Fprime.block(0, H.cols(), A.cols(), A.rows()) = -A.transpose();
    Fprime.block(H.rows(), 0, A.rows(), A.cols()) = A;
    Fprime.block(H.rows(), H.cols(), A.rows(), A.rows()).setZero();

    // [−∇f(x)ₖ]
    // [ −c(x)ₖ]
    Eigen::MatrixXd rhs{x.rows() + A.rows(), 1};
    rhs.block(0, 0, x.rows(), 1) = -gradient(m_f, xAD);
    for (int row = 0; row < c.rows(); ++row) {
      rhs(x.rows() + row, 0) = -val(c(row));
    }

    // Solve the Newton-KKT system
    Eigen::VectorXd step = Fprime.householderQr().solve(rhs);

    currentX = lastX + step.block(0, 0, x.rows(), 1);
    lambda = step.block(x.rows(), 0, lambda.rows(), 1);

    if ((currentX - lastX).norm() < kConvergenceTolerance) {
      return currentX;
    }
    lastX = currentX;
  }
}
