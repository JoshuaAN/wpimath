// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <utility>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <wpi/SmallVector.h>
#include <wpi/SymbolExports.h>

#include "Eigen/Core"
#include "frc/optimization/AutodiffWrapper.h"
#include "frc/optimization/EqualityConstraint.h"
#include "frc/optimization/InequalityConstraint.h"

namespace frc {

/**
 * Specifies an optimization problem and solves it. Several solvers are
 * supported for different types of problems.
 *
 *
 */
class WPILIB_DLLEXPORT Problem {
 public:
  Problem() = default;

  /**
   * Create a matrix in the optimization problem.
   */
  template <int Rows = 1, int Cols = 1>
  Variable<Rows, Cols> Var() {
    Variable<Rows, Cols> vars;

    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        m_leaves.emplace_back();
        vars.GetAutodiffWrapper(row, col) =
            AutodiffWrapper{this, static_cast<int>(m_leaves.size() - 1)};
      }
    }

    return vars;
  }

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void Minimize(const Variable<1, 1>& cost);

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void Minimize(Variable<1, 1>&& cost);

  /**
   * Tells the solver to solve the problem while obeying the given equality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  template <int Rows, int Cols>
  void SubjectTo(Eigen::Matrix<EqualityConstraint, Rows, Cols>&& constraint) {
    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        m_equalityConstraints.emplace_back(
            std::move(constraint(row, col).variable.GetAutodiff()));
      }
    }
  }

  /**
   * Tells the solver to solve the problem while obeying the given inequality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  template <int Rows, int Cols>
  void SubjectTo(Eigen::Matrix<InequalityConstraint, Rows, Cols>&& constraint) {
    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        m_inequalityConstraints.emplace_back(
            std::move(constraint(row, col).variable.GetAutodiff()));
      }
    }
  }

  /**
   * Solve the optimization problem. The solution will be stored in the original
   * variables used to construct the problem.
   */
  void Solve();

 private:
  // Leaves of the problem's expression tree
  wpi::SmallVector<autodiff::var> m_leaves;

  // Cost function: f(x)
  autodiff::var m_f;

  // Inequality constraints: b(x) ≥ 0
  wpi::SmallVector<autodiff::var> m_inequalityConstraints;

  // Equality constraints: c(x) = 0
  wpi::SmallVector<autodiff::var> m_equalityConstraints;

  /**
   * Initialize leaves with the given vector.
   *
   * @param x The input vector.
   */
  void SetLeaves(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
   * The cost function f(x).
   *
   * @param x The input of f(x).
   */
  double f(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
   * Returns the Eigen column vector representation of a wpi::SmallVector of
   * autodiff variables.
   */
  Eigen::Map<autodiff::VectorXvar> static MakeVectorAutodiff(
      wpi::SmallVector<autodiff::var>& vec);

  /**
   * Return the optimal step size alpha using backtracking line search.
   *
   * @param x The initial guess.
   * @param gradient The gradient at x.
   */
  double BacktrackingLineSearch(
      const Eigen::Ref<const Eigen::VectorXd>& x,
      const Eigen::Ref<const Eigen::VectorXd>& gradient);

  /**
   * Find the optimal solution using gradient descent.
   *
   * @param x The initial guess.
   */
  Eigen::VectorXd GradientDescent(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
  Find the optimal solution using a sequential quadratic programming solver.

  A sequential quadratic programming (SQP) problem has the form:

  @verbatim
       min_x f(x)
  subject to b(x) ≥ 0
             c(x) = 0
  @endverbatim

  where f(x) is the cost function, b(x) are the inequality constraints, and c(x)
  are the equality constraints.

  @param x The initial guess.
  */
  Eigen::VectorXd SQP(const Eigen::Ref<const Eigen::VectorXd>& x);

  friend class AutodiffWrapper;
};

}  // namespace frc
