// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <functional>

#include <wpi/SymbolExports.h>

#include "Eigen/Core"
#include "frc/optimization/AutodiffWrapper.h"
#include "frc/optimization/ConstraintUtil.h"
#include "frc/optimization/Variable.h"

namespace frc {

/**
 * An inequality constraint has the form c(x) ≤ 0.
 */
struct WPILIB_DLLEXPORT InequalityConstraint {
  AutodiffWrapper variable;

  constexpr InequalityConstraint() = default;

  explicit InequalityConstraint(AutodiffWrapper variable);
};

/**
 * Returns a matrix of inequality constraints.
 *
 * @param func A function that returns the inequality constraint for a given row
 *             and column.
 */
template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> MakeInequalityConstraintMatrix(
    std::function<InequalityConstraint(int, int)> func) {
  Eigen::Matrix<InequalityConstraint, Rows, Cols> constraints;

  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < Cols; ++col) {
      constraints(row, col) = func(row, col);
    }
  }

  return constraints;
}

// Matrix-matrix less-than operators

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<(
    const Variable<Rows, Cols>& lhs, const Variable<Rows, Cols>& rhs) {
  return lhs <= rhs;
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<(
    const Variable<Rows, Cols>& lhs,
    const Eigen::Matrix<double, Rows, Cols>& rhs) {
  return lhs <= rhs;
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<(
    const Eigen::Matrix<double, Rows, Cols>& lhs,
    const Variable<Rows, Cols>& rhs) {
  return lhs <= rhs;
}

// Matrix-matrix less-than-or-equal-to operators

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<=(
    const Variable<Rows, Cols>& lhs, const Variable<Rows, Cols>& rhs) {
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return InequalityConstraint{lhs.GetAutodiffWrapper(row, col) -
                                rhs.GetAutodiffWrapper(row, col)};
  });
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<=(
    const Variable<Rows, Cols>& lhs,
    const Eigen::Matrix<double, Rows, Cols>& rhs) {
  Variable<Rows, Cols> rhsVar{rhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return InequalityConstraint{lhs.GetAutodiffWrapper(row, col) -
                                rhsVar.GetAutodiffWrapper(row, col)};
  });
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<=(
    const Eigen::Matrix<double, Rows, Cols>& lhs,
    const Variable<Rows, Cols>& rhs) {
  Variable<Rows, Cols> lhsVar{lhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return InequalityConstraint{lhsVar.GetAutodiffWrapper(row, col) -
                                rhs.GetAutodiffWrapper(row, col)};
  });
}

// Matrix-matrix greater-than operators

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>(
    const Variable<Rows, Cols>& lhs, const Variable<Rows, Cols>& rhs) {
  return lhs >= rhs;
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>(
    const Variable<Rows, Cols>& lhs,
    const Eigen::Matrix<double, Rows, Cols>& rhs) {
  return lhs >= rhs;
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>(
    const Eigen::Matrix<double, Rows, Cols>& lhs,
    const Variable<Rows, Cols>& rhs) {
  return lhs >= rhs;
}

// Matrix-matrix greater-than-or-equal-to operators

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>=(
    const Variable<Rows, Cols>& lhs, const Variable<Rows, Cols>& rhs) {
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Flip sign and right-hand side zero
    return InequalityConstraint{rhs.GetAutodiffWrapper(row, col) -
                                lhs.GetAutodiffWrapper(row, col)};
  });
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>=(
    const Variable<Rows, Cols>& lhs,
    const Eigen::Matrix<double, Rows, Cols>& rhs) {
  Variable<Rows, Cols> rhsVar{rhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Flip sign and right-hand side zero
    return InequalityConstraint{rhsVar.GetAutodiffWrapper(row, col) -
                                lhs.GetAutodiffWrapper(row, col)};
  });
}

template <int Rows, int Cols>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>=(
    const Eigen::Matrix<double, Rows, Cols>& lhs,
    const Variable<Rows, Cols>& rhs) {
  Variable<Rows, Cols> lhsVar{lhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Flip sign and right-hand side zero
    return InequalityConstraint{rhs.GetAutodiffWrapper(row, col) -
                                lhsVar.GetAutodiffWrapper(row, col)};
  });
}

// Matrix-scalar comparisons

template <int Rows, int Cols, typename Rhs,
          std::enable_if_t<IsScalar<Rhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<(
    const Variable<Rows, Cols>& lhs, const Rhs& rhs) {
  return lhs <= rhs;
}

template <int Rows, int Cols, typename Rhs,
          std::enable_if_t<IsScalar<Rhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<=(
    const Variable<Rows, Cols>& lhs, const Rhs& rhs) {
  AutodiffWrapper rhsVar{rhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return InequalityConstraint{lhs.GetAutodiffWrapper(row, col) - rhsVar};
  });
}

template <int Rows, int Cols, typename Rhs,
          std::enable_if_t<IsScalar<Rhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>(
    const Variable<Rows, Cols>& lhs, const Rhs& rhs) {
  return lhs >= rhs;
}

template <int Rows, int Cols, typename Rhs,
          std::enable_if_t<IsScalar<Rhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>=(
    const Variable<Rows, Cols>& lhs, const Rhs& rhs) {
  AutodiffWrapper rhsVar{rhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Flip sign and right-hand side zero
    return InequalityConstraint{rhsVar - lhs.GetAutodiffWrapper(row, col)};
  });
}

// Scalar-matrix comparisons

template <int Rows, int Cols, typename Lhs,
          std::enable_if_t<IsScalar<Lhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<(
    const Lhs& lhs, const Variable<Rows, Cols>& rhs) {
  return lhs <= rhs;
}

template <int Rows, int Cols, typename Lhs,
          std::enable_if_t<IsScalar<Lhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator<=(
    const Lhs& lhs, const Variable<Rows, Cols>& rhs) {
  AutodiffWrapper lhsVar{lhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return InequalityConstraint{lhsVar - rhs.GetAutodiffWrapper(row, col)};
  });
}

template <int Rows, int Cols, typename Lhs,
          std::enable_if_t<IsScalar<Lhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>(
    const Lhs& lhs, const Variable<Rows, Cols>& rhs) {
  return lhs >= rhs;
}

template <int Rows, int Cols, typename Lhs,
          std::enable_if_t<IsScalar<Lhs>, int> = 0>
Eigen::Matrix<InequalityConstraint, Rows, Cols> operator>=(
    const Lhs& lhs, const Variable<Rows, Cols>& rhs) {
  AutodiffWrapper lhsVar{lhs};
  return MakeInequalityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Flip sign and right-hand side zero
    return InequalityConstraint{rhs.GetAutodiffWrapper(row, col) - lhsVar};
  });
}

}  // namespace frc
