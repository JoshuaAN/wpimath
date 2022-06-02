// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <initializer_list>
#include <type_traits>
#include <utility>

#include <wpi/SymbolExports.h>

#include "Eigen/Core"
#include "frc/optimization/AutodiffWrapper.h"

namespace frc {

template <int Rows, int Cols>
class Variable {
 public:
  Variable() = default;

  template <int Rows2 = Rows, int Cols2 = Cols,
            std::enable_if_t<Rows2 == 1 && Cols2 == 1, int> = 0>
  Variable(double value) {  // NOLINT
    m_storage(0, 0) = value;
  }

  Variable(std::initializer_list<double> values) : m_storage{values} {}

  Variable(std::initializer_list<std::initializer_list<double>> values)
      : m_storage{values} {}

  template <int Rows2 = Rows, int Cols2 = Cols,
            std::enable_if_t<Rows2 == 1 && Cols2 == 1, int> = 0>
  Variable(const AutodiffWrapper& rhs) {  // NOLINT
    m_storage(0, 0) = rhs;
  }

  template <int Rows2 = Rows, int Cols2 = Cols,
            std::enable_if_t<Rows2 == 1 && Cols2 == 1, int> = 0>
  Variable(AutodiffWrapper&& rhs) {  // NOLINT
    m_storage(0, 0) = std::move(rhs);
  }

  Variable(const Eigen::Matrix<double, Rows, Cols>& values) {  // NOLINT
    for (size_t row = 0; row < Rows; ++row) {
      for (size_t col = 0; col < Cols; ++col) {
        m_storage(row, col) = values(row, col);
      }
    }
  }

  Variable(Eigen::Matrix<double, Rows, Cols>&& values) {  // NOLINT
    for (size_t row = 0; row < Rows; ++row) {
      for (size_t col = 0; col < Cols; ++col) {
        m_storage(row, col) = std::move(values(row, col));
      }
    }
  }

  Variable(const Eigen::Matrix<AutodiffWrapper, Rows, Cols>& values)  // NOLINT
      : m_storage{values} {}

  Variable(Eigen::Matrix<AutodiffWrapper, Rows, Cols>&& values)  // NOLINT
      : m_storage{std::move(values)} {}

  auto operator()(int row, int col) {
    return Variable<1, 1>{m_storage(row, col)};
  }

  template <int Cols2 = Cols, std::enable_if_t<Cols2 == 1, int> = 0>
  auto operator()(int row) {
    return Variable<1, 1>{m_storage(row, 0)};
  }

  template <int Rows2 = Rows, int Cols2 = Cols,
            std::enable_if_t<Rows2 == 1 && Cols2 == 1, int> = 0>
  Variable<Rows, Cols>& operator=(double rhs) {
    m_storage(0, 0) = rhs;
    return *this;
  }

  template <int RowsRhs, int ColsRhs>
  friend Variable<Rows, ColsRhs> operator*(
      const Variable<Rows, Cols>& lhs, const Variable<RowsRhs, ColsRhs>& rhs) {
    static_assert(Cols == RowsRhs, "Matrix dimension mismatch for operator*");
    return Eigen::Matrix<AutodiffWrapper, Rows, ColsRhs>{lhs.m_storage *
                                                         rhs.GetStorage()};
  }

  template <int RowsRhs, int ColsRhs>
  Variable<Rows, ColsRhs>& operator*=(const Variable<RowsRhs, ColsRhs>& rhs) {
    static_assert(Cols == RowsRhs, "Matrix dimension mismatch for operator*");
    m_storage *= rhs.GetStorage();
    return *this;
  }

  friend Variable<Rows, Cols> operator+(const Variable<Rows, Cols>& lhs,
                                        const Variable<Rows, Cols>& rhs) {
    return Eigen::Matrix<AutodiffWrapper, Rows, Cols>{lhs.m_storage +
                                                      rhs.m_storage};
  }

  Variable<Rows, Cols>& operator+=(const Variable<Rows, Cols>& rhs) {
    m_storage += rhs.m_storage;
    return *this;
  }

  friend Variable<Rows, Cols> operator-(const Variable<Rows, Cols>& lhs,
                                        const Variable<Rows, Cols>& rhs) {
    return Eigen::Matrix<AutodiffWrapper, Rows, Cols>{lhs.m_storage -
                                                      rhs.m_storage};
  }

  Variable<Rows, Cols>& operator-=(const Variable<Rows, Cols>& rhs) {
    m_storage -= rhs.m_storage;
    return *this;
  }

  friend Variable<Rows, Cols> operator-(const Variable<Rows, Cols>& lhs) {
    return Eigen::Matrix<AutodiffWrapper, Rows, Cols>{-lhs.m_storage};
  }

  Variable<Cols, Rows> transpose() const {
    return Eigen::Matrix<AutodiffWrapper, Cols, Rows>{m_storage.transpose()};
  }

  AutodiffWrapper& GetAutodiffWrapper(int row, int col) {
    return m_storage(row, col);
  }

  const AutodiffWrapper& GetAutodiffWrapper(int row, int col) const {
    return m_storage(row, col);
  }

  template <int Cols2 = Cols, std::enable_if_t<Cols2 != 1, int> = 0>
  double Value(int row, int col) const {
    return m_storage(row, col).Value();
  }

  template <int Rows2 = Rows, int Cols2 = Cols,
            std::enable_if_t<Rows2 != 1 && Cols2 == 1, int> = 0>
  double Value(int row) const {
    return m_storage(row, 0).Value();
  }

  template <int Rows2 = Rows, int Cols2 = Cols,
            std::enable_if_t<Rows2 != 1 || Cols2 != 1, int> = 0>
  Eigen::Matrix<double, Rows, Cols> Value() const {
    Eigen::Matrix<double, Rows, Cols> ret;
    for (size_t row = 0; row < Rows; ++row) {
      for (size_t col = 0; col < Cols; ++col) {
        ret(row, col) = m_storage(row, col).Value();
      }
    }
    return ret;
  }

  template <int Rows2 = Rows, int Cols2 = Cols,
            std::enable_if_t<Rows2 == 1 && Cols2 == 1, int> = 0>
  double Value() const {
    return m_storage(0, 0).Value();
  }

  const Eigen::Matrix<AutodiffWrapper, Rows, Cols>& GetStorage() const {
    return m_storage;
  }

 private:
  Eigen::Matrix<AutodiffWrapper, Rows, Cols> m_storage;
};

/**
 * std::abs() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> abs(const Variable<1, 1>& x);  // NOLINT

/**
 * std::acos() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> acos(const Variable<1, 1>& x);  // NOLINT

/**
 * std::asin() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> asin(const Variable<1, 1>& x);  // NOLINT

/**
 * std::atan() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> atan(const Variable<1, 1>& x);  // NOLINT

/**
 * std::atan2() for Variables.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> atan2(const Variable<1, 1>& y,  // NOLINT
                                      const Variable<1, 1>& x);

/**
 * std::cos() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> cos(const Variable<1, 1>& x);  // NOLINT

/**
 * std::cosh() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> cosh(const Variable<1, 1>& x);  // NOLINT

/**
 * std::erf() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> erf(const Variable<1, 1>& x);  // NOLINT

/**
 * std::exp() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> exp(const Variable<1, 1>& x);  // NOLINT

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> hypot(const Variable<1, 1>& x,  // NOLINT
                                      const Variable<1, 1>& y);

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 * @param z The z argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> hypot(const Variable<1, 1>& x,  // NOLINT
                                      const Variable<1, 1>& y,
                                      const Variable<1, 1>& z);

/**
 * std::log() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> log(const Variable<1, 1>& x);  // NOLINT

/**
 * std::log10() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> log10(const Variable<1, 1>& x);  // NOLINT

/**
 * std::pow() for Variables.
 *
 * @param base The base.
 * @param power The power.
 */
WPILIB_DLLEXPORT Variable<1, 1> pow(const Variable<1, 1>& base,  // NOLINT
                                    int power);

/**
 * std::sin() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> sin(const Variable<1, 1>& x);  // NOLINT

/**
 * std::sinh() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> sinh(const Variable<1, 1>& x);  // NOLINT

/**
 * std::sqrt() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> sqrt(const Variable<1, 1>& x);  // NOLINT

/**
 * std::tan() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> tan(const Variable<1, 1>& x);  // NOLINT

/**
 * std::tanh() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> tanh(const Variable<1, 1>& x);  // NOLINT

}  // namespace frc
