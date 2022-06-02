// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <type_traits>

#include "Eigen/Core"

namespace frc {

template <class T>
inline constexpr bool IsScalar =
    std::is_same_v<T, double> || std::is_same_v<T, int> ||
    std::is_same_v<T, Eigen::Matrix<double, 1, 1>>;

}  // namespace frc
