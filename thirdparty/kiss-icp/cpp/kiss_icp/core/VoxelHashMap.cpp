// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "VoxelHashMap.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <sophus/se3.hpp>
#include <vector>

#include "VoxelUtils.hpp"

namespace kiss_icp {

std::vector<Eigen::Vector3d> VoxelHashMap::GetPoints(const std::vector<Voxel> &query_voxels) const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(query_voxels.size() * static_cast<size_t>(max_points_per_voxel_));
    std::for_each(query_voxels.cbegin(), query_voxels.cend(), [&](const auto &query) {
        auto search = map_.find(query);
        if (search != map_.end()) {
            const auto &voxel_points = search.value();
            points.insert(points.end(), voxel_points.cbegin(), voxel_points.cend());
        }
    });
    points.shrink_to_fit();
    return points;
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(map_.size() * static_cast<size_t>(max_points_per_voxel_));
    std::for_each(map_.cbegin(), map_.cend(), [&](const auto &map_element) {
        const auto &voxel_points = map_element.second;
        points.insert(points.end(), voxel_points.cbegin(), voxel_points.cend());
    });
    points.shrink_to_fit();
    return points;
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector3d> &points,
                          const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose) {
    std::vector<Eigen::Vector3d> points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points) {
    const double map_resolution = std::sqrt(voxel_size_ * voxel_size_ / max_points_per_voxel_);
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        auto voxel = PointToVoxel(point, voxel_size_);
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_points = search.value();
            if (voxel_points.size() == max_points_per_voxel_ ||
                std::any_of(voxel_points.cbegin(), voxel_points.cend(),
                            [&](const auto &voxel_point) {
                                return (voxel_point - point).norm() < map_resolution;
                            })) {
                return;
            }
            voxel_points.emplace_back(point);
        } else {
            std::vector<Eigen::Vector3d> voxel_points;
            voxel_points.reserve(max_points_per_voxel_);
            voxel_points.emplace_back(point);
            map_.insert({voxel, std::move(voxel_points)});
        }
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = max_distance_ * max_distance_;
    for (auto it = map_.begin(); it != map_.end();) {
        const auto &[voxel, voxel_points] = *it;
        const auto &pt = voxel_points.front();
        if ((pt - origin).squaredNorm() >= (max_distance2)) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace kiss_icp
