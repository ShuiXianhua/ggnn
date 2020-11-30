/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Lukas Rupert, Patrick Wieschollek, Hendrik P.A. Lensch
//
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <stdio.h>

#include <iostream>
#include <vector>

#include "cub/cub.cuh"
#include "ggnn/config.hpp"
#include "ggnn/cuda_knn_config.cuh"
#include "ggnn/cuda_knn_ggnn.cuh"
#include "ggnn/graph/cuda_knn_ggnn_graph.cuh"
#include "ggnn/utils/cuda_knn_core_utils.cuh"
#include "ggnn/utils/cuda_knn_dataset.cuh"

DEFINE_string(base_filename, "", "path to file with base vectors");
DEFINE_string(query_filename, "", "path to file with perform_query vectors");
DEFINE_string(groundtruth_filename, "",
              "path to file with groundtruth vectors");
DEFINE_string(graph_filename, "",
              "path to file that contains the serialized graph");
DEFINE_double(tau, 0.5, "Parameter tau");
DEFINE_int32(refinement_iterations, 2, "Number of refinement iterations");
DEFINE_int32(gpu_id, 0, "GPU id");

constexpr float milliseconds_per_second = 1000.f;

using KeyT = int32_t;
using BaseT = float;
using ValueT = float;
using BAddrT = uint32_t;
using GAddrT = uint32_t;

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  gflags::SetUsageMessage(
      "GGNN: Graph-based GPU Nearest Neighbor Search\n"
      "by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. "
      "Lensch\n"
      "(c) 2020 Computer Graphics University of Tuebingen");
  gflags::SetVersionString("1.0.0");
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Reading files";
  CHECK(file_exists(FLAGS_base_filename))
      << "File for base vectors has to exists";
  CHECK(file_exists(FLAGS_query_filename))
      << "File for perform_query vectors has to exists";
  CHECK(file_exists(FLAGS_groundtruth_filename))
      << "File for groundtruth vectors has to exists";

  CHECK_GE(FLAGS_tau, 0) << "Tau has to be bigger or equal 0.";
  CHECK_LE(FLAGS_tau, 1) << "Tau has to be smaller or equal 1.";
  CHECK_GE(FLAGS_refinement_iterations, 0)
      << "The number of refinement iterations has to non-negative.";

  // ####################################################################
  // compile-time configuration
  //
  // ####################################################################
  // perform_build configuration
  const int KBuild = 24;
  const int KF = KBuild / 2;
  const int S = 32;
  const int L = 4;
  const bool bubble_merge = true;

  // perform_query configuration
  const int KQuery = 10;

  // dataset configuration (here: SIFT1M)
  const int D = 128;

  LOG(INFO) << "Using the following parameters " << KBuild << " (KBuild) " << KF
            << " (KF) " << S << " (S) " << L << " (L) " << D << " (D) ";

  const bool export_graph =
      !FLAGS_graph_filename.empty() && !file_exists(FLAGS_graph_filename);
  const bool import_graph =
      !FLAGS_graph_filename.empty() && file_exists(FLAGS_graph_filename);

  // Set the requested GPU id, if possible.
  {
    int numGpus;
    cudaGetDeviceCount(&numGpus);
    CHECK_GE(FLAGS_gpu_id, 0) << "This GPU does not exist";
    CHECK_LT(FLAGS_gpu_id, numGpus) << "This GPU does not exist";
  }
  cudaSetDevice(FLAGS_gpu_id);

  const bool perform_build = export_graph || !import_graph;
  const bool perform_query = true;

  typedef GGNN<KeyT, ValueT, GAddrT, BaseT, BAddrT, D, KBuild, KF, KQuery, S>
      GGNN;
  GGNN m_ggnn{FLAGS_base_filename, FLAGS_query_filename,
              FLAGS_groundtruth_filename, L, static_cast<float>(FLAGS_tau)};

  if (perform_build) {
    std::vector<float> construction_times;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    LOG(INFO) << "Starting Graph construction...";

    cudaEventRecord(start);
    if (bubble_merge) {
      m_ggnn.build_bubble_merge();
    } else {
      m_ggnn.build_simple_merge();
    }

    cudaEventRecord(stop);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    construction_times.push_back(milliseconds);

    for (int refinement_step = 0; refinement_step < FLAGS_refinement_iterations;
         ++refinement_step) {
      DLOG(INFO) << "Refinement step " << refinement_step;
      m_ggnn.refine();

      cudaEventRecord(stop);
      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaPeekAtLastError());
      cudaEventSynchronize(stop);

      float elapsed_milliseconds = 0;
      cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
      construction_times.push_back(elapsed_milliseconds);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const float elapsed_seconds =
        construction_times[0] / milliseconds_per_second;
    const float ms_per_point = construction_times[0] / m_ggnn.m_ggnn_graph.N;

    for (int refinement_step = 0;
         refinement_step < construction_times.size() - 1; refinement_step++) {
      const float elapsed_milliseconds = construction_times[refinement_step];
      const float elapsed_seconds =
          elapsed_milliseconds / milliseconds_per_second;
      const int number_of_points = m_ggnn.m_ggnn_graph.N;

      LOG(INFO) << "Graph construction + refinement_step";
      LOG(INFO) << "                   -- secs: " << elapsed_seconds;
      LOG(INFO) << "                   -- points: " << number_of_points;
      LOG(INFO) << "                   -- ms/point: "
                << elapsed_milliseconds / number_of_points;
    }

    if (export_graph) {
      m_ggnn.write(FLAGS_graph_filename, FLAGS_base_filename);
    }
  }

  if (perform_query) {
    if (import_graph) {
      m_ggnn.read(FLAGS_graph_filename);
    }

    m_ggnn.prefetch(FLAGS_gpu_id);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaPeekAtLastError());

    auto query_function = [&m_ggnn](const float tau_query) {
      cudaMemcpyToSymbol(c_tau_query, &tau_query, sizeof(float));
      LOG(INFO) << "Query with tau_query " << tau_query;

      m_ggnn.queryLayer();
      m_ggnn.queryLayerFast();
    };

    query_function(0.35f);
    query_function(0.42f);
    query_function(0.60f);
  }

  printf("done! \n");
  gflags::ShutDownCommandLineFlags();
  return 0;
}
