#pragma once

#include <array>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <sl/Camera.hpp>
#include "gl_core_4_5.h"

class PointCloud {
	size_t num_points;
	bool has_data;
	GLuint shader, vao;
	std::array<GLuint, 2> ssbo;
	GLuint model_mat_unif;
	glm::mat4 model_mat;

	std::array<cudaGraphicsResource_t, 2> cuda_ssbo_ref;
	cudaStream_t cuda_stream;
	cudaEvent_t cuda_event;
	size_t copy_target;
	bool is_copying;

public:
	PointCloud(size_t num_points);
	~PointCloud();
	void update_point_cloud(const sl::Mat &pc);
	void set_model_mat(const glm::mat4 &mat);
	void render() const;
};

