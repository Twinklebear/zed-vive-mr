#pragma once

#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <sl/Camera.hpp>
#include "gl_core_4_5.h"

class PointCloud {
	GLuint shader, vao, ssbo;
	cudaGraphicsResource_t cuda_ssbo_ref;
	GLuint model_mat_unif;
	glm::mat4 model_mat;
	size_t num_points;

public:
	PointCloud(size_t num_points);
	~PointCloud();
	void update_point_cloud(const sl::Mat &pc);
	void set_model_mat(const glm::mat4 &mat);
	void render() const;
};

