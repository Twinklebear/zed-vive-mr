#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <sl/Camera.hpp>
#include "gl_core_4_5.h"

class PointCloud {
	GLuint shader, vao, ssbo;
	GLuint model_mat_unif;
	glm::mat4 model_mat;
	size_t num_points;

public:
	PointCloud();
	~PointCloud();
	void update_point_cloud(const sl::Mat &pc);
	void set_model_mat(const glm::mat4 &mat);
	void render() const;
};

