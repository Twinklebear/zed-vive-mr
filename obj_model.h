#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "gl_core_4_5.h"

class ObjModel {
	GLuint shader;
	GLuint model_mat_unif;
	glm::mat4 model_mat;

	struct Model {
		size_t num_verts;
		GLuint vao, vbo;

		Model(const std::vector<glm::vec3> &verts);
		~Model();
	};

	std::vector<Model> models;

public:
	ObjModel(const std::string &file);
	~ObjModel();
	void set_model_mat(const glm::mat4 &mat);
	void render() const;
};

