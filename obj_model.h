#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "gl_core_4_5.h"

class ObjModel {
	GLuint vr_shader, mr_shader, vao;
	GLuint vr_model_mat_unif, mr_model_mat_unif;
	glm::mat4 model_mat;

	struct Vertex {
		glm::vec3 pos, normal;
	};

	struct Model {
		size_t num_verts;
		GLuint verts_buf, ebo;

		Model(const std::vector<Vertex> &verts, const std::vector<uint16_t> &indices);
		~Model();
	};

	std::vector<Model> models;

public:
	ObjModel(const std::string &file);
	~ObjModel();
	void set_model_mat(const glm::mat4 &mat);
	void render_vr() const;
	void render_mr() const;

private:
	void render() const;
};

