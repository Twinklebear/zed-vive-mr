#include <iostream>
#include <glm/ext.hpp>
#include "util.h"
#include "obj_model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

ObjModel::Model::Model(const std::vector<glm::vec3> &verts) : num_verts(verts.size()) {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(glm::vec3),
			verts.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindVertexArray(0);
}
ObjModel::Model::~Model() {
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
}

ObjModel::ObjModel(const std::string &file) {
	const std::string res_path = get_resource_path();
	shader = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "obj_model_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "obj_model_frag.glsl")
	});
	model_mat_unif = glGetUniformLocation(shader, "model_mat");

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> obj_materials;
	std::string obj_load_err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &obj_materials, &obj_load_err, file.c_str())) {
		std::cout << "TinyOBJ load error: " << obj_load_err << std::endl;
		throw std::runtime_error("Failed to load OBJ");
	}

	for (auto &s : shapes) {
		std::vector<uint16_t> indices;
		std::vector<glm::vec3> vertices;

		size_t index_offset = 0;
		for (auto &fv : s.mesh.num_face_vertices) {
			for (int v = 0; v < fv; ++v, ++index_offset) {
				const tinyobj::index_t idx = s.mesh.indices[index_offset];
				vertices.emplace_back(attrib.vertices[3 * idx.vertex_index],
						attrib.vertices[3 * idx.vertex_index + 1],
						attrib.vertices[3 * idx.vertex_index + 2]);
			}
		}
		models.emplace_back(vertices);
	}
}
ObjModel::~ObjModel() {
	glDeleteProgram(shader);
}
void ObjModel::set_model_mat(const glm::mat4 &mat) {
	model_mat = mat;
}
void ObjModel::render() const {
	glUseProgram(shader);
	glUniformMatrix4fv(model_mat_unif, 1, GL_FALSE, glm::value_ptr(model_mat));
	for (const auto &m : models) {
		glBindVertexArray(m.vao);
		glDrawArrays(GL_TRIANGLES, 0, m.num_verts);
	}
}

