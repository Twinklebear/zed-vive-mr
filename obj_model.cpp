#include <iostream>
#include <map>
#include <glm/ext.hpp>
#include "util.h"
#include "obj_model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

ObjModel::Model::Model(const std::vector<Vertex> &verts, const std::vector<uint16_t> &indices)
	: num_verts(indices.size())
{
	glGenBuffers(1, &verts_buf);
	glGenBuffers(1, &ebo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, verts_buf);
	glBufferData(GL_SHADER_STORAGE_BUFFER, verts.size() * sizeof(Vertex),
			verts.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint16_t),
			indices.data(), GL_STATIC_DRAW);
}
ObjModel::Model::~Model() {
	glDeleteBuffers(1, &verts_buf);
	glDeleteBuffers(1, &ebo);
}

struct Compare {
	bool operator()(const tinyobj::index_t &i1, const tinyobj::index_t &i2) const {
		return std::tie(i1.vertex_index, i1.normal_index, i1.texcoord_index)
			< std::tie(i2.vertex_index, i2.normal_index, i2.texcoord_index);
	}
};

ObjModel::ObjModel(const std::string &file) {
	glGenVertexArrays(1, &vao);

	const std::string res_path = get_resource_path();
	vr_shader = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "obj_model_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "obj_model_frag.glsl")
	});
	mr_shader = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "obj_model_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "mr_obj_model_frag.glsl")
	});
	vr_model_mat_unif = glGetUniformLocation(vr_shader, "model_mat");
	mr_model_mat_unif = glGetUniformLocation(mr_shader, "model_mat");

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> obj_materials;
	std::string obj_load_err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &obj_materials, &obj_load_err, file.c_str())) {
		std::cout << "TinyOBJ load error: " << obj_load_err << std::endl;
		throw std::runtime_error("Failed to load OBJ");
	}

	std::cout << "Loaded " << shapes.size() << " shapes" << std::endl;
	for (auto &s : shapes) {
		std::vector<uint16_t> indices;
		std::vector<Vertex> vertices;
		std::map<tinyobj::index_t, uint16_t, Compare> vertex_indices;

		size_t index_offset = 0;
		for (auto &fv : s.mesh.num_face_vertices) {
			for (int v = 0; v < fv; ++v, ++index_offset) {
				const tinyobj::index_t idx = s.mesh.indices[index_offset];
				auto fnd = vertex_indices.find(idx);
				uint16_t index = 0;
				if (fnd == vertex_indices.end()) {
					Vertex vertex;
					for (size_t j = 0; j < 3; ++j) {
						vertex.pos[j] = attrib.vertices[3 * idx.vertex_index + j];
						if (idx.normal_index != -1) {
							vertex.normal[j] = attrib.normals[3 * idx.normal_index + j];
						} else {
							vertex.normal[j] = 1;
						}
					}
					index = vertices.size();
					vertex_indices[idx] = index;
					vertices.push_back(vertex);
				} else {
					index = fnd->second;
				}
				indices.push_back(index);
			}
		}
		models.emplace_back(vertices, indices);
	}
}
ObjModel::~ObjModel() {
	glDeleteProgram(vr_shader);
	glDeleteProgram(mr_shader);
	glDeleteVertexArrays(1, &vao);
}
void ObjModel::set_model_mat(const glm::mat4 &mat) {
	model_mat = mat;
}
void ObjModel::render_vr() const {
	glUseProgram(vr_shader);
	glUniformMatrix4fv(vr_model_mat_unif, 1, GL_FALSE, glm::value_ptr(model_mat));
	render();
}
void ObjModel::render_mr() const {
	glUseProgram(mr_shader);
	glUniformMatrix4fv(mr_model_mat_unif, 1, GL_FALSE, glm::value_ptr(model_mat));
	render();
}
void ObjModel::render() const {
	glBindVertexArray(vao);
	for (const auto &m : models) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m.verts_buf);
		glDrawElements(GL_TRIANGLES, m.num_verts, GL_UNSIGNED_SHORT, 0);
	}
}

