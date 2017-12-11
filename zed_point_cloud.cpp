#include <glm/ext.hpp>
#include "util.h"
#include "zed_point_cloud.h"

PointCloud::PointCloud() : num_points(0) {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &ssbo);

	const std::string res_path = get_resource_path();
	shader = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "point_cloud_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "point_cloud_frag.glsl")
	});
	model_mat_unif = glGetUniformLocation(shader, "model_mat");
}
PointCloud::~PointCloud() {
	glDeleteProgram(shader);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &ssbo);
}
void PointCloud::update_point_cloud(const sl::Mat &pc) {
	num_points = pc.getWidth() * pc.getHeight();
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * pc.getPixelBytes(),
			pc.getPtr<float>(), GL_STREAM_DRAW);
}
void PointCloud::set_model_mat(const glm::mat4 &mat) {
	model_mat = mat;
}
void PointCloud::render() const {
	if (num_points > 0) {
		glBindVertexArray(vao);
		glUseProgram(shader);
		glUniformMatrix4fv(model_mat_unif, 1, GL_FALSE, glm::value_ptr(model_mat));
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo);
		glDrawArrays(GL_POINTS, 0, num_points);
	}
}

