#include <glm/ext.hpp>
#include "util.h"
#include "zed_point_cloud.h"

PointCloud::PointCloud(size_t num_points) : num_points(num_points) {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, num_points * 4 * sizeof(float),
			nullptr, 0);
	cudaGraphicsGLRegisterBuffer(&cuda_ssbo_ref, ssbo,
			cudaGraphicsRegisterFlagsWriteDiscard);

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
	cudaGraphicsUnregisterResource(cuda_ssbo_ref);
	glDeleteBuffers(1, &ssbo);
}
void PointCloud::update_point_cloud(const sl::Mat &pc) {
	assert(num_points == pc.getWidth() * pc.getHeight());
	cudaGraphicsMapResources(1, &cuda_ssbo_ref);
	void *mapped_ptr = nullptr;
	size_t size = 0;
	cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &size, cuda_ssbo_ref);
	cudaMemcpy(mapped_ptr, pc.getPtr<float>(sl::MEM_GPU),
			num_points * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &cuda_ssbo_ref);
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

