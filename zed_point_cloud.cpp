#include <glm/ext.hpp>
#include "util.h"
#include "zed_point_cloud.h"

PointCloud::PointCloud(size_t num_points) : num_points(num_points), has_data(false) {
	glGenVertexArrays(1, &vao);
	glGenBuffers(ssbo.size(), ssbo.data());
	for (size_t i = 0; i < ssbo.size(); ++i) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[i]);
		glBufferStorage(GL_SHADER_STORAGE_BUFFER, num_points * 4 * sizeof(float),
				nullptr, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_ssbo_ref[i], ssbo[i],
				cudaGraphicsRegisterFlagsWriteDiscard);
	}

	const std::string res_path = get_resource_path();
	shader = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "point_cloud_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "point_cloud_frag.glsl")
	});
	model_mat_unif = glGetUniformLocation(shader, "model_mat");

	cudaStreamCreate(&cuda_stream);
	cudaEventCreate(&cuda_event);
	copy_target = 1;
	is_copying = false;
}
PointCloud::~PointCloud() {
	glDeleteProgram(shader);
	glDeleteVertexArrays(1, &vao);
	if (is_copying) {
		cudaEventSynchronize(cuda_event);
		cudaGraphicsUnmapResources(1, &cuda_ssbo_ref[copy_target], cuda_stream);
	}
	cudaEventDestroy(cuda_event);
	cudaGraphicsUnregisterResource(cuda_ssbo_ref[0]);
	cudaGraphicsUnregisterResource(cuda_ssbo_ref[1]);
	cudaStreamDestroy(cuda_stream);
	glDeleteBuffers(ssbo.size(), &ssbo[2]);
}
void PointCloud::update_point_cloud(const sl::Mat &pc) {
	assert(num_points == pc.getWidth() * pc.getHeight());
	if (is_copying) {
		cudaError_t err = cudaEventQuery(cuda_event);
		if (err == cudaSuccess) {
			is_copying = false;
			has_data = true;
			cudaGraphicsUnmapResources(1, &cuda_ssbo_ref[copy_target], cuda_stream);
			copy_target = (copy_target + 1) % 2;
		} else if (err != cudaErrorNotReady) {
			throw std::runtime_error("Failure executing copy of point cloud data");
		}
	}
	if (!is_copying) {
		is_copying = true;
		cudaGraphicsMapResources(1, &cuda_ssbo_ref[copy_target], cuda_stream);
		void *mapped_ptr = nullptr;
		size_t size = 0;
		cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &size, cuda_ssbo_ref[copy_target]);
		cudaMemcpyAsync(mapped_ptr, pc.getPtr<float>(sl::MEM_GPU),
				num_points * 4 * sizeof(float), cudaMemcpyDeviceToDevice,
				cuda_stream);
		cudaEventRecord(cuda_event, cuda_stream);
	}
}
void PointCloud::set_model_mat(const glm::mat4 &mat) {
	model_mat = mat;
}
void PointCloud::render() const {
	if (has_data) {
		glBindVertexArray(vao);
		glUseProgram(shader);
		glUniformMatrix4fv(model_mat_unif, 1, GL_FALSE, glm::value_ptr(model_mat));
		size_t render_buf = (copy_target + 1) % 2;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[render_buf]);
		glDrawArrays(GL_POINTS, 0, num_points);
	}
}

