#pragma once

#include <map>
#include <string>
#include <array>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <sl/Camera.hpp>
#include "gl_core_4_5.h"
#include "openvr_display.h"

/* The calibration is the offset from the device tracking the camera
 * (it's parent), to the camera itself. For compatibility with ZED's Unity
 * calibration app we actually work in Unity's coordinate frame internally
 * in the calibration file but flip the matrix when calling 'tracker_to_camera'
 * into OpenGL's coordinate frame.
 */
struct ZedCalibration {
	glm::vec3 translation, rotation;
	float fov;
	std::string tracker_serial;

	ZedCalibration();
	ZedCalibration(const std::string &calibration_file);
	void save(const std::string &calibration_file) const;
	/* Get the transform from the tracker to the camera, in OpenGL's
	 * coordinate frame.
	 */
	glm::mat4 tracker_to_camera() const;

private:
	void load_calibration(const std::string &file);
	void save_calibration(const std::string &file) const;
};

struct ZedManager {
	ZedCalibration calibration;
	std::shared_ptr<OpenVRDisplay> vr;
	vr::TrackedDeviceIndex_t tracker;
	sl::Camera camera;
	sl::RuntimeParameters runtime_params;
	std::map<sl::VIEW, sl::Mat> image_requests;
	std::map<sl::MEASURE, sl::Mat> measure_requests;

	GLuint prepass_vao, prepass_shader;
	// ZED color and depth textures to do ping-ponging
	std::array<GLuint, 2> color_textures;
	std::array<GLuint, 2> depth_textures;
	// ZED color and depth CUDA resource references
	std::array<cudaGraphicsResource_t, 2> cuda_color_tex_refs;
	std::array<cudaGraphicsResource_t, 2> cuda_depth_tex_refs;
	// CUDA streams to do memcpys async and in parallel
	std::array<cudaStream_t, 2> cuda_streams;
	// Events to check when the copies have finished
	std::array<cudaEvent_t, 2> cuda_events;
	size_t copy_target;
	bool is_copying;

	ZedManager(ZedCalibration calibration, std::shared_ptr<OpenVRDisplay> &vr);
	~ZedManager();
	ZedManager(const ZedManager&) = delete;
	ZedManager& operator=(const ZedManager&) = delete;
	bool is_tracking() const; 
	// Find and attach to the tracker we calibrated for
	void find_tracker();
	// Set the camera as attached to a different tracker, or -1 for untracked
	void set_tracker(vr::TrackedDeviceIndex_t &device);
	/* Request an image to be retrieved when grab is called. By default the
	 * VIEW_LEFT image will be requested
	 */
	void request_image(sl::VIEW view);
	/* Request a measure to be retrieved when grab is called. By default the
	 * MEASURE_DEPTH measure will be requested
	 */
	void request_measure(sl::MEASURE measure);

	/* Setup rendering state for the ZED and retrieve all images and measures
	 * if a new frame is available from the camera
	 */
	void begin_render(glm::mat4 &view, glm::mat4 &projection);
	/* Render the ZED camera's color and depth as a pre-pass to composite
	 * the virtual scene with the camera data
	 */
	void render_zed_prepass();
	// Get the projection matrix matching the camera, based on their Unity sample
	glm::mat4 camera_projection_matrix();
};

std::string zed_error_to_string(const sl::ERROR_CODE &err);

