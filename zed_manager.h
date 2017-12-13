#pragma once

#include <map>
#include <string>
#include <array>
#include <memory>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <sl/Camera.hpp>
#include "gl_core_4_5.h"
#include "openvr_display.h"

// The calibration is the offset from the camera to
// the device tracking it
struct ZedCalibration {
	glm::vec3 translation, rotation;
	std::string tracker_serial;

	ZedCalibration();
	ZedCalibration(const std::string &calibration_file);
	void save(const std::string &calibration_file) const;
	glm::mat4 tracker_to_camera() const;
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
	// ZED color and depth textures
	std::array<GLuint, 2> textures;

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

