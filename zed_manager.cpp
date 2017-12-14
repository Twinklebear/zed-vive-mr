#include <fstream>
#include <iostream>
#include "util.h"
#include "zed_manager.h"

ZedCalibration::ZedCalibration() : translation(0.f), rotation(0.f) {}
ZedCalibration::ZedCalibration(const std::string &calibration_file) {
	std::ifstream calib_file(calibration_file, std::ios::binary);
	calib_file.read(reinterpret_cast<char*>(glm::value_ptr(translation)), 3 * sizeof(float));
	calib_file.read(reinterpret_cast<char*>(glm::value_ptr(rotation)), 3 * sizeof(float));
	size_t serial_num_len = 0;
	calib_file.read(reinterpret_cast<char*>(&serial_num_len), sizeof(size_t));
	tracker_serial.resize(serial_num_len);
	calib_file.read(&tracker_serial[0], serial_num_len);

	std::cout << "Loaded calibration:\nTranslation = " << glm::to_string(translation)
		<< "\nRotation XYZ = " << glm::to_string(rotation)
		<< "\nAttached to object with serial: " << tracker_serial << std::endl;
}
void ZedCalibration::save(const std::string &calibration_file) const {
	std::cout << "Saving calibration to '" << calibration_file
		<< "':\nTranslation = " << glm::to_string(translation)
		<< "\nRotation XYZ = " << glm::to_string(rotation)
		<< "\nAttached to object with serial: " << tracker_serial << std::endl;

	std::ofstream calib_file(calibration_file, std::ios::binary);
	calib_file.write(reinterpret_cast<const char*>(glm::value_ptr(translation)), 3 * sizeof(float));
	calib_file.write(reinterpret_cast<const char*>(glm::value_ptr(rotation)), 3 * sizeof(float));
	const size_t serial_num_len = tracker_serial.size();
	calib_file.write(reinterpret_cast<const char*>(&serial_num_len), sizeof(size_t));
	calib_file.write(tracker_serial.c_str(), serial_num_len);
}
glm::mat4 ZedCalibration::tracker_to_camera() const {
	const glm::mat4 m = glm::translate(translation)
		* glm::rotate(glm::radians(rotation.y), glm::vec3(0.f, 1.f, 0.f))
		* glm::rotate(glm::radians(rotation.x), glm::vec3(1.f, 0.f, 0.f))
		* glm::rotate(glm::radians(rotation.z), glm::vec3(0.f, 0.f, 1.f));
	return m;
}

ZedManager::ZedManager(ZedCalibration calibration, std::shared_ptr<OpenVRDisplay> &vr)
	: calibration(calibration), vr(vr), tracker(vr::k_unTrackedDeviceIndexInvalid)
{
	sl::InitParameters init_params;
	init_params.camera_resolution = sl::RESOLUTION_HD720;
	init_params.camera_fps = 60;
	init_params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
	init_params.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	init_params.sdk_verbose = false;
	init_params.coordinate_units = sl::UNIT_METER;

	sl::ERROR_CODE err = camera.open(init_params);
	if (err != sl::SUCCESS) {
		std::cout << "Failed to open camera: '" << zed_error_to_string(err)
			<< "', retrying" << std::endl;
		camera.close();
		// Maybe try one more time?
		err = camera.open(init_params);
		if (err != sl::SUCCESS) {
			camera.close();
			std::cout << "Failed to open camera again: '" << zed_error_to_string(err)
				<< "', aborting" << std::endl;
			throw std::runtime_error("Failed to open ZED");
		}
	}

	sl::CameraInformation cam_info = camera.getCameraInformation();
	std::cout << "ZED serial number is: " << cam_info.serial_number
		<< "\nDepth range = [" << camera.getDepthMinRangeValue()
		<< ", " << camera.getDepthMaxRangeValue() << "]"
		<< "\nVertical FoV: " << cam_info.calibration_parameters.left_cam.v_fov
		<< " degrees\n";

	if (!calibration.tracker_serial.empty()) {
		find_tracker();
	}

	request_image(sl::VIEW_LEFT);
	request_measure(sl::MEASURE_DEPTH);
	
	const std::string res_path = get_resource_path();
	glGenVertexArrays(1, &prepass_vao);
	prepass_shader = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "zed_prepass_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "zed_prepass_frag.glsl")
	});

	glGenTextures(textures.size(), textures.data());
	for (auto &tex : textures) {
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	}
}
ZedManager::~ZedManager() {
	glDeleteVertexArrays(1, &prepass_vao);
	glDeleteTextures(textures.size(), textures.data());
	glDeleteProgram(prepass_shader);
	camera.close();
}
bool ZedManager::is_tracking() const {
	return tracker != vr::k_unTrackedDeviceIndexInvalid
		&& vr->system->IsTrackedDeviceConnected(tracker);
}
void ZedManager::find_tracker() {
	// Lookup the tracked object with the serial number we're calibrated with
	if (calibration.tracker_serial.empty()) {
		return;
	}
	for (vr::TrackedDeviceIndex_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
		if (vr->system->GetTrackedDeviceClass(i) != vr::TrackedDeviceClass_Invalid) {
			// The len here includes the null terminator
			const size_t len = vr->system->GetStringTrackedDeviceProperty(i, vr::Prop_SerialNumber_String,
					nullptr, 0);
			if (len == 0) {
				continue;
			}
			std::string serial(len - 1, ' ');
			vr->system->GetStringTrackedDeviceProperty(i, vr::Prop_SerialNumber_String, &serial[0], len);
			std::cout << "Device " << i << " serial = " << serial << "\n";
			if (calibration.tracker_serial == serial) {
				tracker = i;
				break;
			}
		}
	}
}
void ZedManager::set_tracker(vr::TrackedDeviceIndex_t &device) {
	tracker = device;
	const size_t len = vr->system->GetStringTrackedDeviceProperty(tracker, vr::Prop_SerialNumber_String,
			nullptr, 0);
	if (len == 0) { 
		std::cout << "Device index: " << device << " not valid!\n";
		return;
	}
	calibration.tracker_serial = std::string(len - 1, ' ');
	vr->system->GetStringTrackedDeviceProperty(tracker, vr::Prop_SerialNumber_String,
			&calibration.tracker_serial[0], len);
	std::cout << "Camera attached to: " << calibration.tracker_serial << "\n";
}
void ZedManager::request_image(sl::VIEW view) {
	image_requests[view] = sl::Mat();
}
void ZedManager::request_measure(sl::MEASURE measure) {
	measure_requests[measure] = sl::Mat();
}
void ZedManager::begin_render(glm::mat4 &view, glm::mat4 &projection) {
	glm::mat4 tracker_to_absolute(1);
	if (is_tracking()) {
		tracker_to_absolute = openvr_m34_to_mat4(vr->tracked_device_poses[tracker].mDeviceToAbsoluteTracking);
	}
	// TODO: Is this going to be correct? Or will the calibration I computed be wrong b/c I had the
	// matrix multiplication in the wrong order. Will this fix the ZED calibration import?
	view = calibration.tracker_to_camera() * glm::inverse(tracker_to_absolute);
	projection = camera_projection_matrix();

	// Reversed-z
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	if (camera.grab(runtime_params) == sl::SUCCESS) {
		for (auto &request : image_requests) {
			camera.retrieveImage(request.second, request.first);
		}
		for (auto &request : measure_requests) {
			camera.retrieveMeasure(request.second, request.first);
		}

		sl::Mat &color_map = image_requests[sl::VIEW_LEFT];
		sl::Mat flip_color = color_map;
		flip_image(flip_color.getPtr<uint8_t>(), flip_color.getWidth(), flip_color.getHeight(),
				flip_color.getChannels());
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textures[0]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, flip_color.getWidth(), flip_color.getHeight(),
				0, GL_BGRA, GL_UNSIGNED_BYTE, flip_color.getPtr<uint8_t>());

		sl::Mat &depth_map = measure_requests[sl::MEASURE_DEPTH];
		sl::Mat flip_depth = depth_map;
		flip_image(flip_depth.getPtr<float>(), flip_depth.getWidth(), flip_depth.getHeight(),
				flip_depth.getChannels());
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, textures[1]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, flip_depth.getWidth(), flip_depth.getHeight(),
				0, GL_RED, GL_FLOAT, flip_depth.getPtr<float>());
	}
}
void ZedManager::render_zed_prepass() {
	for (size_t i = 0; i < textures.size(); ++i) {
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, textures[i]);
	}
	glBindVertexArray(prepass_vao);
	glUseProgram(prepass_shader);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}
glm::mat4 ZedManager::camera_projection_matrix() {
	sl::CalibrationParameters calib_params = camera.getCameraInformation().calibration_parameters;
	const float z_near = camera.getDepthMinRangeValue();
	const float z_far = camera.getDepthMaxRangeValue();
	const float fov_x = glm::radians(calib_params.left_cam.h_fov);
	const float fov_y = glm::radians(calib_params.left_cam.v_fov);
	const float width = static_cast<float>(calib_params.left_cam.image_size.width);
	const float height = static_cast<float>(calib_params.left_cam.image_size.height);

	// From ZED Unity sample, modified for reversed-z
	// https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
	glm::mat4 proj(1);
	proj[0][0] = 1.0 / std::tan(fov_x * 0.5f);
	proj[1][1] = 1.0 / std::tan(fov_y * 0.5f);
	proj[2][0] = 2.0 * ((width - 1.0 * calib_params.left_cam.cx) / width) - 1.0;
	proj[2][1] = -(2.0 * ((height - 1.0 * calib_params.left_cam.cy) / height) - 1.0);
	proj[2][2] = 0.f;
	proj[2][3] = -1.0;
	proj[3][2] = z_near;
	proj[3][3] = 0.f;
	return proj;
}

std::string zed_error_to_string(const sl::ERROR_CODE &err) {
	switch (err) {
		case sl::ERROR_CODE_FAILURE: return "ERROR_CODE_FAILURE";
		case sl::ERROR_CODE_NO_GPU_COMPATIBLE: return "ERROR_CODE_NO_GPU_COMPATIBLE";
		case sl::ERROR_CODE_NOT_ENOUGH_GPUMEM: return "ERROR_CODE_NOT_ENOUGH_GPUMEM";
		case sl::ERROR_CODE_CAMERA_NOT_DETECTED: return "ERROR_CODE_CAMERA_NOT_DETECTED";
		case sl::ERROR_CODE_INVALID_RESOLUTION: return "ERROR_CODE_INVALID_RESOLUTION";
		case sl::ERROR_CODE_LOW_USB_BANDWIDTH: return "ERROR_CODE_LOW_USB_BANDWIDTH";
		case sl::ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE: return "ERROR_CODE_CALIBRATION_FILE_NOT_AVAILABLE";
		case sl::ERROR_CODE_INVALID_SVO_FILE: return "ERROR_CODE_INVALID_SVO_FILE";
		case sl::ERROR_CODE_SVO_RECORDING_ERROR: return "ERROR_CODE_SVO_RECORDING_ERROR";
		case sl::ERROR_CODE_INVALID_COORDINATE_SYSTEM: return "ERROR_CODE_INVALID_COORDINATE_SYSTEM";
		case sl::ERROR_CODE_INVALID_FIRMWARE: return "ERROR_CODE_INVALID_FIRMWARE";
		case sl::ERROR_CODE_INVALID_FUNCTION_PARAMETERS: return "ERROR_CODE_INVALID_FUNCTION_PARAMETERS";
		case sl::ERROR_CODE_NOT_A_NEW_FRAME: return "ERROR_CODE_NOT_A_NEW_FRAME";
		case sl::ERROR_CODE_CUDA_ERROR: return "ERROR_CODE_CUDA_ERROR";
		case sl::ERROR_CODE_CAMERA_NOT_INITIALIZED: return "ERROR_CODE_CAMERA_NOT_INITIALIZED";
		case sl::ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE: return "ERROR_CODE_NVIDIA_DRIVER_OUT_OF_DATE";
		case sl::ERROR_CODE_INVALID_FUNCTION_CALL: return "ERROR_CODE_INVALID_FUNCTION_CALL";
		case sl::ERROR_CODE_CORRUPTED_SDK_INSTALLATION: return "ERROR_CODE_CORRUPTED_SDK_INSTALLATION";
		case sl::ERROR_CODE_INCOMPATIBLE_SDK_VERSION: return "ERROR_CODE_INCOMPATIBLE_SDK_VERSION";
		case sl::ERROR_CODE_LAST: return "ERROR_CODE_LAST";
	}
}

