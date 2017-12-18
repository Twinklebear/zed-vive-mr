#include <fstream>
#include <iostream>
#include "util.h"
#include "zed_manager.h"

ZedCalibration::ZedCalibration() : translation(0.f), rotation(0.f), fov(0.f) {}
ZedCalibration::ZedCalibration(const std::string &calibration_file) {
	load_calibration(calibration_file);
	std::cout << "Loaded calibration:\nTranslation = " << glm::to_string(translation)
		<< "\nRotation XYZ = " << glm::to_string(rotation)
		<< "\nAttached to object with serial: " << tracker_serial << std::endl;
}
void ZedCalibration::save(const std::string &calibration_file) const {
	std::cout << "Saving calibration to '" << calibration_file
		<< "':\nTranslation = " << glm::to_string(translation)
		<< "\nRotation XYZ = " << glm::to_string(rotation)
		<< "\nAttached to object with serial: " << tracker_serial << std::endl;
	save_calibration(calibration_file);
}
glm::mat4 ZedCalibration::tracker_to_camera() const {
	const glm::mat4 unity_swap_handedness(
			-1.f, 0.f, 0.f, 0.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, -1.f, 0.f,
			0.f, 0.f, 0.f, 1.f);
	const glm::mat4 m = glm::translate(translation)
		* glm::rotate(glm::radians(rotation.y), glm::vec3(0.f, 1.f, 0.f))
		* glm::rotate(glm::radians(rotation.x), glm::vec3(1.f, 0.f, 0.f))
		* glm::rotate(glm::radians(rotation.z), glm::vec3(0.f, 0.f, 1.f));
	return unity_swap_handedness * m * unity_swap_handedness;
}
void ZedCalibration::load_calibration(const std::string &file) {
	std::ifstream calib_file(file.c_str());
	std::string line;
	std::getline(calib_file, line);
	if (line != "[Calibration]") {
		throw std::runtime_error("Invalid ZED calibration file");
	}
	while (std::getline(calib_file, line)) {
		auto fnd = line.find('=');
		if (fnd == std::string::npos) {
			throw std::runtime_error("Invalid key in '" + file + "'");
		}
		const std::string key = line.substr(0, fnd);
		const std::string value = line.substr(fnd + 1);
		if (key == "x") {
			translation.x = std::stof(value);
		} else if (key == "y") {
			translation.y = std::stof(value);
		} else if (key == "z") {
			translation.z = std::stof(value);
		} else if (key == "rx") {
			rotation.x = std::stof(value);
		} else if (key == "ry") {
			rotation.y = std::stof(value);
		} else if (key == "rz") {
			rotation.z = std::stof(value);
		} else if (key == "fov") {
			fov = std::stof(value);
		} else if (key == "indexController") {
			tracker_serial = value;
		} else {
			std::cout << "Warning: Unrecognized key in '"
				<< file << "':" << key << "\n";
		}
	}
}
void ZedCalibration::save_calibration(const std::string &file) const {
	std::ofstream calib_file(file.c_str());
	calib_file << "[Calibration]"
		<< "\nx=" << translation.x
		<< "\ny=" << translation.y
		<< "\nz=" << translation.z
		<< "\nrx=" << rotation.x
		<< "\nry=" << rotation.y
		<< "\nry=" << rotation.z
		// FoV is ignored in their unity importer, unsure why it's in the file
		// maybe some legacy reason. Writing it for compatability
		<< "\nfov=" << fov
		<< "\nindexController=" << tracker_serial;
}

ZedManager::ZedManager(ZedCalibration calibration, std::shared_ptr<OpenVRDisplay> &vr)
	: calibration(calibration), vr(vr), tracker(vr::k_unTrackedDeviceIndexInvalid)
{
	sl::InitParameters init_params;
	init_params.camera_resolution = sl::RESOLUTION_HD1080;
	init_params.camera_fps = 60;
	init_params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
	init_params.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	init_params.depth_minimum_distance = 0.7;
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
		<< " degrees\nHorizontal FoV: " << cam_info.calibration_parameters.left_cam.h_fov
		<< " degrees\n";

	// The ZED Unity samples don't actually use the fov in the calibration file
	calibration.fov = cam_info.calibration_parameters.left_cam.v_fov;

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

	is_copying = false;
	copy_target = 1;
	sl::CalibrationParameters calib_params = cam_info.calibration_parameters;
	const size_t width = static_cast<float>(calib_params.left_cam.image_size.width);
	const size_t height = static_cast<float>(calib_params.left_cam.image_size.height);

	glGenTextures(color_textures.size(), color_textures.data());
	for (size_t i = 0; i < color_textures.size(); ++i) {
		GLuint tex = color_textures[i];
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		glBindTexture(GL_TEXTURE_2D, tex);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
		cudaError_t cu_err = cudaGraphicsGLRegisterImage(&cuda_color_tex_refs[i], tex,
				GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		if (cu_err != 0) {
			std::cout << "CUDA error making color resource" << std::endl;
		}
	}

	glGenTextures(depth_textures.size(), depth_textures.data());
	for (size_t i = 0; i < depth_textures.size(); ++i) {
		GLuint tex = depth_textures[i];
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		glBindTexture(GL_TEXTURE_2D, tex);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);
		cudaError_t cu_err = cudaGraphicsGLRegisterImage(&cuda_depth_tex_refs[i], tex,
				GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		if (cu_err != 0) {
			std::cout << "CUDA error making depth resource" << std::endl;
		}
	}

	for (auto &s : cuda_streams) {
		cudaStreamCreate(&s);
	}
	for (auto &e : cuda_events) {
		cudaEventCreate(&e);
	}
}
ZedManager::~ZedManager() {
	for (auto &cu_res : cuda_color_tex_refs) {
		cudaGraphicsUnregisterResource(cu_res);
	}
	for (auto &cu_res : cuda_depth_tex_refs) {
		cudaGraphicsUnregisterResource(cu_res);
	}
	for (auto &s : cuda_streams) {
		cudaStreamDestroy(s);
	}
	for (auto &e : cuda_events) {
		cudaEventDestroy(e);
	}

	glDeleteVertexArrays(1, &prepass_vao);
	glDeleteTextures(color_textures.size(), color_textures.data());
	glDeleteTextures(depth_textures.size(), depth_textures.data());
	glDeleteProgram(prepass_shader);
	// Need to free all the GPU side memory of our images/measures
	// before freeing the camera since it will close the CUDA context
	image_requests.clear();
	measure_requests.clear();
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
	view = calibration.tracker_to_camera() * glm::inverse(tracker_to_absolute);
	projection = camera_projection_matrix();

	// Reversed-z
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	if (is_copying && cudaEventQuery(cuda_events[0]) == cudaSuccess && cudaEventQuery(cuda_events[1]) == cudaSuccess) {
		is_copying = false;
		cudaGraphicsUnmapResources(1, &cuda_color_tex_refs[copy_target]);
		cudaGraphicsUnmapResources(1, &cuda_depth_tex_refs[copy_target]);
		copy_target = (copy_target + 1) % 2;
	}

	if (!is_copying && camera.grab(runtime_params) == sl::SUCCESS) {
		is_copying = true;

		for (auto &request : image_requests) {
			camera.retrieveImage(request.second, request.first, sl::MEM_GPU);
		}
		for (auto &request : measure_requests) {
			camera.retrieveMeasure(request.second, request.first, sl::MEM_GPU);
		}

		// TODO: Can we register the resources together as an array and map just once?
		// TODO: Will doing this copy async help us avoid dropping the frame?
		sl::Mat &color_map = image_requests[sl::VIEW_LEFT];
		cudaArray_t mapped_array;
		cudaGraphicsMapResources(1, &cuda_color_tex_refs[copy_target]);
		cudaGraphicsSubResourceGetMappedArray(&mapped_array, cuda_color_tex_refs[copy_target], 0, 0);
		cudaMemcpy2DToArrayAsync(mapped_array, 0, 0, color_map.getPtr<uint8_t>(sl::MEM_GPU),
				color_map.getStepBytes(sl::MEM_GPU), color_map.getStepBytes(sl::MEM_GPU),
				color_map.getHeight(), cudaMemcpyDeviceToDevice, cuda_streams[0]);
		cudaEventRecord(cuda_events[0], cuda_streams[0]);

		sl::Mat &depth_map = measure_requests[sl::MEASURE_DEPTH];
		cudaGraphicsMapResources(1, &cuda_depth_tex_refs[copy_target]);
		cudaGraphicsSubResourceGetMappedArray(&mapped_array, cuda_depth_tex_refs[copy_target], 0, 0);
		cudaMemcpy2DToArrayAsync(mapped_array, 0, 0, depth_map.getPtr<float>(sl::MEM_GPU),
				depth_map.getStepBytes(sl::MEM_GPU), depth_map.getStepBytes(sl::MEM_GPU),
				depth_map.getHeight(), cudaMemcpyDeviceToDevice, cuda_streams[1]);
		cudaEventRecord(cuda_events[1], cuda_streams[1]);
	}
}
void ZedManager::render_zed_prepass() {
	size_t render_texture = (copy_target + 1) % 2;
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, color_textures[render_texture]);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth_textures[render_texture]);

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

