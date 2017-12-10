#include <iostream>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <sl/Camera.hpp>
#include "gl_core_4_5.h"
#include "debug.h"
#include "openvr_display.h"
#include "util.h"
#include "obj_model.h"

static int WIN_WIDTH = 1280;
static int WIN_HEIGHT = 720;

struct ViewInfo {
	glm::mat4 view, proj;
	glm::vec2 win_dims;
	// in std140 these are vec3 padded to vec4
	glm::vec4 eye_pos;
};

glm::mat4 zed_projection_matrix(sl::Camera &zed);
std::string zed_error_to_string(const sl::ERROR_CODE &err);

int main(int argc, char **argv) {
	sl::Camera zed;

	sl::InitParameters init_params;
	//init_params.camera_resolution = sl::RESOLUTION_HD1080;
	init_params.camera_resolution = sl::RESOLUTION_HD720;
	init_params.camera_fps = 30;
	init_params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
	init_params.sdk_verbose = false;
	init_params.coordinate_units = sl::UNIT_METER;

	sl::ERROR_CODE err = zed.open(init_params);
	if (err != sl::SUCCESS) {
		std::cout << "Failed to open camera: '" << zed_error_to_string(err)
			<< "'" << std::endl;
		zed.close();
		return 1;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);

	SDL_Window *win = SDL_CreateWindow("ZED MR Test", SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED, WIN_WIDTH, WIN_HEIGHT, SDL_WINDOW_OPENGL);
	if (!win){
		std::cout << "Failed to open SDL window: " << SDL_GetError() << "\n";
		return 1;
	}
	SDL_GLContext ctx = SDL_GL_CreateContext(win);
	if (!ctx){
		std::cout << "Failed to get OpenGL context: " << SDL_GetError() << "\n";
		return 1;
	}
	SDL_GL_SetSwapInterval(1);

	if (ogl_LoadFunctions() == ogl_LOAD_FAILED){
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Failed to Load OpenGL Functions",
				"Could not load OpenGL functions for 4.4, OpenGL 4.4 or higher is required",
				NULL);
		SDL_GL_DeleteContext(ctx);
		SDL_DestroyWindow(win);
		SDL_Quit();
		return 1;
	}

	dbg::register_debug_callback();
	glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER,
			0, GL_DEBUG_SEVERITY_NOTIFICATION, 16, "DEBUG LOG START");

	sl::CameraInformation zed_info = zed.getCameraInformation();
	int zed_serial = zed_info.serial_number;
	std::cout << "ZED serial number is: " << zed_serial
		<< "\nDepth range = [" << zed.getDepthMinRangeValue()
		<< ", " << zed.getDepthMaxRangeValue() << "]"
		<< "\nVertical FoV: "
		<< zed_info.calibration_parameters.left_cam.v_fov
		<< " degrees\n";

	sl::RuntimeParameters runtime_params;
	runtime_params.sensing_mode = sl::SENSING_MODE_FILL;

	std::unique_ptr<OpenVRDisplay> vr = std::make_unique<OpenVRDisplay>(win);

	const std::string res_path = get_resource_path();
	ObjModel controller(res_path + "controller.obj");
	ObjModel hmd_model(res_path + "generic_hmd.obj");
	ObjModel suzanne(res_path + "suzanne.obj");

	GLuint draw_camera_view = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "fullscreen_quad_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "composite_real_world_frag.glsl")
	});

	vr::TrackedDeviceIndex_t zed_controller = -1;
	vr::TrackedDeviceIndex_t user_controller = -1;
	std::array<vr::TrackedDeviceIndex_t, 2> controllers = {
		vr->system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_RightHand),
		vr->system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_LeftHand)
	};

	GLuint dummy_vao;
	glGenVertexArrays(1, &dummy_vao);

	ViewInfo view_info;
	view_info.win_dims = glm::vec2(WIN_WIDTH, WIN_HEIGHT);
	GLuint view_info_buf;
	glGenBuffers(1, &view_info_buf);
	glBindBuffer(GL_UNIFORM_BUFFER, view_info_buf);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, view_info_buf);

	glClearColor(0, 0, 0, 1);
	glEnable(GL_DEPTH_TEST);

	sl::Mat zed_img, zed_depth_map;
	GLuint zed_img_tex, zed_depth_tex;
	glGenTextures(1, &zed_img_tex);
	glGenTextures(1, &zed_depth_tex);

	glBindTexture(GL_TEXTURE_2D, zed_img_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	glBindTexture(GL_TEXTURE_2D, zed_depth_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);


	bool quit = false;
	while (!quit) {
		SDL_Event sdl_evt;
		while (SDL_PollEvent(&sdl_evt)){
			if (sdl_evt.type == SDL_QUIT
					|| (sdl_evt.type == SDL_KEYDOWN && sdl_evt.key.keysym.sym == SDLK_ESCAPE))
			{
				quit = true;
				break;
			}
		}
		vr::VREvent_t vr_evt;
		while (vr->system->PollNextEvent(&vr_evt, sizeof(vr_evt))) {
			switch (vr_evt.eventType) {
				case vr::VREvent_TrackedDeviceRoleChanged:
					controllers[0] = vr->system->GetTrackedDeviceIndexForControllerRole(
							vr::TrackedControllerRole_RightHand);
					controllers[1] = vr->system->GetTrackedDeviceIndexForControllerRole(
							vr::TrackedControllerRole_LeftHand);
					std::cout << "Got controller role change\n";
					break;
				case vr::VREvent_ButtonPress:
					if (vr_evt.data.controller.button == vr::k_EButton_SteamVR_Trigger) {
						if (vr_evt.trackedDeviceIndex == controllers[0]) {
							zed_controller = 1;
							user_controller = 0;
						} else {
							zed_controller = 0;
							user_controller = 1;
						}
						std::cout << "ZED camera is tracked by controller "
							<< zed_controller << "\n";
					}
					break;
				default:
					break;
			}
		}

		std::array<glm::mat4, 2> controller_poses;
		for (size_t i = 0; i < controllers.size(); ++i) {
			if (controllers[i] != vr::k_unTrackedDeviceIndexInvalid) {
				controller_poses[i] = openvr_m34_to_mat4(
					vr->tracked_device_poses[controllers[i]].mDeviceToAbsoluteTracking);
			}
		}

		vr->begin_frame();
		for (size_t i = 0; i < vr->render_count(); ++i){
			vr->begin_render(i, view_info.view, view_info.proj);
			glBindBuffer(GL_UNIFORM_BUFFER, view_info_buf);
			glBufferData(GL_UNIFORM_BUFFER, sizeof(ViewInfo), &view_info,
					GL_STREAM_DRAW);

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			for (size_t i = 0; i < controllers.size(); ++i) {
				if (controllers[i] != vr::k_unTrackedDeviceIndexInvalid) {
					controller.set_model_mat(controller_poses[i]);
					controller.render_vr();
				}
			}
			if (user_controller != -1 && controllers[user_controller] != vr::k_unTrackedDeviceIndexInvalid) {
				suzanne.set_model_mat(controller_poses[user_controller]);
				suzanne.render_vr();
			}
		}
		vr->display();

		glViewport(0, 0, WIN_WIDTH, WIN_HEIGHT);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		// Switch back from reversed z
		glClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
		glClearDepth(1.0f);
		glDepthFunc(GL_LESS);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Render the virtual scene from the camera's viewpoint
		if (zed_controller != vr::k_unTrackedDeviceIndexInvalid
				&& controllers[zed_controller] != vr::k_unTrackedDeviceIndexInvalid)
		{
#if 1
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, zed_img_tex);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, zed_depth_tex);

			if (zed.grab(runtime_params) == sl::SUCCESS) {
				zed.retrieveImage(zed_img, sl::VIEW_LEFT);
				zed.retrieveMeasure(zed_depth_map, sl::MEASURE_DEPTH);

				glActiveTexture(GL_TEXTURE0);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, zed_img.getWidth(), zed_img.getHeight(),
						0, GL_BGRA, GL_UNSIGNED_BYTE, zed_img.getPtr<uint8_t>());

				glActiveTexture(GL_TEXTURE1);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, zed_depth_map.getWidth(), zed_depth_map.getHeight(),
						0, GL_RED, GL_FLOAT, zed_depth_map.getPtr<float>());
			}
#endif

			glBindVertexArray(dummy_vao);
			glUseProgram(draw_camera_view);
			glDepthMask(GL_FALSE);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
			glDepthMask(GL_TRUE);

			// TODO: Parse the ZED offset calibration file for the translation, rotation
			// and FOV info
			glm::mat4 camera_offset = glm::translate(glm::vec3(-0.2166734f, -0.1612207f, 0.008288212f))
				* glm::mat4(glm::quat(glm::vec3(glm::radians(357.9935f), glm::radians(5.753352f),
								glm::radians(359.645f))));
			view_info.view = glm::inverse(controller_poses[zed_controller] * camera_offset);
			view_info.eye_pos = glm::column(view_info.view, 3);

			view_info.proj = zed_projection_matrix(zed);
			glBindBuffer(GL_UNIFORM_BUFFER, view_info_buf);
			glBufferData(GL_UNIFORM_BUFFER, sizeof(ViewInfo), &view_info,
					GL_STREAM_DRAW);

			for (size_t i = 0; i < controllers.size(); ++i) {
				if (controllers[i] != vr::k_unTrackedDeviceIndexInvalid) {
					controller.set_model_mat(controller_poses[i]);
					controller.render_mr();
				}
			}
			if (user_controller != -1 && controllers[user_controller] != vr::k_unTrackedDeviceIndexInvalid) {
				suzanne.set_model_mat(controller_poses[user_controller]);
				suzanne.render_mr();
			}
			hmd_model.set_model_mat(openvr_m34_to_mat4(vr->tracked_device_poses[0].mDeviceToAbsoluteTracking));
			hmd_model.render_mr();
		}
		SDL_GL_SwapWindow(win);
	}

	glDeleteBuffers(1, &view_info_buf);
	vr = nullptr;
	zed.close();
	SDL_GL_DeleteContext(ctx);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}
glm::mat4 zed_projection_matrix(sl::Camera &zed) {
	sl::CalibrationParameters calib_params = zed.getCameraInformation().calibration_parameters;
	const float z_near = zed.getDepthMinRangeValue();
	const float z_far = zed.getDepthMaxRangeValue();
	const float fov_x = glm::radians(calib_params.left_cam.h_fov);
	const float fov_y = glm::radians(calib_params.left_cam.v_fov);
	const float width = static_cast<float>(WIN_WIDTH);
	const float height = static_cast<float>(WIN_HEIGHT);
	// From ZED Unity sample
	glm::mat4 proj(1);
	proj[0][0] = 1.0 / std::tan(fov_x * 0.5f);
	proj[1][1] = 1.0 / std::tan(fov_y * 0.5f);
	proj[2][0] = 2.0 * ((width - 1.0 * calib_params.left_cam.cx) / width) - 1.0;
	proj[2][1] = -(2.0 * ((height - 1.0 * calib_params.left_cam.cy) / height) - 1.0);
	proj[2][2] = -(z_far + z_near) / (z_far - z_near);
	proj[2][3] = -1.0;
	proj[3][2] = -(2.0 * z_far * z_near) / (z_far - z_near);
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

