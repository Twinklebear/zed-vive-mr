#include <iostream>
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

std::string zed_error_to_string(const sl::ERROR_CODE &err);

int main(int argc, char **argv) {
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);
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

#if 0
	sl::Camera zed;

	sl::InitParameters init_params;
	//init_params.camera_resolution = sl::RESOLUTION_HD1080;
	init_params.camera_resolution = sl::RESOLUTION_HD720;
	init_params.camera_fps = 30;
	init_params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
	init_params.sdk_verbose = false;

	sl::ERROR_CODE err = zed.open(init_params);
	if (err != sl::SUCCESS) {
		std::cout << "Failed to open camera: '" << zed_error_to_string(err)
			<< "'" << std::endl;
	}

	int zed_serial = zed.getCameraInformation().serial_number;
	std::cout << "ZED serial number is: " << zed_serial << std::endl;

	sl::RuntimeParameters runtime_params;
	runtime_params.sensing_mode = sl::SENSING_MODE_FILL;
#endif

	std::unique_ptr<OpenVRDisplay> vr = std::make_unique<OpenVRDisplay>(win);

	const std::string res_path = get_resource_path();
	ObjModel controller(res_path + "controller.obj");
	ObjModel hmd_model(res_path + "generic_hmd.obj");

	vr::TrackedDeviceIndex_t zed_controller = -1;
	std::array<vr::TrackedDeviceIndex_t, 2> controllers = {
		vr->system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_RightHand),
		vr->system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_LeftHand)
	};

	GLuint view_info_buf;
	glGenBuffers(1, &view_info_buf);
	glBindBuffer(GL_UNIFORM_BUFFER, view_info_buf);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, view_info_buf);

	glClearColor(0, 0, 0, 1);
	glEnable(GL_DEPTH_TEST);

	sl::Mat image;
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
						} else {
							zed_controller = 0;
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
			std::vector<glm::mat4> view_info(2, glm::mat4(1));
			vr->begin_render(i, view_info[0], view_info[1]);
			glBindBuffer(GL_UNIFORM_BUFFER, view_info_buf);
			glBufferData(GL_UNIFORM_BUFFER, view_info.size() * sizeof(glm::mat4),
					view_info.data(), GL_STREAM_DRAW);

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			for (size_t i = 0; i < controllers.size(); ++i) {
				if (controllers[i] != vr::k_unTrackedDeviceIndexInvalid) {
					controller.set_model_mat(controller_poses[i]);
					controller.render();
				}
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
			std::vector<glm::mat4> view_info(2, glm::mat4(1));
			view_info[0] = glm::inverse(controller_poses[zed_controller]);
				/*
				* glm::translate(glm::vec3(-0.1172772f, -0.04968496f, -0.01370035f))
				* glm::rotate(glm::radians(2.951431f), glm::vec3(1.f, 0.f, 0.f))
				* glm::rotate(glm::radians(2.932397f), glm::vec3(0.f, 1.f, 0.f))
				* glm::rotate(glm::radians(1.664886f), glm::vec3(0.f, 0.f, 1.f));
				*/

			/*
			view_info[1] = glm::perspective(glm::radians(42.8344f),
					static_cast<float>(WIN_WIDTH) / static_cast<float>(WIN_HEIGHT),
					0.1f, 150.f);
					*/
			view_info[1] = glm::perspectiveFovRH(glm::radians(75.f),
					static_cast<float>(WIN_WIDTH) , static_cast<float>(WIN_HEIGHT),
					0.1f, 150.f);
			glBindBuffer(GL_UNIFORM_BUFFER, view_info_buf);
			glBufferData(GL_UNIFORM_BUFFER, view_info.size() * sizeof(glm::mat4),
					view_info.data(), GL_STREAM_DRAW);

			for (size_t i = 0; i < controllers.size(); ++i) {
				if (controllers[i] != vr::k_unTrackedDeviceIndexInvalid) {
					controller.set_model_mat(controller_poses[i]);
					controller.render();
				}
			}
			hmd_model.set_model_mat(openvr_m34_to_mat4(vr->tracked_device_poses[0].mDeviceToAbsoluteTracking));
			hmd_model.render();
		}

#if 0
		if (zed.grab(runtime_params) == sl::SUCCESS) {
			zed.retrieveImage(image, sl::VIEW_LEFT);
		}
		glDrawPixels(image.getWidth(), image.getHeight(), GL_BGRA,
				GL_UNSIGNED_BYTE, image.getPtr<sl::uchar1>(sl::MEM_CPU));
#endif

		SDL_GL_SwapWindow(win);
	}

	glDeleteBuffers(1, &view_info_buf);
	vr = nullptr;
//	zed.close();
	SDL_GL_DeleteContext(ctx);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
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

