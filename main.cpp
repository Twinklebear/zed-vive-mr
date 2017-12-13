#include <iostream>
#include <fstream>
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
#include "zed_point_cloud.h"
#include "imgui-1.52/imgui_impl_sdl_gl3.h"

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
	bool calibrating = false;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--calibrate") == 0) {
			calibrating = true;
		}
	}

	sl::Camera zed;
	sl::InitParameters init_params;
	//init_params.camera_resolution = sl::RESOLUTION_HD1080;
	init_params.camera_resolution = sl::RESOLUTION_HD720;
	init_params.camera_fps = 60;
	init_params.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
	init_params.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
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

	ImGui_ImplSdlGL3_Init(win);

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
	if (!calibrating) {
		runtime_params.sensing_mode = sl::SENSING_MODE_FILL;
	}

	std::unique_ptr<OpenVRDisplay> vr = std::make_unique<OpenVRDisplay>(win);

	const std::string res_path = get_resource_path();
	ObjModel controller(res_path + "controller.obj");
	ObjModel hmd_model(res_path + "generic_hmd.obj");
	ObjModel suzanne(res_path + "suzanne.obj");
	ObjModel uv_sphere(res_path + "uv_sphere.obj");
	uv_sphere.set_model_mat(glm::translate(glm::vec3(0.f, 1.5f, 1.f)) * glm::scale(glm::vec3(0.25f)));

	PointCloud point_cloud;

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

	glClearColor(0, 0, 0, 0);
	glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	const int IMGUI_TEX_DIMS = 512;
	GLuint imgui_fbo, imgui_tex;
	glGenTextures(1, &imgui_tex);
	glBindTexture(GL_TEXTURE_2D, imgui_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, IMGUI_TEX_DIMS, IMGUI_TEX_DIMS);

	glGenFramebuffers(1, &imgui_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, imgui_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, imgui_tex, 0);
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLuint imgui_panel_shader = load_program({
		std::make_pair(GL_VERTEX_SHADER, res_path + "imgui_panel_vert.glsl"),
		std::make_pair(GL_FRAGMENT_SHADER, res_path + "imgui_panel_frag.glsl")
	});
	GLuint imgui_model_mat_unif = glGetUniformLocation(imgui_panel_shader, "model_mat");
	GLuint imgui_is_blit_unif = glGetUniformLocation(imgui_panel_shader, "is_blit");
	glUseProgram(imgui_panel_shader);
	glUniform1i(imgui_is_blit_unif, 0);

#if 0
	const glm::mat4 swap_handedness = glm::mat4(
		glm::vec4(-1, 0, 0, 0),
		glm::vec4(0, 1, 0, 0),
		glm::vec4(0, 0, -1, 0),
		glm::vec4(0, 0, 0, 1)
	);

	glm::vec3 calibration_translation = glm::vec3(-0.089805, 0.007778, -0.024980);
	glm::vec3 calibration_rotation = glm::vec3(0.460165, 1.300000, 1.316456);
	glm::mat4 camera_offset = glm::translate(calibration_translation)
		* glm::rotate(glm::radians(calibration_rotation.y), glm::vec3(0.f, 1.f, 0.f))
		* glm::rotate(glm::radians(calibration_rotation.x), glm::vec3(1.f, 0.f, 0.f))
		* glm::rotate(glm::radians(calibration_rotation.z), glm::vec3(0.f, 0.f, 1.f));
	camera_offset = swap_handedness * camera_offset * swap_handedness;
#else
	// Will: Calibration params for the 3D printed mount:
	// translation = vec3(-0.058000, 0.046000, -0.101000)
	// rotation = vec3(4.899998, 0.600000, -2.100000)
	glm::vec3 calibration_translation = glm::vec3(-0.058000, 0.046000, -0.101000);
	glm::vec3 calibration_rotation = glm::vec3(4.899998, 0.600000, -2.100000);
	glm::mat4 camera_offset = glm::translate(calibration_translation)
		* glm::rotate(glm::radians(calibration_rotation.y), glm::vec3(0.f, 1.f, 0.f))
		* glm::rotate(glm::radians(calibration_rotation.x), glm::vec3(1.f, 0.f, 0.f))
		* glm::rotate(glm::radians(calibration_rotation.z), glm::vec3(0.f, 0.f, 1.f));
#endif

	size_t vr_selected_calibration_input = 0;
	std::array<float*, 6> calibration_inputs = {
		&calibration_translation.x, &calibration_translation.y, &calibration_translation.z,
		&calibration_rotation.x, &calibration_rotation.y, &calibration_rotation.z,
	};
	const std::array<float, 2> calibration_ui_steps = { 0.001, 0.1 };
	const std::array<std::string, 6> calibration_ui_labels = {
		"t.x", "t.y", "t.z", "r.x", "r.y", "r.z"
	};
	glm::vec2 press_pos;
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
			ImGui_ImplSdlGL3_ProcessEvent(&sdl_evt);
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
					if (user_controller != -1 && vr_evt.trackedDeviceIndex == controllers[user_controller]) {
						if (vr_evt.data.controller.button == vr::k_EButton_SteamVR_Touchpad) {
							vr::VRControllerState_t state;
							vr->system->GetControllerState(vr_evt.trackedDeviceIndex, &state, sizeof(state));
							// Axis 0 is the touchpad thumb press location
							press_pos = glm::vec2(state.rAxis[0].x, state.rAxis[0].y);
							if (glm::length(press_pos) <= 0.15) {
								*calibration_inputs[vr_selected_calibration_input] = 0;
							} else if (press_pos.y <= -0.8) {
								vr_selected_calibration_input = (vr_selected_calibration_input + 1) % 6;
							} else if (press_pos.y >= 0.8) {
								if (vr_selected_calibration_input == 0) {
									vr_selected_calibration_input = 5;
								} else {
									vr_selected_calibration_input = (vr_selected_calibration_input - 1) % 6;
								}
							}
						}
					}
					break;
				case vr::VREvent_ButtonUnpress:
					if (user_controller != -1 && vr_evt.trackedDeviceIndex == controllers[user_controller]) {
						if (vr_evt.data.controller.button == vr::k_EButton_SteamVR_Touchpad) {
							press_pos = glm::vec2(0);
						}
					}
					break;
				default:
					break;
			}
		}

		// Clicking on the right, left, top, bottoms of the touchpad
		if (press_pos.x >= 0.8) {
			*calibration_inputs[vr_selected_calibration_input]
				+= calibration_ui_steps[vr_selected_calibration_input / 3];
		} else if (press_pos.x <= -0.8) {
			*calibration_inputs[vr_selected_calibration_input]
				-= calibration_ui_steps[vr_selected_calibration_input / 3];
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, imgui_fbo);
		glClear(GL_COLOR_BUFFER_BIT);
		// Render UI to a texture we can show in VR as well
		ImGui_ImplSdlGL3_NewFrame(win);
		ImGuiIO &io = ImGui::GetIO();
		// Hack to render to a smaller ui texture panel
		io.DisplaySize = ImVec2(IMGUI_TEX_DIMS, IMGUI_TEX_DIMS);
		io.DisplayFramebufferScale = ImVec2(1, 1);
		if (ImGui::Begin("Calibration Settings")) {
			for (size_t i = 0; i < calibration_inputs.size(); ++i) {
				if (i == vr_selected_calibration_input) {
					ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.8, 0.2, 0.2, 1.0));
				}
				ImGui::InputFloat(calibration_ui_labels[i].c_str(),
						calibration_inputs[i], calibration_ui_steps[i / 3]);
				if (i == vr_selected_calibration_input) {
					ImGui::PopStyleColor();
				}
			}
		}
		ImGui::End();
		glDisable(GL_FRAMEBUFFER_SRGB);
		ImGui::Render();
		glEnable(GL_FRAMEBUFFER_SRGB);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

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
					controller.render();
				}
			}
			if (user_controller != -1 && controllers[user_controller] != vr::k_unTrackedDeviceIndexInvalid) {
				suzanne.set_model_mat(controller_poses[user_controller]);
				suzanne.render();
			}
			uv_sphere.render();

			if (calibrating) {
				point_cloud.render();

				if (user_controller != -1 && controllers[user_controller] != vr::k_unTrackedDeviceIndexInvalid) {
					glDisable(GL_FRAMEBUFFER_SRGB);
					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, imgui_tex);
					glUseProgram(imgui_panel_shader);
					const glm::mat4 imgui_mat = controller_poses[user_controller]
						* glm::translate(glm::vec3(-0.05f, 0.0f, 0.32f))
						* glm::rotate(glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f))
						* glm::scale(glm::vec3(0.3, 0.3, 1.f));
					glUniform1i(imgui_is_blit_unif, 0);
					glUniformMatrix4fv(imgui_model_mat_unif, 1, GL_FALSE, glm::value_ptr(imgui_mat));
					glDisable(GL_DEPTH_TEST);
					glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
					glEnable(GL_DEPTH_TEST);
					glEnable(GL_FRAMEBUFFER_SRGB);
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

				if (calibrating) {
					sl::Mat points;
					zed.retrieveMeasure(points, sl::MEASURE_XYZRGBA);
					point_cloud.update_point_cloud(points);
				}
			}

			camera_offset = glm::translate(calibration_translation)
				* glm::rotate(glm::radians(calibration_rotation.y), glm::vec3(0.f, 1.f, 0.f))
				* glm::rotate(glm::radians(calibration_rotation.x), glm::vec3(1.f, 0.f, 0.f))
				* glm::rotate(glm::radians(calibration_rotation.z), glm::vec3(0.f, 0.f, 1.f));

			// Would using the OpenVR SDK's ApplyTransform be better here? To account for the motion of
			// the camera mount?
			view_info.view = glm::inverse(glm::inverse(camera_offset) * controller_poses[zed_controller]);
			view_info.eye_pos = glm::column(view_info.view, 3);

			point_cloud.set_model_mat(glm::inverse(view_info.view));
			view_info.proj = zed_projection_matrix(zed);

			glBindBuffer(GL_UNIFORM_BUFFER, view_info_buf);
			glBufferData(GL_UNIFORM_BUFFER, sizeof(ViewInfo), &view_info,
					GL_STREAM_DRAW);

			glBindVertexArray(dummy_vao);
			glUseProgram(draw_camera_view);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

			if (user_controller != -1 && controllers[user_controller] != vr::k_unTrackedDeviceIndexInvalid) {
				controller.set_model_mat(controller_poses[user_controller]);
				controller.render();

				suzanne.set_model_mat(controller_poses[user_controller]);
				suzanne.render();
			}
			hmd_model.set_model_mat(openvr_m34_to_mat4(vr->tracked_device_poses[0].mDeviceToAbsoluteTracking));
			hmd_model.render();
			uv_sphere.render();
		}

		glDisable(GL_FRAMEBUFFER_SRGB);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, imgui_tex);
		glUseProgram(imgui_panel_shader);
		glm::mat4 imgui_blit_mat = glm::translate(glm::vec3(-0.6, 0.288888f, 0.f))
			* glm::scale(glm::vec3(0.5625f * 0.711, 0.711f, 1.f));
		glUniform1i(imgui_is_blit_unif, 1);
		glUniformMatrix4fv(imgui_model_mat_unif, 1, GL_FALSE, glm::value_ptr(imgui_blit_mat));
		glDisable(GL_DEPTH_TEST);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_FRAMEBUFFER_SRGB);

		SDL_GL_SwapWindow(win);
	}

	if (calibrating && zed_controller != -1 && controllers[zed_controller] != vr::k_unTrackedDeviceIndexInvalid) {
		const size_t serial_num_len = vr->system->GetStringTrackedDeviceProperty(controllers[zed_controller],
				vr::Prop_SerialNumber_String, nullptr, 0);
		std::vector<char> serial_num(serial_num_len, '\0');
		vr->system->GetStringTrackedDeviceProperty(controllers[zed_controller], vr::Prop_SerialNumber_String,
				serial_num.data(), serial_num_len);

		std::cout << "Calibration:\nTranslation = " << glm::to_string(calibration_translation)
			<< "\nRotation XYZ = " << glm::to_string(calibration_rotation)
			<< "\nAttached to object with serial: " << serial_num.data() << std::endl;

		std::ofstream calib_out_file("zed_mr_calibration.bin", std::ios::binary);
		calib_out_file.write(reinterpret_cast<char*>(glm::value_ptr(calibration_translation)), 3 * sizeof(float));
		calib_out_file.write(reinterpret_cast<char*>(glm::value_ptr(calibration_rotation)), 3 * sizeof(float));
		calib_out_file.write(reinterpret_cast<const char*>(&serial_num_len), sizeof(size_t));
		calib_out_file.write(serial_num.data(), serial_num.size());
	}

	ImGui_ImplSdlGL3_Shutdown();
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

