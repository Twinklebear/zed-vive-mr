#ifndef VISLIGHT_OPENVR_DISPLAY_H
#define VISLIGHT_OPENVR_DISPLAY_H

#include <SDL.h>
#include <glm/glm.hpp>
#include <openvr.h>

#include "gl_core_4_5.h"

struct FBDesc {
	GLuint render_fb;
	GLuint render_texture;
	GLuint depth_texture;
};

struct Matrices {
	/* head_to_eyes and projection_eyes do not change over time, but can't be
	   combined, as the view matrix is head_to_eyes*absolute_to_device */
	glm::mat4 head_to_eyes[2];
	glm::mat4 projection_eyes[2];
	glm::mat4 absolute_to_device;
};

class OpenVRDisplay {
public:
	OpenVRDisplay(SDL_Window *window);
	~OpenVRDisplay();
	void begin_frame();
	size_t render_count();
	void begin_render(const size_t iteration, glm::mat4 &view, glm::mat4 &projection);
	void display();

	vr::IVRSystem *system;
	SDL_Window *window;
	struct FBDesc fb_descs[2];
	struct Matrices matrices;
	uint32_t render_dims[2];
	vr::IVRCompositor *vr_compositor;
	vr::TrackedDevicePose_t tracked_device_poses[vr::k_unMaxTrackedDeviceCount];
	vr::IVRChaperone *vr_chaperone;
};

#endif
