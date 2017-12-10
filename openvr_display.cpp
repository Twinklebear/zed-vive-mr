#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "openvr_display.h"

OpenVRDisplay::OpenVRDisplay(SDL_Window *window) : window(window) {
	vr::EVRInitError error;
	system = vr::VR_Init(&error, vr::VRApplication_Scene);
	if (error != vr::VRInitError_None) {
		std::cout << "OpenVR: Initialization error " << error << "\n";
	}

	const vr::TrackedDeviceIndex_t hmd_index = 0;
	if (!system->IsTrackedDeviceConnected(vr::k_unTrackedDeviceIndex_Hmd + hmd_index)) {
		std::cout << "OpenVR: Tracking device " << hmd_index << " not connected\n";
	}

	vr_compositor = (vr::IVRCompositor *)vr::VR_GetGenericInterface(vr::IVRCompositor_Version, &error);
	if (error != vr::VRInitError_None) {
		std::cout <<  "OpenVR: Compositor initialization error\n";
	}

	vr_chaperone = (vr::IVRChaperone *)vr::VR_GetGenericInterface(vr::IVRChaperone_Version, &error);
	if (error != vr::VRInitError_None) {
		std::cout << "OpenVR: Chaperone initialization error\n";
	}

	system->GetRecommendedRenderTargetSize(&render_dims[0], &render_dims[1]);
	std::cout << "Render target resolution: " << render_dims[0] << 'x' << render_dims[1] << '\n';

	for (int eye = 0; eye < 2; ++eye) {
		glCreateFramebuffers(1, &fb_descs[eye].render_fb);

		glCreateTextures(GL_TEXTURE_2D, 1, &fb_descs[eye].render_texture);
		glTextureStorage2D(fb_descs[eye].render_texture, 1, GL_SRGB8_ALPHA8, render_dims[0], render_dims[1]);

		glCreateTextures(GL_TEXTURE_2D, 1, &fb_descs[eye].depth_texture);
		glTextureStorage2D(fb_descs[eye].depth_texture, 1, GL_DEPTH_COMPONENT32F, render_dims[0], render_dims[1]);

		glNamedFramebufferTexture(fb_descs[eye].render_fb, GL_COLOR_ATTACHMENT0, fb_descs[eye].render_texture, 0);
		glNamedFramebufferTexture(fb_descs[eye].render_fb, GL_DEPTH_ATTACHMENT, fb_descs[eye].depth_texture, 0);
		GLenum status = glCheckNamedFramebufferStatus(fb_descs[eye].render_fb, GL_DRAW_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			std::cout << "GL: Render framebuffer incomplete\n";
		}
	}

	// reversed z buffer
	// TODO: How will reversed z effect the depth compositing we get from the camera?
	// Would it matter? We won't be viewing the composited view in VR anyways
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	/* eye to head transform and eye projection matrices */
	const float near_clip = 0.2f;
	float left, right, top, bottom;

	system->GetProjectionRaw(vr::Eye_Left, &left, &right, &top, &bottom);
	matrices.projection_eyes[0] = mk_projection_mat(left, right, top, bottom, near_clip);
	matrices.head_to_eyes[0]    = glm::inverse(openvr_m34_to_mat4(system->GetEyeToHeadTransform(vr::Eye_Left)));

	system->GetProjectionRaw(vr::Eye_Right, &left, &right, &top, &bottom);
	matrices.projection_eyes[1] = mk_projection_mat(left, right, top, bottom, near_clip);
	matrices.head_to_eyes[1] = glm::inverse(openvr_m34_to_mat4(system->GetEyeToHeadTransform(vr::Eye_Right)));
}
OpenVRDisplay::~OpenVRDisplay() {
	for (int eye = 0; eye < 2; ++eye) {
		glDeleteTextures(1, &fb_descs[eye].depth_texture);

		glDeleteTextures(1, &fb_descs[eye].render_texture);
		glDeleteFramebuffers(1, &fb_descs[eye].render_fb);
	}

	vr::VR_Shutdown();
}
void OpenVRDisplay::begin_frame() {
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	/* this is necessary for rendering to work properly (without it there is strange flicker) */
	vr_compositor->WaitGetPoses(tracked_device_poses, vr::k_unMaxTrackedDeviceCount, NULL, 0);

	/* update device position matrix (device -> world, so we need to take inverse) */
	/* TODO: maybe move into begin_frame */
	matrices.absolute_to_device = glm::inverse(openvr_m34_to_mat4(tracked_device_poses[0].mDeviceToAbsoluteTracking));
}

size_t OpenVRDisplay::render_count() {
	return 2;
}
void OpenVRDisplay::begin_render(const size_t iteration, glm::mat4 &view, glm::mat4 &projection) {
	glEnable(GL_FRAMEBUFFER_SRGB);
	glBindFramebuffer(GL_FRAMEBUFFER, fb_descs[iteration].render_fb);

	glViewport(0, 0, render_dims[0], render_dims[1]);

	view       = matrices.head_to_eyes[iteration] * matrices.absolute_to_device;
	projection = matrices.projection_eyes[iteration];
}
void OpenVRDisplay::display() {
	/* TODO: this is constant and can be cached */
	vr::Texture_t left_eye = {};
	left_eye.handle      = (void *)fb_descs[0].render_texture;
	left_eye.eType       = vr::TextureType_OpenGL;
	left_eye.eColorSpace = vr::ColorSpace_Gamma;

	vr::Texture_t right_eye = {};
	right_eye.handle = (void *)fb_descs[1].render_texture;
	right_eye.eType = vr::TextureType_OpenGL;
	right_eye.eColorSpace = vr::ColorSpace_Gamma;
	vr_compositor->Submit(vr::Eye_Left, &left_eye, NULL, vr::Submit_Default);

	vr_compositor->Submit(vr::Eye_Right, &right_eye, NULL, vr::Submit_Default);

	glFlush();
}
glm::mat4 openvr_m34_to_mat4(const vr::HmdMatrix34_t &t) {
	return glm::mat4(
		t.m[0][0], t.m[1][0], t.m[2][0], 0.0f,
		t.m[0][1], t.m[1][1], t.m[2][1], 0.0f,
		t.m[0][2], t.m[1][2], t.m[2][2], 0.0f,
		t.m[0][3], t.m[1][3], t.m[2][3], 1.0f
	);
}

glm::mat4 mk_projection_mat(const float left, const float right,
		const float top, const float bottom, const float near_clip)
{
	return glm::mat4(
		2.0f/(right - left), 0.0f, 0.0f, 0.0f,
		0.0f, 2.0f/(bottom - top), 0.0f, 0.0f,
		(right + left)/(right - left), (bottom + top)/(bottom - top), 0.0f, -1.0f,
		0.0f, 0.0f, near_clip, 0.0f
	);
}

