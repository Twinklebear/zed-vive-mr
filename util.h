#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include <utility>
#include <string>
#include <thread>

#include "gl_core_4_5.h"

const char PATH_SEP = '/';

inline float to_radians(float deg){
	return deg * 0.01745f;
}
template<typename T>
constexpr inline T clamp(T x, T l, T h){
	return x < l ? l : x > h ? h : x;
}
// Get the resource path for resources located under res/<sub_dir>
// sub_dir defaults to empty to just return res
std::string get_resource_path(const std::string &sub_dir = "");
// Read the contents of a file into the string
std::string get_file_content(const std::string &fname);
// Load a file's content and its includes returning the file with includes inserted
// and #line directives for better GLSL error messages within the included files
// the vector of file names will be filled with the file name for each file name number
// in the #line directive
std::string load_shader_file(const std::string &fname, std::vector<std::string> &file_names);
// Load a GLSL shader from the file. Returns -1 if loading fails and prints
// out the compilation errors
GLint load_shader(GLenum type, const std::string &file);
// Load a GLSL shader program from the shader files specified. The pair
// to specify a shader is { shader type, shader file }
// Returns -1 if program creation fails
GLint load_program(const std::vector<std::pair<GLenum, std::string>> &shader_files);
/*
 * Load an image into a 2D texture, creating a new texture id
 * The texture unit desired for this texture should be set active
 * before loading the texture as it will be bound during the loading process
 * Can also optionally pass width & height variables to return the width
 * and height of the loaded image
 */
GLuint load_texture(const std::string &file, size_t *width = nullptr, size_t *height = nullptr);
/*
 * Load a series of images into a 2D texture array, creating a new texture id
 * The images will appear in the array in the same order they're passed in
 * It is an error if the images don't all have the same dimensions
 * or have different formats
 * The texture unit desired for this texture should be set active
 * before loading the texture as it will be bound during the loading process
 * Can also optionally pass width & height variables to return the width
 * and height of the loaded image
 */
GLuint load_texture_array(const std::vector<std::string> &files, size_t *w = nullptr, size_t *h = nullptr);
void set_thread_name(std::thread &thread, const char *name);
bool check_framebuffer(GLuint fbo);

// Y-flip a WxH image with n components per pixel.
template<typename T>
void flip_image(T *img, const size_t w, const size_t h, const size_t n){
	for (size_t y = 0; y < h / 2; ++y) {
		T *row_a = img + y * w * n;
		T *row_b = img + (h - y - 1) * w * n;
		std::swap_ranges(row_a, row_a + w * n, row_b);
	}
}

