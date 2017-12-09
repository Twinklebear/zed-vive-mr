#version 440

#include "global.glsl"

uniform mat4 model_mat;

layout(location = 0) in vec3 pos;

void main(void) {
	gl_Position = proj * view * model_mat * vec4(pos, 1.0);
}

