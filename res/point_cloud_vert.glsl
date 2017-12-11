#version 440

#include "global.glsl"

uniform mat4 model_mat;

struct Vertex {
	float px, py, pz;
	int color;
};

layout(binding = 1, std430) buffer VertexBlock {
	Vertex vertices[];
};

out vec4 vcolor;

void main(void) {
	Vertex vert = vertices[gl_VertexID];
	gl_Position = proj * view * model_mat * vec4(vert.px, vert.py, vert.pz, 1.0);
	vcolor.b = ((vert.color & 0x00ff0000) >> 16) / 255.0;
	vcolor.g = ((vert.color & 0x0000ff00) >> 8) / 255.0;
	vcolor.r = ((vert.color & 0x000000ff) >> 0) / 255.0;
	vcolor.rgb = srgb_to_linear(vcolor.rgb);
	vcolor.a = 1;
}


