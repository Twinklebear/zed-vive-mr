#version 440

#include "global.glsl"

uniform mat4 model_mat;

struct Vertex {
	float px, py, pz;
	float nx, ny, nz;
};

layout(binding = 1, std430) buffer VertexBlock {
	Vertex vertices[];
};

out vec3 vnormal;

void main(void) {
	Vertex vert = vertices[gl_VertexID];
	gl_Position = proj * view * model_mat * vec4(vert.px, vert.py, vert.pz, 1.0);
	vnormal = vec3(vert.ny, vert.nx, vert.nz);
}

