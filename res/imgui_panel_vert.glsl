#version 440

#include "global.glsl"

uniform mat4 model_mat;
uniform bool is_blit;

const vec2 quad_verts[4] = vec2[](
	vec2(-1, 1), vec2(-1, -1), vec2(1, 1), vec2(1, -1)
);

out vec2 uv;

void main(void){
	if (is_blit) {
		gl_Position = model_mat * vec4(quad_verts[gl_VertexID], 0.0, 1.0);
	} else {
		gl_Position = proj * view * model_mat * vec4(quad_verts[gl_VertexID], 0.0, 1.0);
	}
	uv = (quad_verts[gl_VertexID] + vec2(1)) * 0.5;
}


