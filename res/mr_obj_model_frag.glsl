#version 440

#include "global.glsl"

layout(binding = 0) uniform sampler2D cam_color;
layout(binding = 1) uniform sampler2D cam_depth;

out vec4 color;

in vec3 vnormal;
in vec3 proj_pos;

void main(void) {
	color = vec4((vnormal + vec3(1)) * 0.5, 1);

	vec2 uv = gl_FragCoord.xy / win_dims;
	// Y-flip for textures
	uv.y = 1.0 - uv.y;
	float depth = texture(cam_depth, uv).x;
	depth = -depth * 2 + 1;
	vec4 depth_p = proj * vec4(0, 0, depth, 1);
	depth = depth_p.z / depth_p.w;
	if (!isinf(depth) && !isnan(depth)) {
		if (gl_FragCoord.z >= depth) {
			discard;
		}
	}
}


