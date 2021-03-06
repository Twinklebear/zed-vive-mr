#version 440

#include "global.glsl"

layout(binding = 0) uniform sampler2D cam_color;
layout(binding = 1) uniform sampler2D cam_depth;

out vec4 color;

void main(void) {
	vec2 uv = gl_FragCoord.xy / win_dims;
	uv.y = 1.0 - uv.y;
	float depth = -texture(cam_depth, uv).x;
	if (isnan(depth)) {
		gl_FragDepth = -1.f;
		discard;
	} else if (isinf(depth)) {
		depth = clamp(depth, 0.0, 1.0);
	} else {
		vec4 depth_p = proj * vec4(0, 0, depth, 1);
		depth = depth_p.z / depth_p.w;
	}
	gl_FragDepth = depth;

	color.rgb = srgb_to_linear(texture(cam_color, uv).bgr);
	color.a = 1;
}

