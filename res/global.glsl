float linear_to_srgb(float x) {
	if (x <= 0.0031308) {
		return 12.92 * x;
	}
	return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}
float srgb_to_linear(float x) {
	if (x <= 0.04045) {
		return x / 12.92;
	}
	return pow((x + 0.055) / 1.055, 2.4);
}
vec3 linear_to_srgb(vec3 v) {
	return vec3(linear_to_srgb(v.x),
			linear_to_srgb(v.y),
			linear_to_srgb(v.z));
}
vec3 srgb_to_linear(vec3 v) {
	return vec3(srgb_to_linear(v.x),
			srgb_to_linear(v.y),
			srgb_to_linear(v.z));
}

layout(std140, binding = 0) uniform Viewing {
	mat4 view, proj;
	vec2 win_dims;
	vec3 eye_pos;
};


