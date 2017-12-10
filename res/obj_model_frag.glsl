#version 440

out vec4 color;

in vec3 vnormal;

void main(void) {
	color = vec4((vnormal + vec3(1)) * 0.5, 1);
}

