#version 440

layout(binding = 0) uniform sampler2D imgui_tex;

out vec4 color;

in vec2 uv;

void main(void) {
	color = texture(imgui_tex, uv);
}

