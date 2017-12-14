#version 440

const vec2 quad_verts[4] = vec2[](
	vec2(-1, 1), vec2(-1, -1), vec2(1, 1), vec2(1, -1)
);

void main(void){
	gl_Position = vec4(quad_verts[gl_VertexID], 0.0, 1.0);
}

