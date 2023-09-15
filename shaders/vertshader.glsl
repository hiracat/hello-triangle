#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo; //careful there are a bunch of allignment requriements for this so make sure that a bug that you have is noit because of that
// if i start to use a nested structure of custom data types then i have to be specific about the allignment with the alignas(x) directive which forces the thing after it to be on multiple of x

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
