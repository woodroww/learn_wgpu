// vertex shader

struct CameraUniform {
	view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
	@builtin(position) clip_position: vec4<f32>,
	@location(0) tex_coords: vec2<f32>,
};

struct InstanceInput {
	@location(5) model_matrix_0: vec4<f32>,
	@location(6) model_matrix_1: vec4<f32>,
	@location(7) model_matrix_2: vec4<f32>,
	@location(8) model_matrix_3: vec4<f32>,
}

@vertex
fn vs_main(
	model: VertexInput,
	instance: InstanceInput,
) -> VertexOutput {
	let model_matrix = mat4x4<f32>(
		instance.model_matrix_0,
		instance.model_matrix_1,
		instance.model_matrix_2,
		instance.model_matrix_3,
	);
	var out: VertexOutput;
	out.tex_coords = model.tex_coords;
	out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
	return out;
}

// fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

	let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
	let object_normal: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

	// we don't want or need much ambient light, so 0.1 is fine
	let ambient_strength = 0.1;
	let ambient_color = light.color * ambient_strength;

	// create the lighting vectors
	let tanget_normal = object_normal.xyz * 2.0 - 1.0;
	let light_dir = normalize(light.position - in.world_position);
	let view_dir = normalize(camera.view_pos.xyz - in.world_position);
	let half_dir = normalize(view_dir + light_dir);

	let diffuse_strength = max(dot(tanget_normal, light_dir), 0.0);
	let diffuse_color = light.color * diffuse_strength;

	let specular_strength = pow(max(dot(tanget_normal, half_dir), 0.0), 32.0);
	let specular_color = specular_strength * light.color;

	let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

	return vec4<f32>(result, object_color.a);
}
