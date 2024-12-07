use std::{
    f32::consts::PI, ffi::{CStr, CString}, fs::File, io::{BufReader, Read}, iter::Peekable, path::Path
};

use anyhow::{anyhow, bail, Result};
use ash::{vk, Device};
use glam::{Mat4, Vec2, Vec3, Vec4};
use obj::Obj;
use toml::{Table, Value};

use crate::scene::{type_lexer::{Token, TokenIter}, Scene};

const MESHES_DIR: &str = "resources/meshes";
const SHADERS_DIR: &str = "resources/shaders";

#[derive(Clone)]
pub struct MeshScene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub objects: Vec<Object>,
    pub meshes: Vec<Obj>,
    pub light_meshes: Vec<(Obj, u32)>,

    pub raygen_shader: Shader,
    pub miss_shader: Shader,
    pub emitter_hit_shader: Option<Shader>,
    pub hit_shaders: Vec<Shader>,
}

#[derive(Clone)]
pub struct Camera {
    // matrix from world space to camera space
    pub view: Mat4,

    // matrix from camera to clip space
    pub perspective: Mat4,
}

#[derive(Clone)]
pub enum Light {
    Point {
        color: Vec3,
        position: Vec3,
    },
    Triangle {
        color: Vec3,
        vertices: [Vec3; 3],
        normal: Vec3,
    },
}

#[derive(Clone)]
pub enum Shader {
    Uncompiled(CString, Box<[u32]>),
    Compiled(CString, vk::ShaderModule),
}

#[derive(Clone)]
pub struct Object {
    pub transform: Mat4,

    // its just like aris fr
    pub mesh_i: usize,
    pub brdf_i: usize,
    pub brdf_params: Vec<u8>,
    pub alignment: usize,
}

#[derive(Debug, PartialEq)]
enum ShaderType {
    Float,
    Vec3,
    UInt,
    Int,
    Array(Box<ShaderType>, u64),
}

#[derive(Debug)]
pub enum MeshSceneUpdate {
    NewView(Mat4),
    NewAspectRatio(f32),
    NewFovDegrees(f32),
}

impl Scene for MeshScene {
    type Updates = [MeshSceneUpdate];
}

impl Shader {
    pub fn module(&self) -> vk::ShaderModule {
        let Shader::Compiled(_, module) = self else {
            panic!("shader is not compiled")
        };

        *module
    }

    pub fn name(&self) -> &CStr {
        match self {
            Shader::Uncompiled(name, _) => name,
            Shader::Compiled(name, _) => name,
        }
    }

    pub fn compile(self, device: Device) -> Result<Self> {
        match self {
            Shader::Uncompiled(name, code) => {
                let create_info = vk::ShaderModuleCreateInfo {
                    code_size: code.len() * size_of::<u32>(),
                    p_code: code.as_ptr(),
                    ..Default::default()
                };

                let module = unsafe { device.create_shader_module(&create_info, None) }?;
                Ok(Shader::Compiled(name, module))
            },
            x @ Shader::Compiled(..) => Ok(x),
        }
    }
}

impl MeshScene {
    pub const MAX_LIGHTS: u32 = 1000;

    pub fn load_from(mut reader: impl Read) -> Result<Self> {
        let mut toml_conf = String::new();
        reader.read_to_string(&mut toml_conf)?;

        let conf: Table = toml_conf.parse()?;

        let camera = Self::parse_toml_camera(&conf)?;

        let mut light_meshes = Vec::new();
        let lights = Self::parse_toml_lights(&conf, &mut light_meshes)?;

        Ok(Self {
            camera,
            lights: Vec::new(),
            objects: Vec::new(),
            meshes: Vec::new(),
            raygen_shader: Shader::Uncompiled(c"raygen".to_owned(), Default::default()),
            miss_shader: Shader::Uncompiled(c"miss".to_owned(), Default::default()),
            hit_shaders: Vec::new(),
            light_meshes,
            emitter_hit_shader: None,
        })
    }

    pub fn parse_toml_lights(conf: &Table, meshes: &mut Vec<(Obj, u32)>) -> Result<Vec<Light>> {
        let Value::Array(light_confs) = conf.get("light").ok_or(anyhow!("no lights provided"))? else {
            bail!("lights field must be an array of lights");
        };

        let mut lights = Vec::new();

        for light_conf in light_confs {
            let Value::String(light_type) = light_conf.get("type").ok_or(anyhow!("no type field found for light"))? else {
                bail!("light type must be a string");
            };

            let color = Self::parse_toml_vec3(conf.get("color").ok_or(anyhow!("no color field found for light"))?)?;

            match light_type.as_str() {
                "point" => {
                    let position = Self::parse_toml_vec3(conf.get("position").ok_or(anyhow!("no top_left field found for light"))?)?;
                    lights.push(Light::Point { color, position, });
                },
                "area" => {
                    let transform = Self::parse_toml_transform(conf.get("transform").ok_or(anyhow!("no transform field provided"))?)?;
                    let mesh_file = Self::parse_toml_file(MESHES_DIR, conf.get("mesh").ok_or(anyhow!("no mesh file provided"))?)?;
                    let reader = BufReader::new(mesh_file);

                    let mesh: Obj = obj::load_obj(reader)?;

                    let start_idx = lights.len();

                    // load triangles to get triangle lights
                    let triangles = mesh.indices.chunks_exact(3);
                    if !triangles.remainder().is_empty() {
                        bail!("obj face list was not a multiple of 3 in length")
                    }
                    for triangle in triangles {
                        let vertices: Vec<_> = triangle.into_iter().map(|&i| {
                            let pos = Vec4::from((Vec3::from_array(mesh.vertices[i as usize].position), 1f32));
                            let v = transform * pos;

                            Vec3::new(v.x, v.y, v.z)
                        }).collect();
                        let e1 = vertices[1] - vertices[0];
                        let e2 = vertices[2] - vertices[0];
                        let normal = e1.cross(e2).normalize();

                        lights.push(Light::Triangle {
                            color,
                            vertices: vertices.try_into().unwrap(),
                            normal,
                        })
                    }

                    meshes.push((mesh, start_idx as u32));
                }
                _ => bail!("unknown light type"),
            };
        }

        Ok(lights)
    }

    fn parse_toml_file(dir: impl AsRef<Path>, conf: &Value) -> Result<File> {
        let dir_path = dir.as_ref();

        let Value::String(file_path) = conf else {
            bail!("file path should be a string");
        };

        Ok(File::open(dir_path.join(file_path))?)
    }

    fn parse_toml_vec3(conf: &Value) -> Result<Vec3> {
        let Value::Array(values) = conf else {
            bail!("array was not provided for vec3");
        };

        let mut values = values.iter();

        let x = Self::parse_toml_f32(values.next().ok_or(anyhow!("vec3 requires x y z - x not provided"))?)?;
        let y = Self::parse_toml_f32(values.next().ok_or(anyhow!("vec3 requires x y z - y not provided"))?)?;
        let z = Self::parse_toml_f32(values.next().ok_or(anyhow!("vec3 requires x y z - xznot provided"))?)?;

        if values.next().is_some() {
            bail!("vec3 requires 3 arguments x y z, but saw extra");
        }

        Ok(Vec3::new(x, y, z))
    }

    fn parse_toml_f32(val: &Value) -> Result<f32> {
        Ok(match val {
            Value::Integer(x) => *x as f32,
            Value::Float(x) => *x as f32,
            _ => bail!("float requires toml int or float"),
        })
    }

    fn parse_toml_transform(value: &Value) -> Result<Mat4> {
        let Value::String(transform_str) = value else {
            bail!("transform must be a string");
        };

        Self::parse_transform(transform_str)
    }

    fn parse_transform(transform_str: &str) -> Result<Mat4> {
        let mut transform = Mat4::IDENTITY;

        for line in transform_str.lines() {
            let mut tokens = line.trim().split_ascii_whitespace();

            let Some(action) = tokens.next() else {
                // empty means we ignore
                continue;
            };

            // match on action (omg thats a cinema term)
            match action {
                "identity" => transform = Mat4::IDENTITY,
                "translate" => {
                    let x = Self::parse_f32(&mut tokens)?;
                    let y = Self::parse_f32(&mut tokens)?;
                    let z = Self::parse_f32(&mut tokens)?;

                    if tokens.next().is_some() {
                        bail!("transform requires only x y z, but extra info was provided");
                    }

                    let translation = Mat4::from_translation(Vec3::new(x, y, z));
                    transform = translation * transform;
                }
                "rotate" => {
                    let angle = Self::parse_f32(&mut tokens)? * PI / 180f32;
                    let x = Self::parse_f32(&mut tokens)?;
                    let y = Self::parse_f32(&mut tokens)?;
                    let z = Self::parse_f32(&mut tokens)?;
                    let axis = Vec3::new(x, y, z);

                    if tokens.next().is_some() {
                        bail!("rotate requires only angle x y z, but extra info was provided");
                    }

                    let rotation = Mat4::from_axis_angle(axis, angle);
                    transform = rotation * transform;
                }
                "scale" => {
                    let x = Self::parse_f32(&mut tokens)?;
                    let y = Self::parse_f32(&mut tokens)?;
                    let z = Self::parse_f32(&mut tokens)?;
                    let scale = Vec3::new(x, y, z);

                    if tokens.next().is_some() {
                        bail!("scale requires only x y z, but extra info was provided");
                    }

                    let scale = Mat4::from_scale(scale);
                    transform = scale * transform;
                }
                "lookat" => {
                    let eye_x = Self::parse_f32(&mut tokens)?;
                    let eye_y = Self::parse_f32(&mut tokens)?;
                    let eye_z = Self::parse_f32(&mut tokens)?;
                    let eye = Vec3::new(eye_x, eye_y, eye_z);

                    let center_x = Self::parse_f32(&mut tokens)?;
                    let center_y = Self::parse_f32(&mut tokens)?;
                    let center_z = Self::parse_f32(&mut tokens)?;
                    let center = Vec3::new(center_x, center_y, center_z);

                    let up_x = Self::parse_f32(&mut tokens)?;
                    let up_y = Self::parse_f32(&mut tokens)?;
                    let up_z = Self::parse_f32(&mut tokens)?;
                    let up = Vec3::new(up_x, up_y, up_z);

                    if tokens.next().is_some() {
                        bail!("lookat requires only eye_x eye_y eye_z center_x center_y center_z up_x up_y up_z, but extra info was provided");
                    }

                    let lookat = Mat4::look_at_lh(eye, center, up);
                    transform = lookat;
                }
                _ if action.starts_with("#") => (),
                x => bail!("invalid transform action: {x}"),
            };
        }

        Ok(transform)
    }

    fn parse_f32<'a>(mut tokens: impl Iterator<Item = &'a str>) -> Result<f32> {
        let num = tokens
            .next()
            .ok_or(anyhow!("float expected but not found"))?;
        Ok(num.parse()?)
    }

    fn parse_type_str(type_str: &str) -> Result<ShaderType> {
        let mut tokens = TokenIter::new(type_str).peekable();
        Self::parse_type(&mut tokens)
    }

    fn parse_type<'a>(tokens: &mut Peekable<TokenIter<'a>>) -> Result<ShaderType> {
        let lookahead = tokens.peek().ok_or(anyhow!("incomplete type - no tokens remaining"))?;
        let parsed_type = match lookahead {
            Token::LSqBracket => Self::parse_array(tokens)?,
            Token::Semicolon => todo!(),
            Token::Typename(_) => Self::parse_simple_type(tokens)?,
            Token::Integer(int) => bail!("type should never start with integer token, but started with one: {int}"),
            Token::RSqBracket => bail!("type should never start with right square bracket"),
            Token::LexerError(_) => {
                let Token::LexerError(error) = tokens.next().unwrap() else {
                    panic!("failed to match lexer error that was just matched on");
                };

                return Err(error);
            },
        };

        Ok(parsed_type)
    }

    fn parse_array<'a>(tokens: &mut Peekable<TokenIter<'a>>) -> Result<ShaderType> {
        if !matches!(tokens.next().ok_or(anyhow!("no next token"))?, Token::LSqBracket) {
            bail!("no [ found for start of array");
        }

        let parsed_type = Self::parse_type(tokens)?;

        if !matches!(tokens.next().ok_or(anyhow!("no next token"))?, Token::Semicolon) {
            bail!("no semicolon found after parsing array type")
        }

        let Token::Integer(array_size) = tokens.next().ok_or(anyhow!("no next token"))? else {
            bail!("array size should be a constant unsigned integer")
        };

        if !matches!(tokens.next().ok_or(anyhow!("no next token"))?, Token::RSqBracket) {
            bail!("no ] found for end of array")
        }

        Ok(ShaderType::Array(Box::new(parsed_type), array_size))
    }

    fn parse_simple_type<'a>(tokens: &mut Peekable<TokenIter<'a>>) -> Result<ShaderType> {
        let the_token = tokens.next().ok_or(anyhow!("no next token"))?;
        let Token::Typename(typename) = the_token else {
            bail!("token was not a typename: {:?}", the_token)
        };
        Ok(match typename {
            "float" => ShaderType::Float,
            "int" => ShaderType::Int,
            "uint" => ShaderType::UInt,
            "vec3" => ShaderType::Vec3,
            s => bail!("invalid typename: {s}")
        })
    }

    fn parse_toml_camera(conf: &Table) -> Result<Camera> {
        let Some(Value::Table(camera_table)) = conf.get("camera") else {
            bail!("camera must be a table")
        };

        let fov = camera_table
            .get("fov")
            .ok_or(anyhow!("camera.fov must be set"))?;
        let fov = match fov {
            Value::Integer(x) => *x as f32,
            Value::Float(x) => *x as f32,
            _ => bail!("camera.fov must be an integer or float"),
        };

        // convert to radians
        let fov_radians = fov * PI / 180f32;

        // use 1 as default aspect ratio
        // ideally this will never actually be used since it will be updated immediately after window creation
        let perspective = Mat4::perspective_lh(fov_radians, 1f32, 0f32, 1000f32);

        let Value::String(view_str) = camera_table
            .get("view")
            .ok_or(anyhow!("camera.view must be set"))?
        else {
            bail!("camera.view must be a transform string")
        };
        let view = Self::parse_transform(view_str)?;

        Ok(Camera { view, perspective })
    }
}

#[cfg(test)]
mod tests {
    use glam::{Vec3, Vec4};

    use crate::scene::scenes::mesh::{Shader, ShaderType};

    use super::MeshScene;

    #[test]
    fn parse_lookat() {
        let camera_mat = MeshScene::parse_transform(
            "
        # this is a comment that should be ignored
        lookat 3 2 1   0 0 0   0 0 1
        ",
        )
        .expect("failed to parse");

        let point = Vec4::new(0f32, 0f32, 0f32, 1f32);
        let point_cam = camera_mat * point;

        let eye = Vec3::new(3f32, 2f32, 1f32);
        let dist = eye.dot(eye).sqrt();

        assert!((point_cam.z - dist) < 2e-4, "point_cam.z: {}, ||eye - origin||: {}", point_cam.z, dist);
    }

    #[test]
    fn parse_type_complicated() {
        let parsed_type = MeshScene::parse_type_str("[[vec3; 3]; 1]").expect("failed to parse");

        assert_eq!(parsed_type, ShaderType::Array(
            Box::new(
                ShaderType::Array(
                    Box::new(ShaderType::Vec3),
                    3
                )),
            1
        ))
    }
}