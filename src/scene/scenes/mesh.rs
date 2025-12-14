use std::{
    alloc::{self, Layout},
    collections::HashMap,
    f32::consts::PI,
    ffi::{CStr, CString},
    fs::File,
    io::Read,
    iter::{self, Peekable},
    path::Path,
    ptr::NonNull,
};

use anyhow::{anyhow, bail, Result};
use ash::{vk, Device};
use bytemuck::BoxBytes;
use glam::{Mat4, Vec2, Vec3, Vec4};
use log::warn;
use tobj::Model;
use toml::{map::Map, Table, Value};

use crate::{
    camera::Camera,
    scene::{
        type_lexer::{Token, TokenIter},
        Scene,
    },
};

const MESHES_DIR: &str = "resources/meshes";
const SPIRV_DIR: &str = "resources/shaders/spv/";
const SPIRV_EXTENSION: &str = ".spv";
const SPIRV_MAGIC: u32 = 0x07230203;

#[derive(Debug)]
pub struct MeshScene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub objects: Vec<Object>,
    pub meshes: Vec<Model>,

    pub raygen_shader: Shader,
    pub miss_shader: Shader,
    pub hit_shaders: Vec<Shader>,

    pub procedural_geometries: Vec<ProceduralGeometry>,
    pub procedural_objects: Vec<ProceduralObject>,

    pub brdf_buf: Vec<u8>,
    pub offset_buf: Vec<u32>,
}

#[derive(Debug, Clone)]
pub enum Light {
    Point {
        color: Vec3,
        position: Vec3,
    },
    Triangle {
        color: Vec3,
        vertices: [Vec3; 3],
    },
    Directional {
        color: Vec3,
        position: Vec3,
        direction: Vec3,
        radius: f32,
    },
}

#[derive(Debug, Clone)]
pub enum Shader {
    Uncompiled(CString, Box<[u32]>),
    Compiled(CString, vk::ShaderModule),
}

#[derive(Debug, Clone)]
pub struct Object {
    pub transform: Mat4,

    // its just like aris fr
    pub mesh_i: usize,
    pub brdf_i: usize,
    pub brdf_params: Vec<u8>,

    // this is pretty much just the base of the mesh in the list of all vertices
    pub vertex_index: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Debug)]
pub struct ProceduralGeometry {
    pub aabbs: Vec<Aabb>,
    pub intersection_shader: Shader,
    pub closest_hit_shader: Shader,
}

#[derive(Debug, Clone)]
pub struct ProceduralObject {
    pub transform: Mat4,
    pub geometry_index: usize,
    pub custom_index: u32,
}

#[derive(Debug, PartialEq)]
enum ShaderType {
    Float,
    Vec3,
    Vec2,
    UInt,
    Int,
    Array(Box<ShaderType>, u64),
}

#[derive(Debug)]
struct Shaders {
    raygen: Shader,
    miss: Shader,
    rchit: Vec<Shader>,
}

#[derive(Debug)]
pub enum MeshSceneUpdate {
    NewView(Mat4),
    NewSize((u32, u32, Mat4)),
}

impl Scene for MeshScene {
    type Update = MeshSceneUpdate;
}

impl Shader {
    pub fn module(&self) -> vk::ShaderModule {
        let Shader::Compiled(_, module) = self else {
            panic!("shader is not compiled")
        };

        *module
    }

    fn name(&self) -> &CStr {
        match self {
            Shader::Uncompiled(name, _) => name,
            Shader::Compiled(name, _) => name,
        }
    }

    pub fn compile(&self, device: &Device) -> Result<Self> {
        match self {
            Shader::Uncompiled(name, code) => {
                let create_info = vk::ShaderModuleCreateInfo {
                    code_size: code.len() * size_of::<u32>(),
                    p_code: code.as_ptr(),
                    ..Default::default()
                };

                let module = unsafe { device.create_shader_module(&create_info, None) }?;
                Ok(Shader::Compiled(name.clone(), module))
            }
            x @ Shader::Compiled(..) => Ok(x.clone()),
        }
    }
}

impl MeshScene {
    pub fn load_from(mut reader: impl Read) -> Result<Self> {
        let mut toml_conf = String::new();
        reader.read_to_string(&mut toml_conf)?;

        let conf: Table = toml_conf.parse()?;

        let camera = Self::parse_toml_camera(&conf)?;

        // load the global shaders
        let (shaders, shader_type_map) = Self::parse_toml_shaders(&conf)?;
        let (meshes, mesh_map) = Self::parse_toml_meshes(&conf)?;

        // load objects before lights
        // this is to give them the correct brdf_params_index
        let mut objects =
            Self::parse_toml_objects(&conf, &mesh_map, &meshes, &shaders.rchit, &shader_type_map)?;
        let lights = Self::parse_toml_lights(&conf, &mesh_map, &meshes, &mut objects)?;

        let (procedural_geometries, procedural_objects) =
            Self::parse_procedural_geometries(&conf, &lights)?;

        let (brdf_buf, offset_buf) =
            Self::get_brdf_params_buffer_and_indices(&objects, &shaders.rchit);

        Ok(Self {
            camera,
            lights,
            objects,
            meshes,
            raygen_shader: shaders.raygen,
            miss_shader: shaders.miss,
            hit_shaders: shaders.rchit,
            procedural_geometries,
            procedural_objects,
            brdf_buf,
            offset_buf,
        })
    }

    fn get_field<'a>(conf: &'a Table, field: &str) -> Result<&'a Value> {
        conf.get(field)
            .ok_or(anyhow!("field {} not provided", field))
    }

    fn get_array<'a>(conf: &'a Table, field: &str) -> Result<&'a Vec<Value>> {
        match Self::get_field(conf, field)? {
            Value::Array(vals) => Ok(vals),
            _ => Err(anyhow!("field {} must be an array", field)),
        }
    }

    fn get_string<'a>(conf: &'a Table, field: &str) -> Result<&'a String> {
        match Self::get_field(conf, field)? {
            Value::String(str) => Ok(str),
            _ => Err(anyhow!("field {} must be an array", field)),
        }
    }

    fn get_table<'a>(conf: &'a Table, field: &str) -> Result<&'a Map<String, Value>> {
        match Self::get_field(conf, field)? {
            Value::Table(table) => Ok(table),
            _ => Err(anyhow!("field {} must be an array", field)),
        }
    }

    fn get_brdf_params_buffer_and_indices(
        objects: &[Object],
        hit_shaders: &[Shader],
    ) -> (Vec<u8>, Vec<u32>) {
        // create vecs for the array of each brdf
        let mut arrays: Vec<Vec<u8>> = vec![Vec::new(); hit_shaders.len()];
        let mut param_sizes = vec![0usize; arrays.len()];
        for object in objects {
            let brdf_i = object.brdf_i;
            arrays[brdf_i].extend_from_slice(&object.brdf_params);
            param_sizes[brdf_i] = object.brdf_params.len();
        }

        // concatenate arrays
        // we have to account for the size of each individual parameter struct when doing this
        let mut data = Vec::new();
        let mut indices = Vec::new();
        for (array, param_size) in arrays.iter().zip(param_sizes) {
            if param_size == 0 {
                indices.push(0);
            } else {
                let current_size = data.len();
                let padding = if current_size % param_size != 0 {
                    param_size - (current_size % param_size)
                } else {
                    0
                };

                data.extend(iter::repeat_n(0, padding));
                assert!(data.len() % param_size == 0);

                let start_index = data.len() / param_size;
                indices.push(start_index);

                data.extend_from_slice(array);
            }
        }

        // create offset buffer
        let mut offsets = Vec::new();
        for object in objects {
            let index = &mut indices[object.brdf_i];
            offsets.push(*index as u32);
            *index += 1;
        }

        (data, offsets)
    }

    fn parse_toml_objects(
        conf: &Table,
        mesh_map: &HashMap<String, u32>,
        meshes: &[Model],
        shaders: &[Shader],
        type_map: &HashMap<String, Vec<ShaderType>>,
    ) -> Result<Vec<Object>> {
        // get primitive start offsets of meshes
        let mut offset = 0;
        let start_offsets: Vec<_> = meshes
            .iter()
            .map(|m| {
                let num_vertices = m.mesh.indices.len();
                let this_offset = offset;
                offset += num_vertices;

                this_offset
            })
            .collect();

        let mut objects = Vec::new();

        let object_confs = Self::get_array(conf, "object")?;
        for object in object_confs {
            let Value::Table(object) = object else {
                bail!("object should be a table");
            };

            let mesh_name = Self::get_string(object, "mesh")?;
            let transform = Self::parse_toml_transform(Self::get_field(object, "transform")?)?;

            let brdf_info = Self::get_table(object, "brdf")?;
            let brdf_name = Self::get_string(brdf_info, "name")?;
            let brdf_fields = Self::get_array(brdf_info, "fields")?;
            let field_types = type_map
                .get(brdf_name)
                .ok_or(anyhow!("undefined brdf name: {}", brdf_name))?;

            if field_types.len() != brdf_fields.len() {
                bail!(
                    "expected number of fields ({}) doesn't match up with provided fields ({})",
                    field_types.len(),
                    brdf_fields.len()
                );
            }

            let mut datas = Vec::new();
            for (field, type_info) in brdf_fields.iter().zip(field_types) {
                // similar to array comment in parse_toml_field - technically there can be padding between fields
                // but like there will not be :)
                let data = Self::parse_toml_field(field, type_info)?;
                datas.extend_from_slice(&data);
            }

            let brdf_name = CString::new(brdf_name.clone())?;
            let brdf_i = shaders
                .iter()
                .position(|x| x.name() == &brdf_name[..])
                .ok_or(anyhow!("undefined brdf: {:?}", brdf_name))?;
            let mesh_i = *mesh_map.get(mesh_name).ok_or(anyhow!("asd"))? as usize;
            let vertex_index = start_offsets[mesh_i] as u32;

            objects.push(Object {
                transform,
                mesh_i,
                brdf_i,
                brdf_params: datas,
                vertex_index,
            })
        }

        Ok(objects)
    }

    fn parse_toml_field(field: &Value, type_info: &ShaderType) -> Result<Vec<u8>> {
        match type_info {
            ShaderType::Float => {
                let float = Self::parse_toml_f32(field)?;
                Ok(float.to_le_bytes().to_vec())
            }
            ShaderType::Vec3 => {
                let Value::Array(array) = field else {
                    bail!("vec3 type requires array of length 3");
                };

                if array.len() != 3 {
                    bail!("vec3 type requires array of length 3");
                }

                let x = Self::parse_toml_f32(&array[0])?;
                let y = Self::parse_toml_f32(&array[1])?;
                let z = Self::parse_toml_f32(&array[2])?;

                let bytes: [u8; 12] = bytemuck::cast(Vec3::new(x, y, z));
                Ok(bytes.to_vec())
            }
            ShaderType::Vec2 => {
                let Value::Array(array) = field else {
                    bail!("vec2 type requires array of length 2");
                };

                if array.len() != 2 {
                    bail!("vec2 type requires array of length 2");
                }

                let x = Self::parse_toml_f32(&array[0])?;
                let y = Self::parse_toml_f32(&array[1])?;

                let bytes: [u8; 8] = bytemuck::cast(Vec2::new(x, y));
                Ok(bytes.to_vec())
            }
            ShaderType::UInt => {
                let &Value::Integer(num) = field else {
                    bail!("uint type requires integer")
                };

                let num: u32 = num.try_into()?;
                Ok(num.to_le_bytes().to_vec())
            }
            ShaderType::Int => {
                let &Value::Integer(num) = field else {
                    bail!("int type requires integer")
                };

                let num: i32 = num.try_into()?;
                Ok(num.to_le_bytes().to_vec())
            }
            ShaderType::Array(shader_type, _) => {
                let Value::Array(array) = field else {
                    bail!("array type requires toml array");
                };

                let mut full_data = Vec::new();
                let datas = array.iter().map(|f| Self::parse_toml_field(f, shader_type));
                for data in datas {
                    // technically there should be padding for alignment
                    // but with the types we are using w/ layout scalar everything has same alignment (4)
                    // so who cares
                    full_data.extend_from_slice(&data?[..]);
                }

                Ok(full_data)
            }
        }
    }

    fn parse_toml_shaders(conf: &Table) -> Result<(Shaders, HashMap<String, Vec<ShaderType>>)> {
        let Value::Table(_global_shaders) = Self::get_field(conf, "global_shaders")? else {
            bail!("global_shaders must be a table");
        };
        let global_shaders = Self::get_table(conf, "global_shaders")?;

        let raygen = Self::parse_toml_shader(Self::get_field(global_shaders, "raygen")?, "raygen")?;
        let miss = Self::parse_toml_shader(Self::get_field(global_shaders, "miss")?, "miss")?;

        let mut chit_shaders = Vec::new();
        if global_shaders.get("emitter_hit").is_some() {
            let emitter_hit = Self::parse_toml_shader(
                Self::get_field(global_shaders, "emitter_hit")?,
                "emitter_hit",
            )?;
            chit_shaders.push(emitter_hit);
        }

        // parse shaders in brdfs
        // these also include types
        let Value::Array(brdfs) = Self::get_field(conf, "brdf")? else {
            bail!("brdf must be a list")
        };

        let mut type_map = HashMap::new();

        for brdf in brdfs {
            let Value::Table(brdf) = brdf else {
                bail!("brdf entry must be tables");
            };

            let name = Self::get_string(brdf, "name")?;
            let chit_shader = Self::parse_toml_shader(Self::get_field(brdf, "chit_shader")?, name)?;

            let fields = Self::get_array(brdf, "field")?;
            let mut shader_types = Vec::new();
            for field in fields {
                let Value::Table(field) = field else {
                    bail!("field must be a table");
                };

                let shader_type = Self::parse_type_str(Self::get_string(field, "type")?)?;
                shader_types.push(shader_type);
            }

            type_map.insert(name.clone(), shader_types);
            chit_shaders.push(chit_shader);
        }

        Ok((
            Shaders {
                raygen,
                miss,
                rchit: chit_shaders,
            },
            type_map,
        ))
    }

    fn parse_toml_shader(name: &Value, shader_name: &str) -> Result<Shader> {
        let Value::String(name) = name else {
            bail!("shader path must be a string");
        };

        let mut spv_name = name.clone();
        spv_name.push_str(SPIRV_EXTENSION);

        let spv_path = Path::new(SPIRV_DIR).join(spv_name);
        let mut spv_file = File::open(spv_path)?;
        let file_info = spv_file.metadata()?;

        let shader_size = file_info.len();
        if shader_size == 0 || shader_size % 4 != 0 {
            bail!("invalid shader size: {shader_size} - must be aligned to 4 bytes and greater than 0");
        }

        // allocate a buffer that is aligned to u32 since that is required for shader code
        let layout = Layout::array::<u8>(shader_size as usize)?;
        let layout = layout.align_to(align_of::<u32>()).unwrap();

        let code = unsafe { alloc::alloc(layout) };
        if code.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let mut code = unsafe { BoxBytes::from_raw_parts(NonNull::new_unchecked(code), layout) };
        spv_file.read_exact(&mut code)?;

        // now that the code has been read in, we can cast as u32
        // this should be guaranteed to succeed because of the alignment stuff above
        #[allow(unused_mut)]
        let mut code: Box<[u32]> = bytemuck::from_box_bytes(code);

        // on big endian systems, we need to swap endianness of every u32
        // this is because the shader is in little-endian
        #[cfg(target_endian = "big")]
        for word in &mut code {
            *word = (*word).swap_bytes();
        }

        // assert SPIRV magic number: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_magic_number
        if code[0] != SPIRV_MAGIC {
            bail!("invalid SPIR-V magic number");
        }

        Ok(Shader::Uncompiled(CString::new(shader_name)?, code))
    }

    fn parse_toml_meshes(conf: &Table) -> Result<(Vec<Model>, HashMap<String, u32>)> {
        let Value::Array(obj_confs) = Self::get_field(conf, "object")? else {
            bail!("objects field must be an array of objects");
        };

        let Value::Array(light_confs) = Self::get_field(conf, "light")? else {
            bail!("lights field must be an array of objects");
        };

        // get only the area light configs
        let area_lights = light_confs.iter().filter(|c| {
            let Value::Table(light_conf) = c else {
                return false;
            };
            let Ok(Value::String(light_type)) = Self::get_field(light_conf, "type") else {
                return false;
            };

            light_type == "area"
        });

        let mut meshes = Vec::new();
        let mut mesh_map = HashMap::new();

        for obj in obj_confs.iter().chain(area_lights) {
            let Value::Table(obj) = obj else {
                bail!("expected table, but found {}", obj);
            };

            let Value::String(mesh_name) = Self::get_field(obj, "mesh")? else {
                bail!("expected mesh name to be a string")
            };

            // don't add a mesh multiple times
            if mesh_map.contains_key(mesh_name) {
                continue;
            }

            let mesh_path = Path::new(MESHES_DIR).join(Self::get_string(obj, "mesh")?);
            let (mesh, _) = tobj::load_obj(mesh_path, &tobj::GPU_LOAD_OPTIONS)?;

            // only take the first model
            if mesh.len() > 1 {
                warn!(
                    "mesh file has {} meshes - only using first ({})",
                    mesh.len(),
                    mesh[0].name
                );
            }

            if let Some(mesh) = mesh.into_iter().next() {
                mesh_map.insert(mesh_name.clone(), meshes.len() as u32);
                meshes.push(mesh);
            }
        }

        Ok((meshes, mesh_map))
    }

    fn parse_toml_lights(
        conf: &Table,
        mesh_map: &HashMap<String, u32>,
        meshes: &[Model],
        objects: &mut Vec<Object>,
    ) -> Result<Vec<Light>> {
        let Value::Array(light_confs) = conf
            .get("light")
            .ok_or(anyhow!("no lights field provided"))?
        else {
            bail!("lights field must be an array of lights");
        };

        let mut lights = Vec::new();

        for light_conf in light_confs {
            let Value::Table(light_conf) = light_conf else {
                bail!("light must be a table");
            };
            let Value::String(light_type) = light_conf
                .get("type")
                .ok_or(anyhow!("no type field found for light"))?
            else {
                bail!("light type must be a string");
            };

            let color = Self::parse_toml_vec3(
                light_conf
                    .get("color")
                    .ok_or(anyhow!("no color field found for light"))?,
            )?;

            match light_type.as_str() {
                "point" => {
                    let position = Self::parse_toml_vec3(
                        light_conf
                            .get("position")
                            .ok_or(anyhow!("no top_left field found for light"))?,
                    )?;
                    lights.push(Light::Point { color, position });
                }
                "area" => {
                    let transform = Self::parse_toml_transform(
                        light_conf
                            .get("transform")
                            .ok_or(anyhow!("no transform field provided"))?,
                    )?;

                    let mesh_name = Self::get_string(light_conf, "mesh")?;
                    let mesh_i = *mesh_map
                        .get(mesh_name)
                        .ok_or(anyhow!("no such mesh: {}", mesh_name))?
                        as usize;
                    let mesh = &meshes[mesh_i].mesh;

                    let start_idx = lights.len();

                    // load triangles to get triangle lights
                    let triangles = mesh.indices.chunks_exact(3);
                    if !triangles.remainder().is_empty() {
                        bail!("obj face list was not a multiple of 3 in length")
                    }
                    for triangle in triangles {
                        let vertices: Vec<_> = triangle
                            .iter()
                            .map(|&i| {
                                let pos = Vec4::from((
                                    Vec3::from_slice(
                                        &mesh.positions[3 * i as usize..3 * i as usize + 3],
                                    ),
                                    1.0,
                                ));
                                let v = transform * pos;

                                Vec3::new(v.x, v.y, v.z)
                            })
                            .collect();

                        lights.push(Light::Triangle {
                            color,
                            vertices: vertices.try_into().unwrap(),
                        })
                    }

                    objects.push(Object {
                        transform,
                        mesh_i,
                        brdf_i: 0, // emitter hit brdf is always 0
                        brdf_params: Vec::new(),
                        vertex_index: start_idx as u32, // vertex index is actually light index
                    });
                }
                "directional" => {
                    let position = Self::parse_toml_vec3(
                        light_conf
                            .get("position")
                            .ok_or(anyhow!("no position field found for light"))?,
                    )?;
                    let direction = Self::parse_toml_vec3(
                        light_conf
                            .get("direction")
                            .ok_or(anyhow!("no direction field found for light"))?,
                    )?;
                    let radius = Self::parse_toml_f32(
                        light_conf
                            .get("radius")
                            .ok_or(anyhow!("no radius field found for light"))?,
                    )?;
                    lights.push(Light::Directional {
                        color,
                        position,
                        direction,
                        radius,
                    });
                }
                _ => bail!("unknown light type"),
            };
        }

        Ok(lights)
    }

    fn parse_toml_vec3(conf: &Value) -> Result<Vec3> {
        let Value::Array(values) = conf else {
            bail!("array was not provided for vec3");
        };

        let mut values = values.iter();

        let x = Self::parse_toml_f32(
            values
                .next()
                .ok_or(anyhow!("vec3 requires x y z - x not provided"))?,
        )?;
        let y = Self::parse_toml_f32(
            values
                .next()
                .ok_or(anyhow!("vec3 requires x y z - y not provided"))?,
        )?;
        let z = Self::parse_toml_f32(
            values
                .next()
                .ok_or(anyhow!("vec3 requires x y z - xznot provided"))?,
        )?;

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

            // ignore comments
            if action.starts_with('#') {
                continue;
            }

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

    fn parse_type(tokens: &mut Peekable<TokenIter<'_>>) -> Result<ShaderType> {
        let lookahead = tokens
            .peek()
            .ok_or(anyhow!("incomplete type - no tokens remaining"))?;
        let parsed_type = match lookahead {
            Token::LSqBracket => Self::parse_array(tokens)?,
            Token::Semicolon => todo!(),
            Token::Typename(_) => Self::parse_simple_type(tokens)?,
            Token::Integer(int) => {
                bail!("type should never start with integer token, but started with one: {int}")
            }
            Token::RSqBracket => bail!("type should never start with right square bracket"),
            Token::LexerError(_) => {
                let Token::LexerError(error) = tokens.next().unwrap() else {
                    panic!("failed to match lexer error that was just matched on");
                };

                return Err(error);
            }
        };

        Ok(parsed_type)
    }

    fn parse_array(tokens: &mut Peekable<TokenIter<'_>>) -> Result<ShaderType> {
        if !matches!(
            tokens.next().ok_or(anyhow!("no next token"))?,
            Token::LSqBracket
        ) {
            bail!("no [ found for start of array");
        }

        let parsed_type = Self::parse_type(tokens)?;

        if !matches!(
            tokens.next().ok_or(anyhow!("no next token"))?,
            Token::Semicolon
        ) {
            bail!("no semicolon found after parsing array type")
        }

        let Token::Integer(array_size) = tokens.next().ok_or(anyhow!("no next token"))? else {
            bail!("array size should be a constant unsigned integer")
        };

        if !matches!(
            tokens.next().ok_or(anyhow!("no next token"))?,
            Token::RSqBracket
        ) {
            bail!("no ] found for end of array")
        }

        Ok(ShaderType::Array(Box::new(parsed_type), array_size))
    }

    fn parse_simple_type(tokens: &mut Peekable<TokenIter<'_>>) -> Result<ShaderType> {
        let the_token = tokens.next().ok_or(anyhow!("no next token"))?;
        let Token::Typename(typename) = the_token else {
            bail!("token was not a typename: {:?}", the_token)
        };
        Ok(match typename {
            "float" => ShaderType::Float,
            "int" => ShaderType::Int,
            "uint" => ShaderType::UInt,
            "vec3" => ShaderType::Vec3,
            "vec2" => ShaderType::Vec2,
            s => bail!("invalid typename: {s}"),
        })
    }

    fn parse_procedural_geometries(
        conf: &Table,
        lights: &[Light],
    ) -> Result<(Vec<ProceduralGeometry>, Vec<ProceduralObject>)> {
        let mut geometries = Vec::new();
        let mut geometry_map = HashMap::new();
        let mut objects = Vec::new();

        if let Some(Value::Array(geom_confs)) = conf.get("procedural_geometry") {
            for geom_conf in geom_confs {
                let Value::Table(geom_conf) = geom_conf else {
                    bail!("procedural_geometry must be a table");
                };

                let name = Self::get_string(geom_conf, "name")?;
                let int_shader_name = Self::get_string(geom_conf, "intersection_shader")?;
                let hit_shader_name = Self::get_string(geom_conf, "closest_hit_shader")?;

                let int_shader = Self::parse_toml_shader(
                    &Value::String(int_shader_name.clone()),
                    &format!("{}_int", name),
                )?;
                let hit_shader = Self::parse_toml_shader(
                    &Value::String(hit_shader_name.clone()),
                    &format!("{}_hit", name),
                )?;

                let aabbs_conf = Self::get_array(geom_conf, "aabbs")?;
                let mut aabbs = Vec::new();
                for aabb_conf in aabbs_conf {
                    let Value::Array(coords) = aabb_conf else {
                        bail!("aabb must be an array of 6 floats [min_x, min_y, min_z, max_x, max_y, max_z]");
                    };
                    if coords.len() != 6 {
                        bail!("aabb must have exactly 6 values");
                    }
                    let min_x = Self::parse_toml_f32(&coords[0])?;
                    let min_y = Self::parse_toml_f32(&coords[1])?;
                    let min_z = Self::parse_toml_f32(&coords[2])?;
                    let max_x = Self::parse_toml_f32(&coords[3])?;
                    let max_y = Self::parse_toml_f32(&coords[4])?;
                    let max_z = Self::parse_toml_f32(&coords[5])?;
                    aabbs.push(Aabb {
                        min: Vec3::new(min_x, min_y, min_z),
                        max: Vec3::new(max_x, max_y, max_z),
                    });
                }

                geometry_map.insert(name.clone(), geometries.len());
                geometries.push(ProceduralGeometry {
                    aabbs,
                    intersection_shader: int_shader,
                    closest_hit_shader: hit_shader,
                });
            }
        }

        if let Some(Value::Array(obj_confs)) = conf.get("procedural_object") {
            for obj_conf in obj_confs {
                let Value::Table(obj_conf) = obj_conf else {
                    bail!("procedural_object must be a table");
                };

                let geom_name = Self::get_string(obj_conf, "geometry")?;
                let geometry_index = *geometry_map
                    .get(geom_name)
                    .ok_or_else(|| anyhow!("unknown procedural geometry: {}", geom_name))?;

                let transform =
                    Self::parse_toml_transform(Self::get_field(obj_conf, "transform")?)?;

                let custom_index = obj_conf
                    .get("custom_index")
                    .map(|v| match v {
                        Value::Integer(i) => Ok(*i as u32),
                        _ => bail!("custom_index must be an integer"),
                    })
                    .transpose()?
                    .unwrap_or(0);

                objects.push(ProceduralObject {
                    transform,
                    geometry_index,
                    custom_index,
                });
            }
        }

        let directional_lights: Vec<_> = lights
            .iter()
            .enumerate()
            .filter_map(|(i, l)| match l {
                Light::Directional {
                    position,
                    direction,
                    radius,
                    ..
                } => Some((i, *position, direction.normalize(), *radius)),
                _ => None,
            })
            .collect();

        if !directional_lights.is_empty() {
            let global_shaders = conf
                .get("global_shaders")
                .and_then(|v| v.as_table())
                .ok_or_else(|| anyhow!("global_shaders required when using directional lights"))?;

            let int_shader_name = global_shaders
                .get("directional_emitter_int")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    anyhow!(
                        "global_shaders.directional_emitter_int required for directional lights"
                    )
                })?;
            let hit_shader_name = global_shaders
                .get("directional_emitter_hit")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    anyhow!(
                        "global_shaders.directional_emitter_hit required for directional lights"
                    )
                })?;

            let int_shader = Self::parse_toml_shader(
                &Value::String(int_shader_name.to_string()),
                "directional_emitter_int",
            )?;
            let hit_shader = Self::parse_toml_shader(
                &Value::String(hit_shader_name.to_string()),
                "directional_emitter_hit",
            )?;

            let geometry_index = geometries.len();
            geometries.push(ProceduralGeometry {
                aabbs: vec![Aabb {
                    min: Vec3::new(-1.0, -1.0, -0.001),
                    max: Vec3::new(1.0, 1.0, 0.001),
                }],
                intersection_shader: int_shader,
                closest_hit_shader: hit_shader,
            });

            for (light_index, position, direction, radius) in directional_lights {
                let transform = Self::compute_light_geometry_transform(position, direction, radius);
                objects.push(ProceduralObject {
                    transform,
                    geometry_index,
                    custom_index: light_index as u32,
                });
            }
        }

        Ok((geometries, objects))
    }

    fn compute_light_geometry_transform(position: Vec3, direction: Vec3, radius: f32) -> Mat4 {
        let target_normal = direction.normalize();
        let object_normal = Vec3::Z;

        let rotation = if object_normal.dot(target_normal).abs() > 0.9999 {
            if object_normal.dot(target_normal) > 0.0 {
                glam::Quat::IDENTITY
            } else {
                glam::Quat::from_axis_angle(Vec3::X, PI)
            }
        } else {
            glam::Quat::from_rotation_arc(object_normal, target_normal)
        };

        let scale = Mat4::from_scale(Vec3::new(radius, radius, 1.0));
        let rotation_mat = Mat4::from_quat(rotation);
        let translation = Mat4::from_translation(position);

        translation * rotation_mat * scale
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

        let Value::String(view_str) = camera_table
            .get("view")
            .ok_or(anyhow!("camera.view must be set"))?
        else {
            bail!("camera.view must be a transform string")
        };
        let view = Self::parse_transform(view_str)?;

        Ok(Camera::new(view, fov))
    }
}
