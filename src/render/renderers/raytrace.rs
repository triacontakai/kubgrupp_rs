use std::{ffi::c_char, sync::LazyLock};

use anyhow::anyhow;
use ash::{khr, vk, Device, Entry, Instance};
use gpu_allocator::{vulkan::*, MemoryLocation};
use obj::Obj;

use crate::{
    features::{vk_features, VkFeatureGuard, VkFeatures},
    render::Renderer,
    scene::{
        scenes::mesh::{self, MeshScene},
        Scene,
    },
    utils::{AllocatedBuffer, QueueFamilyInfo},
    window::WindowData,
};

pub struct RaytraceRenderer {
    vk_lib: Entry,
    instance: Instance,
    device: Device,
    physical_device: vk::PhysicalDevice,
    device_properties: vk::PhysicalDeviceProperties,
    command_pool: vk::CommandPool,
    compute_queue: vk::Queue,
    acceleration_structure: vk::AccelerationStructureKHR,
    pipeline: vk::Pipeline,
}

impl RaytraceRenderer {
    fn build_accel_structs(
        &self,
        acceleration_structure: &khr::acceleration_structure::Device,
        ty: vk::AccelerationStructureTypeKHR,
        geometries: &[vk::AccelerationStructureGeometryKHR],
        primitive_counts: &[u32],
        allocator: &mut Allocator,
    ) -> anyhow::Result<(Vec<vk::AccelerationStructureKHR>, Vec<AllocatedBuffer>)> {
        let mut build_infos = Vec::new();
        let mut build_range_infos = Vec::new();
        let mut scratch_buffers = Vec::new();

        let mut accel_structs = Vec::new();
        let mut buffers = Vec::new();

        for (geometry, primitive_count) in geometries.iter().zip(primitive_counts) {
            let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR {
                first_vertex: 0,
                primitive_count: *primitive_count,
                primitive_offset: 0,
                transform_offset: 0,
            };

            let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
                flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                p_geometries: geometry as *const _,
                geometry_count: 1,
                mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                ty,
                ..Default::default()
            };

            let mut size_info: vk::AccelerationStructureBuildSizesInfoKHR = Default::default();
            unsafe {
                acceleration_structure.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[*primitive_count],
                    &mut size_info,
                );
            }

            let buffer = AllocatedBuffer::new(
                &self.device,
                allocator,
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                self.device_properties.limits,
            )?;

            let create_info = vk::AccelerationStructureCreateInfoKHR {
                ty: build_info.ty,
                size: size_info.acceleration_structure_size,
                buffer: buffer.buffer,
                offset: 0,
                ..Default::default()
            };
            let accel_struct = unsafe {
                acceleration_structure.create_acceleration_structure(&create_info, None)
            }?;
            build_info.dst_acceleration_structure = accel_struct;

            let scratch_buffer = AllocatedBuffer::new(
                &self.device,
                allocator,
                size_info.build_scratch_size,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                self.device_properties.limits,
            )?;

            build_info.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: unsafe { scratch_buffer.get_device_address(&self.device) },
            };

            scratch_buffers.push(scratch_buffer);
            build_infos.push(build_info);
            build_range_infos.push(build_range_info);

            accel_structs.push(accel_struct);
            buffers.push(buffer);
        }

        let unsqueezed_build_range_infos: Vec<_> = build_range_infos
            .iter()
            .map(|s| std::slice::from_ref(s))
            .collect();

        let build_command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_buffer_count: 1,
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            };

            let command_buffers = unsafe { self.device.allocate_command_buffers(&allocate_info) }?;
            command_buffers[0]
        };

        unsafe {
            self.device.begin_command_buffer(
                build_command_buffer,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;

            acceleration_structure.cmd_build_acceleration_structures(
                build_command_buffer,
                &build_infos,
                &unsqueezed_build_range_infos,
            );
            self.device.end_command_buffer(build_command_buffer)?;
            self.device.queue_submit(
                self.compute_queue,
                &[vk::SubmitInfo {
                    p_command_buffers: &raw const build_command_buffer,
                    command_buffer_count: 1,
                    ..Default::default()
                }],
                vk::Fence::null(),
            )?;

            self.device.queue_wait_idle(self.compute_queue)?;
            self.device
                .free_command_buffers(self.command_pool, &[build_command_buffer]);

            for scratch_buffer in scratch_buffers {
                scratch_buffer.destroy(&self.device, allocator)?;
            }
        }

        Ok((accel_structs, buffers))
    }

    fn get_mesh_geometries(
        &self,
        meshes: &[Obj],
        allocator: &mut Allocator,
    ) -> anyhow::Result<(
        Vec<vk::AccelerationStructureGeometryKHR>,
        Vec<(AllocatedBuffer, AllocatedBuffer)>,
        Vec<u32>,
    )> {
        let mut geometries = Vec::new();
        let mut buffers = Vec::new();
        let mut primitive_counts = Vec::new();
        for mesh in meshes {
            let vertex_count = mesh.vertices.len();
            let vertex_stride = std::mem::size_of_val(&mesh.vertices[0]);

            let mut vertex_buffer = AllocatedBuffer::new(
                &self.device,
                allocator,
                (vertex_stride * vertex_count) as vk::DeviceSize,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
                self.device_properties.limits,
            )?;
            vertex_buffer.store(&mesh.vertices)?;

            let index_count = mesh.indices.len();
            let index_stride = std::mem::size_of_val(&mesh.indices[0]);

            let mut index_buffer = AllocatedBuffer::new(
                &self.device,
                allocator,
                (index_stride * index_count) as vk::DeviceSize,
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
                self.device_properties.limits,
            )?;
            index_buffer.store(&mesh.indices)?;

            let geometry = vk::AccelerationStructureGeometryKHR {
                geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                geometry: vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                        vertex_data: vk::DeviceOrHostAddressConstKHR {
                            device_address: unsafe {
                                vertex_buffer.get_device_address(&self.device)
                            },
                        },
                        max_vertex: vertex_count as u32 - 1,
                        vertex_stride: vertex_stride as u64,
                        vertex_format: vk::Format::R32G32B32_SFLOAT,
                        index_data: vk::DeviceOrHostAddressConstKHR {
                            device_address: unsafe {
                                index_buffer.get_device_address(&self.device)
                            },
                        },
                        index_type: vk::IndexType::UINT32,
                        ..Default::default()
                    },
                },
                ..Default::default()
            };

            geometries.push(geometry);
            buffers.push((vertex_buffer, index_buffer));
            primitive_counts.push(index_count as u32 / 3);
        }
        Ok((geometries, buffers, primitive_counts))
    }

    fn get_instance_geometry(
        &self,
        acceleration_structure: &khr::acceleration_structure::Device,
        objects: &[mesh::Instance],
        bottom_accel_structs: &[vk::AccelerationStructureKHR],
        allocator: &mut Allocator,
    ) -> anyhow::Result<(vk::AccelerationStructureGeometryKHR, AllocatedBuffer, u32)> {
        let mut accel_handles = Vec::new();
        for bottom_accel_struct in bottom_accel_structs {
            accel_handles.push({
                let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR {
                    acceleration_structure: *bottom_accel_struct,
                    ..Default::default()
                };
                unsafe {
                    acceleration_structure.get_acceleration_structure_device_address(&as_addr_info)
                }
            });
        }

        let mut instances = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            let mut matrix = [0f32; 16];
            object
                .transform
                .transpose()
                .write_cols_to_slice(&mut matrix);

            let mut matrix_3_4 = [0f32; 12];
            matrix_3_4.copy_from_slice(&matrix[0..12]);

            instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: matrix_3_4 },
                instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    object.brdf_i as u32,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: accel_handles[object.mesh_i],
                },
            });
        }

        let instance_buffer_size = std::mem::size_of_val(&instances[0]) * instances.len();
        let mut instance_buffer = AllocatedBuffer::new(
            &self.device,
            allocator,
            instance_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::CpuToGpu,
            self.device_properties.limits,
        )?;
        instance_buffer.store(&instances)?;

        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                    array_of_pointers: false as u32,
                    data: vk::DeviceOrHostAddressConstKHR {
                        device_address: unsafe { instance_buffer.get_device_address(&self.device) },
                    },
                    ..Default::default()
                },
            },
            ..Default::default()
        };

        Ok((geometry, instance_buffer, instances.len() as u32))
    }
}

impl Renderer<MeshScene, WindowData> for RaytraceRenderer {
    fn new(
        vk_lib: &Entry,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_family_info: &QueueFamilyInfo,
    ) -> anyhow::Result<Self> {
        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let compute_queue_index = queue_family_info
            .compute_index
            .ok_or(anyhow!("no compute"))?;
        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo {
                queue_family_index: compute_queue_index,
                ..Default::default()
            };
            unsafe { device.create_command_pool(&create_info, None) }?
        };
        let compute_queue = unsafe { device.get_device_queue(compute_queue_index, 0) };
        Ok(RaytraceRenderer {
            vk_lib: vk_lib.clone(),
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            device_properties,
            command_pool,
            compute_queue,
            acceleration_structure: Default::default(),
            pipeline: Default::default(),
        })
    }

    fn ingest_scene(&mut self, scene: &MeshScene, allocator: &mut Allocator) -> anyhow::Result<()> {
        let acceleration_structure =
            khr::acceleration_structure::Device::new(&self.instance, &self.device);

        let (mesh_geometries, mesh_buffers, mesh_primitive_counts) =
            self.get_mesh_geometries(&scene.meshes, allocator)?;

        let (bottom_accel_structs, bottom_as_buffers) = self.build_accel_structs(
            &acceleration_structure,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &mesh_geometries,
            &mesh_primitive_counts,
            allocator,
        )?;

        let (instance_geometry, instance_buffer, instance_count) = self.get_instance_geometry(
            &acceleration_structure,
            &scene.instances,
            &bottom_accel_structs,
            allocator,
        )?;

        let (top_as, top_as_buffer) = {
            let (top_as, mut top_as_buffer) = self.build_accel_structs(
                &acceleration_structure,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                &[instance_geometry],
                &[instance_count],
                allocator,
            )?;
            (top_as[0], top_as_buffer.remove(0))
        };

        let descriptor_set_layout = {
            let binding_flags_inner = [
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            ];

            let binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                binding_count: binding_flags_inner.len() as u32,
                p_binding_flags: binding_flags_inner.as_ptr(),
                ..Default::default()
            };

            unsafe {
                let bindings = [
                    vk::DescriptorSetLayoutBinding {
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                        binding: 0,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        binding: 1,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        descriptor_count: MeshScene::MAX_LIGHTS,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        stage_flags: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        binding: 2,
                        ..Default::default()
                    },
                ];

                let create_info = vk::DescriptorSetLayoutCreateInfo {
                    p_bindings: bindings.as_ptr(),
                    binding_count: bindings.len() as u32,
                    p_next: &raw const binding_flags as *const std::ffi::c_void,
                    ..Default::default()
                };
                self.device
                    .create_descriptor_set_layout(&create_info, None)?
            }
        };

        Ok(())
    }

    fn render_to(
        &mut self,
        updates: &<MeshScene as Scene>::Updates,
        target: &mut WindowData,
    ) -> anyhow::Result<()> {
        todo!()
    }

    fn required_instance_extensions() -> &'static [*const c_char] {
        &[]
    }

    fn required_device_extensions() -> &'static [*const c_char] {
        const EXTENSIONS: &[*const c_char] = &[
            khr::acceleration_structure::NAME.as_ptr(),
            khr::deferred_host_operations::NAME.as_ptr(),
            khr::ray_tracing_pipeline::NAME.as_ptr(),
        ];
        EXTENSIONS
    }

    fn required_features() -> VkFeatureGuard<'static> {
        static FEATURES: LazyLock<VkFeatures> = LazyLock::new(|| {
            vk_features! {
                vk::PhysicalDeviceFeatures {},
                vk::PhysicalDeviceVulkan12Features {
                    buffer_device_address,
                    descriptor_indexing,
                    descriptor_binding_partially_bound,
                    descriptor_binding_variable_descriptor_count,
                    runtime_descriptor_array,
                },
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                    acceleration_structure,
                },
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
                    ray_tracing_pipeline,
                },
            }
        });

        // this does allocation and could theoretically be optimized by putting in a const
        // but uh who cares lol
        FEATURES.get_list()
    }

    fn has_required_queue_families(queue_family_info: &QueueFamilyInfo) -> bool {
        queue_family_info.compute_index.is_some() && queue_family_info.present_index.is_some()
    }

    fn get_queue_info(queue_family_info: &QueueFamilyInfo) -> Vec<vk::DeviceQueueCreateInfo> {
        let create_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family_info.compute_index.unwrap(),
            queue_count: 1,
            p_queue_priorities: &1.0,
            ..Default::default()
        };

        vec![create_info]
    }
}
