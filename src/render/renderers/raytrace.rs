use std::{ffi::c_char, sync::LazyLock, u64};

use anyhow::anyhow;
use ash::{khr, vk, Device, Entry, Instance};
use gpu_allocator::{vulkan::*, MemoryLocation};
use tobj::Model;

use crate::{
    features::{vk_features, VkFeatureGuard, VkFeatures},
    render::Renderer,
    scene::{
        scenes::mesh::{Light, MeshScene, Object},
        Scene,
    },
    utils::{align_up, AllocatedBuffer, QueueFamilyInfo},
    window::WindowData,
};

pub struct RaytraceRenderer {
    device: Device,
    accel_struct_device: khr::acceleration_structure::Device,
    rt_pipeline_device: khr::ray_tracing_pipeline::Device,
    device_properties: vk::PhysicalDeviceProperties,
    rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    command_pool: vk::CommandPool,
    compute_queue: vk::Queue,
    mesh_buffers: Vec<(AllocatedBuffer, AllocatedBuffer)>,
    instance_buffer: Option<AllocatedBuffer>,
    top_as: vk::AccelerationStructureKHR,
    top_as_buffer: Option<AllocatedBuffer>,
    bottom_ass: Vec<vk::AccelerationStructureKHR>,
    bottom_as_buffers: Vec<AllocatedBuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sbt_buffer: Option<AllocatedBuffer>,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
    callable_region: vk::StridedDeviceAddressRegionKHR,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    storage_image: vk::Image,
    storage_image_view: vk::ImageView,
    storage_image_allocation: Allocation,
    camera_buffer: Option<AllocatedBuffer>,
    vertex_normal_buffer: Option<AllocatedBuffer>,
    light_buffer: Option<AllocatedBuffer>,
    offset_buffer: Option<AllocatedBuffer>,
    brdf_param_buffer: Option<AllocatedBuffer>,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl RaytraceRenderer {
    fn build_accel_structs(
        &self,
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
                self.accel_struct_device
                    .get_acceleration_structure_build_sizes(
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
                self.accel_struct_device
                    .create_acceleration_structure(&create_info, None)
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

            self.accel_struct_device.cmd_build_acceleration_structures(
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
        meshes: &[Model],
        allocator: &mut Allocator,
    ) -> anyhow::Result<(
        Vec<vk::AccelerationStructureGeometryKHR<'static>>,
        Vec<(AllocatedBuffer, AllocatedBuffer)>,
        Vec<u32>,
    )> {
        let mut geometries = Vec::new();
        let mut buffers = Vec::new();
        let mut primitive_counts = Vec::new();
        for mesh in meshes {
            let vertex_count = mesh.mesh.positions.len() / 3;
            let vertex_stride = std::mem::size_of_val(&mesh.mesh.positions[0]) * 3;

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
            vertex_buffer.store(&mesh.mesh.positions)?;

            let index_count = mesh.mesh.indices.len();
            let index_stride = std::mem::size_of_val(&mesh.mesh.indices[0]);

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
            index_buffer.store(&mesh.mesh.indices)?;

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
        objects: &[Object],
        bottom_accel_structs: &[vk::AccelerationStructureKHR],
        allocator: &mut Allocator,
    ) -> anyhow::Result<(
        vk::AccelerationStructureGeometryKHR<'static>,
        AllocatedBuffer,
        u32,
    )> {
        let mut accel_handles = Vec::new();
        for bottom_accel_struct in bottom_accel_structs {
            accel_handles.push({
                let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR {
                    acceleration_structure: *bottom_accel_struct,
                    ..Default::default()
                };
                unsafe {
                    self.accel_struct_device
                        .get_acceleration_structure_device_address(&as_addr_info)
                }
            });
        }

        let mut instances = Vec::new();
        for object in objects {
            let mut matrix = [0f32; 16];
            object
                .transform
                .transpose()
                .write_cols_to_slice(&mut matrix);

            let mut matrix_3_4 = [0f32; 12];
            matrix_3_4.copy_from_slice(&matrix[0..12]);

            instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: matrix_3_4 },
                instance_custom_index_and_mask: vk::Packed24_8::new(object.vertex_index, 0xff),
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

    fn get_descriptor_set_layout(
        &self,
    ) -> anyhow::Result<(vk::DescriptorSetLayout, Vec<vk::DescriptorPoolSize>)> {
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
            // camera
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                stage_flags: vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                binding: 2,
                ..Default::default()
            },
            // vertices and normals
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                binding: 3,
                ..Default::default()
            },
            // lights
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                binding: 4,
                ..Default::default()
            },
            // offsets
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                binding: 5,
                ..Default::default()
            },
            // brdf params
            vk::DescriptorSetLayoutBinding {
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                binding: 6,
                ..Default::default()
            },
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo {
            p_bindings: bindings.as_ptr(),
            binding_count: bindings.len() as u32,
            ..Default::default()
        };

        let layout = unsafe {
            self.device
                .create_descriptor_set_layout(&create_info, None)?
        };

        let mut descriptor_sizes = Vec::new();
        for binding in bindings {
            descriptor_sizes.push(vk::DescriptorPoolSize {
                ty: binding.descriptor_type,
                descriptor_count: binding.descriptor_count,
            });
        }

        Ok((layout, descriptor_sizes))
    }

    fn create_pipeline(
        &self,
        scene: &MeshScene,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> anyhow::Result<(vk::PipelineLayout, vk::Pipeline, usize)> {
        let layout_create_info = vk::PipelineLayoutCreateInfo {
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            set_layout_count: descriptor_set_layouts.len() as u32,
            ..Default::default()
        };
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_create_info, None)?
        };

        let raygen_module = scene.raygen_shader.clone().compile(&self.device)?.module();
        let miss_module = scene.miss_shader.clone().compile(&self.device)?.module();
        let mut shader_stages = vec![
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::RAYGEN_KHR,
                module: raygen_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::MISS_KHR,
                module: miss_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
        ];
        let mut shader_groups = vec![
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader: 0,
                closest_hit_shader: vk::SHADER_UNUSED_KHR,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
            vk::RayTracingShaderGroupCreateInfoKHR {
                ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader: 1,
                closest_hit_shader: vk::SHADER_UNUSED_KHR,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            },
        ];

        for hit_shader in scene.hit_shaders.iter() {
            let module = hit_shader.clone().compile(&self.device)?.module();
            shader_stages.push(vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            });
            shader_groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
                ty: vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
                general_shader: vk::SHADER_UNUSED_KHR,
                closest_hit_shader: shader_stages.len() as u32 - 1,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            });
        }

        let pipeline = unsafe {
            let out = self.rt_pipeline_device.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[vk::RayTracingPipelineCreateInfoKHR {
                    stage_count: shader_stages.len() as u32,
                    p_stages: shader_stages.as_ptr(),
                    group_count: shader_groups.len() as u32,
                    p_groups: shader_groups.as_ptr(),
                    max_pipeline_ray_recursion_depth: 1,
                    layout: pipeline_layout,
                    ..Default::default()
                }],
                None,
            );
            match out {
                Ok(x) => x[0],
                Err((x, y)) => *x
                    .get(0)
                    .ok_or(anyhow!("failed to construct pipeline: {y}"))?,
            }
        };

        Ok((pipeline_layout, pipeline, shader_groups.len()))
    }

    unsafe fn copy_buffer(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: u64,
    ) -> anyhow::Result<()> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: 1,
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffer = self
            .device
            .allocate_command_buffers(&command_buffer_allocate_info)?[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        self.device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

        self.device.cmd_copy_buffer(
            command_buffer,
            src,
            dst,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        );

        self.device.end_command_buffer(command_buffer)?;

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &raw const command_buffer,
            ..Default::default()
        };

        self.device
            .queue_submit(self.compute_queue, &[submit_info], vk::Fence::null())?;

        self.device.queue_wait_idle(self.compute_queue)?;
        self.device
            .free_command_buffers(self.command_pool, &[command_buffer]);

        Ok(())
    }

    unsafe fn create_device_buffer<T: Copy>(
        &self,
        data: &[T],
        usage: vk::BufferUsageFlags,
        allocator: &mut Allocator,
    ) -> anyhow::Result<AllocatedBuffer> {
        let size = std::mem::size_of_val(data) as u64;
        let mut staging_buffer = AllocatedBuffer::new(
            &self.device,
            allocator,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            self.device_properties.limits,
        )?;
        staging_buffer.store(data)?;

        let buffer = AllocatedBuffer::new(
            &self.device,
            allocator,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            self.device_properties.limits,
        )?;

        self.copy_buffer(staging_buffer.buffer, buffer.buffer, size)?;

        staging_buffer.destroy(&self.device, allocator)?;

        Ok(buffer)
    }

    fn create_sbt(
        &self,
        shader_group_count: usize,
        allocator: &mut Allocator,
    ) -> anyhow::Result<(
        AllocatedBuffer,
        vk::StridedDeviceAddressRegionKHR,
        vk::StridedDeviceAddressRegionKHR,
        vk::StridedDeviceAddressRegionKHR,
        vk::StridedDeviceAddressRegionKHR,
    )> {
        let unaligned_table_data = unsafe {
            self.rt_pipeline_device
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    shader_group_count as u32,
                    shader_group_count
                        * self.rt_pipeline_properties.shader_group_handle_size as usize,
                )?
        };

        let handle_size = self.rt_pipeline_properties.shader_group_handle_size as usize;
        let handle_stride = align_up(
            handle_size as u32,
            self.rt_pipeline_properties.shader_group_handle_alignment,
        ) as usize;
        let base_stride = align_up(
            handle_size as u32,
            self.rt_pipeline_properties.shader_group_base_alignment,
        ) as usize;

        let table_size = 2 * base_stride + (shader_group_count - 2) * handle_stride;
        let mut table_data = vec![0u8; table_size];

        // raygen
        table_data[0..handle_size].copy_from_slice(&unaligned_table_data[0..handle_size]);
        // miss
        table_data[base_stride..base_stride + handle_size]
            .copy_from_slice(&unaligned_table_data[handle_size..2 * handle_size]);
        // closest hit
        for i in 0..shader_group_count - 2 {
            let aligned_base = 2 * base_stride + i * handle_stride;
            table_data[aligned_base..aligned_base + handle_size].copy_from_slice(
                &unaligned_table_data[(i + 2) * handle_size..(i + 3) * handle_size],
            );
        }

        let sbt_buffer = unsafe {
            self.create_device_buffer(
                &table_data,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                allocator,
            )?
        };

        let sbt_address = unsafe { sbt_buffer.get_device_address(&self.device) };
        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address,
            stride: handle_stride as u64,
            size: handle_stride as u64,
        };
        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + base_stride as u64,
            stride: handle_stride as u64,
            size: handle_stride as u64,
        };
        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + 2 * base_stride as u64,
            stride: handle_stride as u64,
            size: (shader_group_count as u64 - 2) * handle_stride as u64,
        };
        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        Ok((
            sbt_buffer,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
        ))
    }

    fn create_descriptor_pool_and_set(
        &self,
        layout: vk::DescriptorSetLayout,
        sizes: &[vk::DescriptorPoolSize],
    ) -> anyhow::Result<(vk::DescriptorPool, vk::DescriptorSet)> {
        let pool = {
            let pool_info = vk::DescriptorPoolCreateInfo {
                pool_size_count: sizes.len() as u32,
                p_pool_sizes: sizes.as_ptr(),
                max_sets: 1,
                ..Default::default()
            };

            unsafe { self.device.create_descriptor_pool(&pool_info, None) }?
        };

        let set = unsafe {
            let allocate_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                p_set_layouts: &raw const layout,
                descriptor_set_count: 1,
                ..Default::default()
            };
            self.device.allocate_descriptor_sets(&allocate_info)?[0]
        };

        Ok((pool, set))
    }

    fn create_storage_image(
        &self,
        window_data: &WindowData,
        allocator: &mut Allocator,
    ) -> anyhow::Result<(vk::Image, vk::ImageView, Allocation)> {
        let (width, height) = window_data.get_size();
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let image = unsafe { self.device.create_image(&image_create_info, None)? };

        let memory_req = unsafe { self.device.get_image_memory_requirements(image) };
        let image_allocation = allocator.allocate(&AllocationCreateDesc {
            name: "storage image",
            requirements: memory_req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            self.device.bind_image_memory(
                image,
                image_allocation.memory(),
                image_allocation.offset(),
            )?;
        }

        let image_view = {
            let image_view_create_info = vk::ImageViewCreateInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                format: image_create_info.format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image,
                ..Default::default()
            };

            unsafe {
                self.device
                    .create_image_view(&image_view_create_info, None)?
            }
        };

        let command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_buffer_count: 1,
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            };

            unsafe { self.device.allocate_command_buffers(&allocate_info)?[0] }
        };

        let image_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );

            self.device.end_command_buffer(command_buffer)?;
        }

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &raw const command_buffer,
            ..Default::default()
        };

        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], vk::Fence::null())?;

            self.device.queue_wait_idle(self.compute_queue)?;
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }

        Ok((image, image_view, image_allocation))
    }

    fn create_command_buffer(
        &self,
        target_images: &[vk::Image],
        (width, height): (u32, u32),
    ) -> anyhow::Result<Vec<vk::CommandBuffer>> {
        let mut command_buffers = Vec::new();
        for target_image in target_images {
            let command_buffer = {
                let allocate_info = vk::CommandBufferAllocateInfo {
                    command_buffer_count: 1,
                    command_pool: self.command_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    ..Default::default()
                };

                unsafe { self.device.allocate_command_buffers(&allocate_info)?[0] }
            };

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

            unsafe {
                self.device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );

                self.rt_pipeline_device.cmd_trace_rays(
                    command_buffer,
                    &self.raygen_region,
                    &self.miss_region,
                    &self.hit_region,
                    &self.callable_region,
                    width,
                    height,
                    1,
                );

                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
                        | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier {
                        src_access_mask: vk::AccessFlags::SHADER_WRITE,
                        dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                        ..Default::default()
                    }],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::NONE,
                        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        image: *target_image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        ..Default::default()
                    }],
                );

                self.device.cmd_blit_image(
                    command_buffer,
                    self.storage_image,
                    vk::ImageLayout::GENERAL,
                    *target_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: width as i32,
                                y: height as i32,
                                z: 1,
                            },
                        ],
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        dst_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: width as i32,
                                y: height as i32,
                                z: 1,
                            },
                        ],
                    }],
                    vk::Filter::LINEAR,
                );

                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::NONE,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                        image: *target_image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        ..Default::default()
                    }],
                );

                self.device.end_command_buffer(command_buffer)?;
            }

            command_buffers.push(command_buffer)
        }

        Ok(command_buffers)
    }
}

impl Renderer<MeshScene, WindowData> for RaytraceRenderer {
    fn new(
        _vk_lib: &Entry,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_family_info: &QueueFamilyInfo,
        target: &WindowData,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let accel_struct_device = khr::acceleration_structure::Device::new(instance, device);
        let rt_pipeline_device = khr::ray_tracing_pipeline::Device::new(instance, device);

        let mut rt_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut physical_device_properties2 = vk::PhysicalDeviceProperties2 {
            p_next: &raw mut rt_pipeline_properties as *mut std::ffi::c_void,
            ..Default::default()
        };
        unsafe {
            instance
                .get_physical_device_properties2(physical_device, &mut physical_device_properties2)
        };

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

        let mut out = RaytraceRenderer {
            device: device.clone(),
            accel_struct_device,
            rt_pipeline_device,
            device_properties: physical_device_properties2.properties,
            rt_pipeline_properties,
            command_pool,
            compute_queue,
            mesh_buffers: Default::default(),
            instance_buffer: Default::default(),
            top_as: Default::default(),
            top_as_buffer: Default::default(),
            bottom_ass: Default::default(),
            bottom_as_buffers: Default::default(),
            pipeline_layout: Default::default(),
            pipeline: Default::default(),
            sbt_buffer: Default::default(),
            raygen_region: Default::default(),
            miss_region: Default::default(),
            hit_region: Default::default(),
            callable_region: Default::default(),
            descriptor_pool: Default::default(),
            descriptor_set: Default::default(),
            storage_image: Default::default(),
            storage_image_view: Default::default(),
            storage_image_allocation: Default::default(),
            camera_buffer: Default::default(),
            vertex_normal_buffer: Default::default(),
            light_buffer: Default::default(),
            offset_buffer: Default::default(),
            brdf_param_buffer: Default::default(),
            command_buffers: Default::default(),
        };

        (
            out.storage_image,
            out.storage_image_view,
            out.storage_image_allocation,
        ) = out.create_storage_image(target, allocator)?;

        Ok(out)
    }

    fn ingest_scene(&mut self, scene: &MeshScene, allocator: &mut Allocator) -> anyhow::Result<()> {
        let (mesh_geometries, mesh_buffers, mesh_primitive_counts) =
            self.get_mesh_geometries(&scene.meshes, allocator)?;
        self.mesh_buffers = mesh_buffers;

        (self.bottom_ass, self.bottom_as_buffers) = self.build_accel_structs(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &mesh_geometries,
            &mesh_primitive_counts,
            allocator,
        )?;

        let (instance_geometry, instance_buffer, instance_count) =
            self.get_instance_geometry(&scene.objects, &self.bottom_ass, allocator)?;
        self.instance_buffer = Some(instance_buffer);

        (self.top_as, self.top_as_buffer) = {
            let (top_as, mut top_as_buffer) = self.build_accel_structs(
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                &[instance_geometry],
                &[instance_count],
                allocator,
            )?;
            (top_as[0], Some(top_as_buffer.remove(0)))
        };

        let (descriptor_set_layout, descriptor_sizes) = self.get_descriptor_set_layout()?;

        let shader_group_count: usize;
        (self.pipeline_layout, self.pipeline, shader_group_count) =
            self.create_pipeline(scene, &[descriptor_set_layout])?;

        let sbt_buffer: AllocatedBuffer;
        (
            sbt_buffer,
            self.raygen_region,
            self.miss_region,
            self.hit_region,
            self.callable_region,
        ) = self.create_sbt(shader_group_count, allocator)?;
        self.sbt_buffer = Some(sbt_buffer);

        (self.descriptor_pool, self.descriptor_set) =
            self.create_descriptor_pool_and_set(descriptor_set_layout, &descriptor_sizes)?;

        let vertex_normal_data: Vec<f32> = scene
            .meshes
            .iter()
            .flat_map(|x| {
                x.mesh
                    .positions
                    .chunks_exact(3)
                    .zip(x.mesh.normals.chunks_exact(3))
                    .flat_map(|(p, n)| p.iter().chain(n.iter()))
            })
            .map(|&x| x)
            .collect();

        self.vertex_normal_buffer = Some(unsafe {
            self.create_device_buffer(
                &vertex_normal_data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                allocator,
            )?
        });

        let mut light_data = Vec::new();
        for light in scene.lights.iter() {
            if let Light::Point { color, position } = light {
                light_data.extend(color.to_array());
                light_data.extend(position.to_array());
            } else if let Light::Triangle { color, vertices } = light {
                light_data.extend(color.to_array());
                for vertex in vertices {
                    light_data.extend(vertex.to_array());
                }
            }
        }

        self.light_buffer = Some(unsafe {
            self.create_device_buffer(&light_data, vk::BufferUsageFlags::STORAGE_BUFFER, allocator)?
        });

        let offset_data: Vec<_> = scene.objects.iter().map(|x| x.brdf_params_index).collect();

        self.offset_buffer = Some(unsafe {
            self.create_device_buffer(
                &offset_data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                allocator,
            )?
        });

        let brdf_param_data: Vec<_> = scene
            .objects
            .iter()
            .flat_map(|x| &x.brdf_params)
            .map(|&x| x)
            .collect();
        self.brdf_param_buffer = Some(unsafe {
            self.create_device_buffer(
                &brdf_param_data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                allocator,
            )?
        });

        let mut camera_data = Vec::new();
        let view_proj = scene.camera.perspective * scene.camera.view;
        let view_inverse = scene.camera.view.inverse();
        let proj_inverse = scene.camera.perspective.inverse();
        camera_data.extend(view_proj.to_cols_array());
        camera_data.extend(view_inverse.to_cols_array());
        camera_data.extend(proj_inverse.to_cols_array());

        self.camera_buffer = Some(unsafe {
            self.create_device_buffer(
                &camera_data,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                allocator,
            )?
        });

        let mut writes = Vec::new();

        let image_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: self.storage_image_view,
            sampler: vk::Sampler::null(),
        };
        writes.push(vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            p_image_info: &raw const image_info,
            ..Default::default()
        });

        let accel_info = vk::WriteDescriptorSetAccelerationStructureKHR {
            acceleration_structure_count: 1,
            p_acceleration_structures: &raw const self.top_as,
            ..Default::default()
        };
        writes.push(vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
            p_next: &raw const accel_info as *const std::ffi::c_void,
            ..Default::default()
        });

        let camera_info = vk::DescriptorBufferInfo {
            buffer: self.camera_buffer.as_ref().unwrap().buffer,
            range: vk::WHOLE_SIZE,
            offset: 0,
        };
        writes.push(vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 2,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &raw const camera_info,
            ..Default::default()
        });

        let mut buffer_infos = Vec::new();

        for (i, buf) in [
            &self.vertex_normal_buffer,
            &self.light_buffer,
            &self.offset_buffer,
            &self.brdf_param_buffer,
        ]
        .iter()
        .enumerate()
        {
            buffer_infos.push(vk::DescriptorBufferInfo {
                buffer: buf.as_ref().unwrap().buffer,
                range: vk::WHOLE_SIZE,
                offset: 0,
            });
            writes.push(vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: i as u32 + 3,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                p_buffer_info: unsafe { buffer_infos.as_ptr().add(i) },
                ..Default::default()
            });
        }

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }

        Ok(())
    }

    fn render_to(
        &mut self,
        _updates: &[<MeshScene as Scene>::Update],
        target: &mut WindowData,
    ) -> anyhow::Result<()> {
        if self.command_buffers.is_empty() {
            self.command_buffers =
                self.create_command_buffer(target.get_images(), target.get_size())?;
        }

        let image_index = target.acquire_next_image()?;
        let (image_semaphore, render_semaphore) = target.get_current_semaphores();
        let wait_stage = vk::PipelineStageFlags::TRANSFER;
        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &raw const self.command_buffers[image_index as usize],
            signal_semaphore_count: 1,
            p_signal_semaphores: &raw const render_semaphore,
            wait_semaphore_count: 1,
            p_wait_semaphores: &raw const image_semaphore,
            p_wait_dst_stage_mask: &raw const wait_stage,
            ..Default::default()
        };

        let flight_fence = target.get_current_flight_fence();

        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], flight_fence)?;
        }

        target.present(self.compute_queue)?;

        Ok(())
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
                    timeline_semaphore,
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
