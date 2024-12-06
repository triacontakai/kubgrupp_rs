use std::{ffi::c_char, sync::LazyLock};

use anyhow::anyhow;
use ash::{khr, vk, Device, Entry, Instance};

use crate::{
    features::{vk_features, VkFeatureGuard, VkFeatures},
    render::Renderer,
    scene::scenes::mesh::MeshScene,
    scene::Scene,
    utils::{AllocatedBuffer, QueueFamilyInfo, QueueInfo},
    window::WindowData,
};

const MAX_LIGHTS: u32 = 1000;

pub struct RaytraceRenderer {
    vk_lib: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    command_pool: vk::CommandPool,
    compute_queue: vk::Queue,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    acceleration_structure: vk::AccelerationStructureKHR,
    pipeline: vk::Pipeline,
}

impl RaytraceRenderer {
    fn build_accel_struct(
        &self,
        acceleration_structure: &khr::acceleration_structure::Device,
        geometry: &[vk::AccelerationStructureGeometryKHR],
        primitive_count: u32,
        ty: vk::AccelerationStructureTypeKHR,
    ) -> anyhow::Result<(vk::AccelerationStructureKHR, AllocatedBuffer)> {
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR {
            first_vertex: 0,
            primitive_count,
            primitive_offset: 0,
            transform_offset: 0,
        };

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            p_geometries: geometry.as_ptr(),
            geometry_count: geometry.len() as u32,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            ty,
            ..Default::default()
        };

        let mut size_info: vk::AccelerationStructureBuildSizesInfoKHR = Default::default();
        unsafe {
            acceleration_structure.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[primitive_count],
                &mut size_info,
            );
        }

        let buffer = AllocatedBuffer::new(
            &self.device,
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            self.device_memory_properties,
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR {
            ty: build_info.ty,
            size: size_info.acceleration_structure_size,
            buffer: buffer.buffer,
            offset: 0,
            ..Default::default()
        };
        let accel_struct =
            unsafe { acceleration_structure.create_acceleration_structure(&create_info, None) }?;
        build_info.dst_acceleration_structure = accel_struct;

        let scratch_buffer = AllocatedBuffer::new(
            &self.device,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            self.device_memory_properties,
        )?;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: unsafe { scratch_buffer.get_device_address(&self.device) },
        };

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
                &[build_info],
                &[&[build_range_info]],
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
            scratch_buffer.destroy(&self.device);
        }

        Ok((accel_struct, buffer))
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
        unsafe {
            let device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);
            Ok(RaytraceRenderer {
                vk_lib: vk_lib.clone(),
                instance: instance.clone(),
                physical_device: physical_device.clone(),
                device: device.clone(),
                command_pool,
                compute_queue,
                device_memory_properties,
                acceleration_structure: Default::default(),
                pipeline: Default::default(),
            })
        }
    }

    fn ingest_scene(&mut self, scene: &MeshScene) -> anyhow::Result<()> {
        let vertices: [[f32; 3]; 3] = [[-0.5, -0.5, 0.0], [0.0, 0.5, 0.0], [0.5, -0.5, 0.0]];
        let vertex_count = vertices.len();
        let vertex_stride = std::mem::size_of_val(&vertices[0]);

        let mut vertex_buffer = AllocatedBuffer::new(
            &self.device,
            (vertex_stride * vertex_count) as vk::DeviceSize,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            self.device_memory_properties,
        )?;
        vertex_buffer.store(&self.device, &vertices)?;

        let indices: [u32; 3] = [0, 1, 2];
        let index_count = indices.len();

        let mut index_buffer = AllocatedBuffer::new(
            &self.device,
            (std::mem::size_of_val(&indices[0]) * index_count) as vk::DeviceSize,
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            self.device_memory_properties,
        )?;
        index_buffer.store(&self.device, &indices)?;

        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::TRIANGLES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                    vertex_data: vk::DeviceOrHostAddressConstKHR {
                        device_address: unsafe { vertex_buffer.get_device_address(&self.device) },
                    },
                    max_vertex: vertex_count as u32 - 1,
                    vertex_stride: vertex_stride as u64,
                    vertex_format: vk::Format::R32G32B32_SFLOAT,
                    index_data: vk::DeviceOrHostAddressConstKHR {
                        device_address: unsafe { index_buffer.get_device_address(&self.device) },
                    },
                    index_type: vk::IndexType::UINT32,
                    ..Default::default()
                },
            },
            ..Default::default()
        };

        let acceleration_structure =
            khr::acceleration_structure::Device::new(&self.instance, &self.device);

        let (bottom_as, bottom_as_buffer) = self.build_accel_struct(
            &acceleration_structure,
            &[geometry],
            index_count as u32 / 3,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )?;

        let accel_handle = {
            let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR {
                acceleration_structure: bottom_as,
                ..Default::default()
            };
            unsafe {
                acceleration_structure.get_acceleration_structure_device_address(&as_addr_info)
            }
        };

        let (instance_count, instance_buffer) = {
            let transform: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            let instances = vec![vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: transform },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: accel_handle,
                },
            }];

            let instance_buffer_size = std::mem::size_of_val(&instances[0]) * instances.len();

            let mut instance_buffer = AllocatedBuffer::new(
                &self.device,
                instance_buffer_size as vk::DeviceSize,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                self.device_memory_properties,
            )?;

            instance_buffer.store(&self.device, &instances)?;

            (instances.len(), instance_buffer)
        };

        let instances = vk::AccelerationStructureGeometryInstancesDataKHR {
            array_of_pointers: false as u32,
            data: vk::DeviceOrHostAddressConstKHR {
                device_address: unsafe { instance_buffer.get_device_address(&self.device) },
            },
            ..Default::default()
        };

        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR { instances },
            ..Default::default()
        };

        let (top_as, top_as_buffer) = self.build_accel_struct(
            &acceleration_structure,
            &[geometry],
            instance_count as u32,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        )?;

        let descriptor_set_layout = {
            let binding_flags_inner = [
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            ];

            let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
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
                        descriptor_count: MAX_LIGHTS,
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

    fn get_queue_info(queue_family_info: &QueueFamilyInfo) -> QueueInfo {
        let create_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family_info.compute_index.unwrap(),
            queue_count: 1,
            p_queue_priorities: &1.0,
            ..Default::default()
        };

        QueueInfo {
            infos: vec![create_info],
        }
    }
}
