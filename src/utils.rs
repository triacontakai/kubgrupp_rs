use anyhow::Result;
use ash::{khr, vk, Device, Entry, Instance};
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

#[derive(Default, Clone)]
pub struct QueueFamilyInfo {
    pub graphics_index: Option<u32>,
    pub present_index: Option<u32>,
    pub compute_index: Option<u32>,
    pub transfer_index: Option<u32>,
}

pub fn query_queue_families(
    vk_lib: &Entry,
    instance: &Instance,
    device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<QueueFamilyInfo> {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let mut info = QueueFamilyInfo::default();

    let surface_loader = khr::surface::Instance::new(vk_lib, instance);

    // this currently just chooses the first available queue family for each thing
    // possibly suboptimal idk, but oh well
    for (i, family) in queue_families.iter().enumerate() {
        if info.graphics_index.is_none() && family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            info.graphics_index = Some(i as u32);
        }
        if info.compute_index.is_none() && family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            info.compute_index = Some(i as u32);
        }
        if info.transfer_index.is_none() && family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
            info.transfer_index = Some(i as u32);
        }

        let present_support = unsafe {
            surface_loader.get_physical_device_surface_support(device, i as u32, surface)
        }?;
        if info.present_index.is_none() && present_support {
            info.present_index = Some(i as u32);
        }
    }

    Ok(info)
}

pub fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    allocation: Allocation,
    offset_alignment: usize,
}

impl AllocatedBuffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        limits: vk::PhysicalDeviceLimits,
    ) -> Result<AllocatedBuffer> {
        Self::new_with_alignment(device, allocator, size, usage, location, limits, 0)
    }

    pub fn new_with_alignment(
        device: &Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        limits: vk::PhysicalDeviceLimits,
        alignment: u32,
    ) -> Result<AllocatedBuffer> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                size,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer = device.create_buffer(&buffer_info, None)?;

            let mut memory_req = device.get_buffer_memory_requirements(buffer);
            if alignment > 0 {
                memory_req.alignment = align_up(memory_req.alignment as u32, alignment) as u64;
            }

            let allocation = allocator.allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements: memory_req,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            let mut offset_alignment: usize = 1;
            if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                offset_alignment = limits.min_storage_buffer_offset_alignment as usize;
            } else if usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
                offset_alignment = limits.min_uniform_buffer_offset_alignment as usize;
            }

            Ok(AllocatedBuffer {
                buffer,
                allocation,
                offset_alignment,
            })
        }
    }

    pub fn store<T: Copy>(&mut self, data: &[T]) -> Result<()> {
        presser::copy_from_slice_to_offset_with_align(
            data,
            &mut self.allocation,
            0,
            self.offset_alignment,
        )?;
        Ok(())
    }

    pub unsafe fn get_device_address(&self, device: &Device) -> u64 {
        let buffer_device_address_info = vk::BufferDeviceAddressInfo {
            buffer: self.buffer,
            ..Default::default()
        };
        device.get_buffer_device_address(&buffer_device_address_info)
    }

    pub unsafe fn destroy(self, device: &Device, allocator: &mut Allocator) {
        device.destroy_buffer(self.buffer, None);
        allocator.free(self.allocation).unwrap();
    }
}

pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    allocation: Allocation,
    layout: vk::ImageLayout,
}

impl AllocatedImage {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        size: (u32, u32),
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        location: MemoryLocation,
    ) -> Result<AllocatedImage> {
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: size.0,
                height: size.1,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let image = unsafe { device.create_image(&image_create_info, None)? };

        let memory_req = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "image",
            requirements: memory_req,
            location,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        let image_view = {
            let image_view_create_info = vk::ImageViewCreateInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                format,
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

            unsafe { device.create_image_view(&image_view_create_info, None)? }
        };

        Ok(AllocatedImage {
            image,
            image_view,
            width: size.0,
            height: size.1,
            format,
            usage,
            allocation,
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn transition(
        &mut self,
        device: &Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        layout: vk::ImageLayout,
    ) -> Result<()> {
        let command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_buffer_count: 1,
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            };

            unsafe { device.allocate_command_buffers(&allocate_info)?[0] }
        };

        let image_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
            old_layout: self.layout,
            new_layout: layout,
            image: self.image,
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
            device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );

            device.end_command_buffer(command_buffer)?;
        }

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &raw const command_buffer,
            ..Default::default()
        };

        unsafe {
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;

            device.queue_wait_idle(queue)?;
            device.free_command_buffers(command_pool, &[command_buffer]);
        }

        self.layout = layout;

        Ok(())
    }

    pub unsafe fn destroy(self, device: &Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}
