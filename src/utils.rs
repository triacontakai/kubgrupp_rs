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
        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                size,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer = device.create_buffer(&buffer_info, None)?;

            let memory_req = device.get_buffer_memory_requirements(buffer);
            println!("{}", memory_req.alignment);

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

    pub unsafe fn destroy(self, device: &Device, allocator: &mut Allocator) -> Result<()> {
        device.destroy_buffer(self.buffer, None);
        allocator.free(self.allocation)?;
        Ok(())
    }
}
