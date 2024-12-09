use std::{ffi::c_char, ptr, u64};

use anyhow::{anyhow, Result};
use ash::{khr, vk, Device, Entry, Instance};
use winit::window::Window;

use crate::{defer::Defer, utils};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

pub struct WindowData {
    swapchain: vk::SwapchainKHR,
    surface: vk::SurfaceKHR,
    window: Window,

    swapchain_loader: khr::swapchain::Device,
    surface_loader: khr::surface::Instance,
    device: Device,

    image_format: vk::Format,
    image_extent: vk::Extent2D,
    images: Vec<vk::Image>,
    current_image: u32,

    image_semaphores: Vec<vk::Semaphore>,
    render_semaphores: Vec<vk::Semaphore>,
    flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl WindowData {
    pub fn new(
        vk_lib: &Entry,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        window: Window,
    ) -> Result<WindowData> {
        let swapchain_loader = khr::swapchain::Device::new(instance, device);
        let surface_loader = khr::surface::Instance::new(vk_lib, instance);
        let surface = surface.defer(|x| unsafe { surface_loader.destroy_surface(*x, None) });

        let (swapchain, image_format, image_extent, images) =
            Self::create_swapchain(vk_lib, instance, device, physical_device, *surface, &window)?;

        let (image_semaphores, render_semaphores, flight_fences) =
            Self::create_sync_objects(device)?;

        let surface = surface.undefer();
        Ok(WindowData {
            surface_loader,
            swapchain_loader,
            swapchain,
            surface,
            window,
            device: device.clone(),
            image_format,
            image_extent,
            images,
            current_image: 0,
            image_semaphores,
            render_semaphores,
            flight_fences,
            current_frame: 0,
        })
    }

    pub fn present(&mut self, queue: vk::Queue) -> Result<()> {
        let render_semaphore = self.render_semaphores[self.current_frame];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &raw const render_semaphore,
            swapchain_count: 1,
            p_swapchains: &raw const self.swapchain,
            p_image_indices: &raw const self.current_image,
            ..Default::default()
        };

        unsafe { self.swapchain_loader.queue_present(queue, &present_info)? };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT as usize;

        Ok(())
    }

    pub fn get_current_semaphores(&self) -> (vk::Semaphore, vk::Semaphore) {
        (
            self.image_semaphores[self.current_frame],
            self.render_semaphores[self.current_frame],
        )
    }

    pub fn get_current_flight_fence(&self) -> vk::Fence {
        self.flight_fences[self.current_frame]
    }

    pub fn acquire_next_image(&mut self) -> Result<(vk::Image, u32)> {
        let image_semaphore = self.image_semaphores[self.current_frame];
        let flight_fence = self.flight_fences[self.current_frame];

        unsafe {
            self.device
                .wait_for_fences(&[flight_fence], true, u64::MAX)?
        }

        (self.current_image, _) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_semaphore,
                vk::Fence::null(),
            )
        }?;

        unsafe { self.device.reset_fences(&[flight_fence])? };

        Ok((self.images[self.current_image as usize], self.current_image))
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn get_size(&self) -> (u32, u32) {
        (self.image_extent.width, self.image_extent.height)
    }

    fn create_sync_objects(
        device: &Device,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
        let mut image_semaphores = Vec::new();
        let mut render_semaphores = Vec::new();
        let mut flight_fences = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None)? }
            };

            let render_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None)? }
            };

            let flight_fence = {
                let fence_info = vk::FenceCreateInfo {
                    flags: vk::FenceCreateFlags::SIGNALED,
                    ..Default::default()
                };
                unsafe { device.create_fence(&fence_info, None)? }
            };

            image_semaphores.push(image_semaphore);
            render_semaphores.push(render_semaphore);
            flight_fences.push(flight_fence);
        }

        Ok((image_semaphores, render_semaphores, flight_fences))
    }

    fn create_swapchain(
        vk_lib: &Entry,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        window: &Window,
    ) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
        let swapchain_loader = khr::swapchain::Device::new(instance, device);

        let support_details =
            Self::query_swapchain_support_details(vk_lib, instance, physical_device, surface)?;
        let surface_format = Self::choose_surface_format(&support_details.formats);
        let present_mode = Self::choose_present_mode(&support_details.present_modes);
        let image_extent = Self::choose_extent(window, &support_details.capabilities);

        let queue_info = utils::query_queue_families(vk_lib, instance, physical_device, surface)?;
        let queue_indices = [
            queue_info
                .compute_index
                .ok_or(anyhow!("no compute index found"))?,
            queue_info
                .present_index
                .ok_or(anyhow!("no present index found"))?,
        ];

        let (image_sharing_mode, queue_family_count, queue_family_indices) =
            if queue_info.compute_index.unwrap() == queue_info.present_index.unwrap() {
                (vk::SharingMode::EXCLUSIVE, 0, ptr::null())
            } else {
                (vk::SharingMode::CONCURRENT, 2, queue_indices.as_ptr())
            };

        let image_count = if support_details.capabilities.max_image_count > 0 {
            (support_details.capabilities.min_image_count + 1)
                .min(support_details.capabilities.max_image_count)
        } else {
            support_details.capabilities.min_image_count + 1
        };

        let create_info = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode,
            queue_family_index_count: queue_family_count,
            p_queue_family_indices: queue_family_indices,
            pre_transform: support_details.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        };
        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None) }?;

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;

        Ok((swapchain, surface_format.format, image_extent, images))
    }

    fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        for format in formats {
            if format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }

        formats[0]
    }

    fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_extent(window: &Window, capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let min_extent = capabilities.min_image_extent;
            let max_extent = capabilities.max_image_extent;
            let size = window.inner_size();

            let actual_extent = vk::Extent2D {
                width: size.width.clamp(min_extent.width, max_extent.width),
                height: size.height.clamp(min_extent.height, max_extent.height),
            };

            actual_extent
        }
    }

    fn query_swapchain_support_details(
        vk_lib: &Entry,
        instance: &Instance,
        device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<SwapchainSupportDetails> {
        let surface_loader = khr::surface::Instance::new(vk_lib, instance);
        let capabilities =
            unsafe { surface_loader.get_physical_device_surface_capabilities(device, surface) }?;
        let formats =
            unsafe { surface_loader.get_physical_device_surface_formats(device, surface) }?;
        let present_modes =
            unsafe { surface_loader.get_physical_device_surface_present_modes(device, surface) }?;

        Ok(SwapchainSupportDetails {
            capabilities,
            formats,
            present_modes,
        })
    }

    pub fn is_device_suitable(
        vk_lib: &Entry,
        instance: &Instance,
        device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<bool> {
        let support_details =
            Self::query_swapchain_support_details(vk_lib, instance, device, surface)?;
        Ok(!support_details.formats.is_empty() && !support_details.present_modes.is_empty())
    }

    pub fn required_device_extensions() -> &'static [*const c_char] {
        const EXTENSIONS: &[*const c_char] = &[khr::swapchain::NAME.as_ptr()];
        EXTENSIONS
    }
}

impl Drop for WindowData {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("failed to wait for device idle");
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
