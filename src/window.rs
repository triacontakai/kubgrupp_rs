use std::{ffi::c_char, ptr};

use anyhow::{anyhow, Result};
use ash::{khr, vk, Device, Entry, Instance};
use winit::window::Window;

use crate::{defer::Defer, utils};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct WindowData {
    swapchain: vk::SwapchainKHR,
    surface: vk::SurfaceKHR,
    window: Window,

    swapchain_loader: khr::swapchain::Device,
    surface_loader: khr::surface::Instance,
    vk_lib: Entry,
    device: Device,
    instance: Instance,
    physical_device: vk::PhysicalDevice,

    image_extent: vk::Extent2D,
    images: Vec<vk::Image>,
    current_image: u32,

    image_semaphores: Vec<vk::Semaphore>,
    frame_fences: Vec<vk::Fence>,
    render_semaphores: Vec<vk::Semaphore>,
    current_frame: usize,
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl WindowData {
    pub const DEFAULT_WIDTH: u32 = 1000;
    pub const DEFAULT_HEIGHT: u32 = 1000;

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

        let (swapchain, image_extent, images) =
            Self::create_swapchain(vk_lib, instance, device, physical_device, *surface, &window)?;

        let image_count = images.len();
        let (image_semaphores, frame_fences, render_semaphores) =
            Self::create_sync_objects(device, image_count)?;

        let surface = surface.undefer();
        Ok(WindowData {
            surface_loader,
            swapchain_loader,
            swapchain,
            surface,
            window,
            vk_lib: vk_lib.clone(),
            device: device.clone(),
            instance: instance.clone(),
            physical_device,
            image_extent,
            images,
            current_image: 0,
            image_semaphores,
            frame_fences,
            render_semaphores,
            current_frame: 0,
        })
    }

    pub fn present(&mut self, queue: vk::Queue) -> Result<()> {
        let render_semaphore = self.render_semaphores[self.current_image as usize];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &raw const render_semaphore,
            swapchain_count: 1,
            p_swapchains: &raw const self.swapchain,
            p_image_indices: &raw const self.current_image,
            ..Default::default()
        };

        let needs_recreate =
            match unsafe { self.swapchain_loader.queue_present(queue, &present_info) } {
                Ok(suboptimal) => suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
                Err(e) => return Err(e.into()),
            };

        if needs_recreate {
            self.recreate_swapchain()?;
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None)
        };

        let (swapchain, image_extent, images) = Self::create_swapchain(
            &self.vk_lib,
            &self.instance,
            &self.device,
            self.physical_device,
            self.surface,
            &self.window,
        )?;

        if images.len() != self.images.len() {
            self.recreate_render_semaphores(images.len())?;
        }

        self.swapchain = swapchain;
        self.image_extent = image_extent;
        self.images = images;
        Ok(())
    }

    fn recreate_render_semaphores(&mut self, count: usize) -> Result<()> {
        unsafe {
            for semaphore in &self.render_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
        }
        self.render_semaphores = (0..count)
            .map(|_| unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(())
    }

    pub fn get_current_semaphores(&self) -> (vk::Semaphore, vk::Semaphore) {
        (
            self.image_semaphores[self.current_frame],
            self.render_semaphores[self.current_image as usize],
        )
    }

    pub fn get_current_flight_fence(&self) -> vk::Fence {
        self.frame_fences[self.current_frame]
    }

    pub fn acquire_next_image(&mut self) -> Result<(vk::Image, u32)> {
        let frame_fence = self.frame_fences[self.current_frame];
        let image_semaphore = self.image_semaphores[self.current_frame];

        unsafe {
            self.device
                .wait_for_fences(&[frame_fence], true, u64::MAX)?;
            self.device.reset_fences(&[frame_fence])?;
        }

        self.current_image = match self.do_acquire(image_semaphore) {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                self.do_acquire(image_semaphore)?.0
            }
            Err(e) => return Err(e.into()),
        };

        Ok((self.images[self.current_image as usize], self.current_image))
    }

    fn do_acquire(&self, semaphore: vk::Semaphore) -> std::result::Result<(u32, bool), vk::Result> {
        unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore,
                vk::Fence::null(),
            )
        }
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn get_size(&self) -> (u32, u32) {
        (self.image_extent.width, self.image_extent.height)
    }

    fn create_sync_objects(
        device: &Device,
        swapchain_image_count: usize,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Fence>, Vec<vk::Semaphore>)> {
        let mut image_semaphores = Vec::new();
        let mut frame_fences = Vec::new();
        let mut render_semaphores = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None)? }
            };

            let frame_fence = {
                let fence_info = vk::FenceCreateInfo {
                    flags: vk::FenceCreateFlags::SIGNALED,
                    ..Default::default()
                };
                unsafe { device.create_fence(&fence_info, None)? }
            };

            image_semaphores.push(image_semaphore);
            frame_fences.push(frame_fence);
        }

        for _ in 0..swapchain_image_count {
            let render_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None)? }
            };
            render_semaphores.push(render_semaphore);
        }

        Ok((image_semaphores, frame_fences, render_semaphores))
    }

    fn create_swapchain(
        vk_lib: &Entry,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        window: &Window,
    ) -> Result<(vk::SwapchainKHR, vk::Extent2D, Vec<vk::Image>)> {
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

        Ok((swapchain, image_extent, images))
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

            vk::Extent2D {
                width: size.width.clamp(min_extent.width, max_extent.width),
                height: size.height.clamp(min_extent.height, max_extent.height),
            }
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

            for semaphore in &self.image_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
            for fence in &self.frame_fences {
                self.device.destroy_fence(*fence, None);
            }
            for semaphore in &self.render_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
