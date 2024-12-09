use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::ptr;
use std::sync::Arc;

use anyhow::Result;
use ash::vk::{
    DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCreateInfoEXT, EXT_DEBUG_UTILS_NAME,
};
use ash::{ext, khr, Device};
use ash::{
    vk::{self},
    Entry, Instance,
};

use debug::DebugUtilsData;
use defer::Defer;
use env_logger::Builder;
use glam::{Mat3, Mat4, Vec3, Vec4};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use log::{debug, warn, LevelFilter};
use render::renderers::RaytraceRenderer;
use render::Renderer;
use scene::scenes::mesh::{MeshScene, MeshSceneUpdate};
use scene::Scene;
use utils::{query_queue_families, QueueFamilyInfo};
use window::WindowData;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, KeyEvent, RawKeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, KeyCode, PhysicalKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{CursorGrabMode, WindowAttributes, WindowId};

mod debug;
mod defer;
mod features;
mod render;
mod scene;
mod utils;
mod window;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

#[cfg(debug_assertions)]
const DEBUG_MODE: bool = true;

#[cfg(not(debug_assertions))]
const DEBUG_MODE: bool = false;

const APPLICATION_NAME: &'static str = concat!(env!("CARGO_PKG_NAME"), "\0");

struct MeshApp<R> {
    // WARNING: ORDER MATTERS HERE!!!
    // fields are dropped from top to bottom (not bottom to top like C++)
    // make sure to also update the Drop impl when adding fields
    renderer: Option<R>,
    window: Option<WindowData>,
    allocator: Option<Allocator>,
    debug_data: Option<DebugUtilsData>,
    physical_device: Option<vk::PhysicalDevice>,
    device: Option<Device>,
    instance: Instance,
    vk_lib: Entry,
    scene: MeshScene,
    position: Vec3,
    direction: Vec3,
    view_updated: bool,
    w_down: bool,
    a_down: bool,
    s_down: bool,
    d_down: bool,
    shift_down: bool,
    space_down: bool,
}

impl<R> MeshApp<R>
where
    R: Renderer<MeshScene, WindowData>,
{
    pub fn new(event_loop: &EventLoop<()>, scene: MeshScene, debug_mode: bool) -> Result<Self> {
        let vk_lib = unsafe { Entry::load().expect("failed to load Vulkan library") };

        let enable_vk_debug = debug_mode && Self::is_vk_debug_supported(&vk_lib)?;
        if debug_mode && !enable_vk_debug {
            warn!("running in debug mode, but validation layer/debug_utils extension are not found/supported");
        }

        let mut debug_utils_info = enable_vk_debug.then(|| DebugUtilsMessengerCreateInfoEXT {
            message_severity: DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::INFO
                | DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            message_type: DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(debug::debug_callback),
            p_user_data: ptr::null_mut(),
            ..Default::default()
        });

        let validation_feature_enable = [
            vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
        ];
        let mut validation_features = enable_vk_debug.then(|| vk::ValidationFeaturesEXT {
            enabled_validation_feature_count: validation_feature_enable.len() as u32,
            p_enabled_validation_features: validation_feature_enable.as_ptr(),
            ..Default::default()
        });

        let instance = Self::create_instance(
            &vk_lib,
            event_loop,
            debug_utils_info.as_mut(),
            validation_features.as_mut(),
        )?
        .defer(|x| unsafe { x.destroy_instance(None) });

        let debug_data = debug_utils_info
            .map(|x| {
                let loader = ext::debug_utils::Instance::new(&vk_lib, &instance);
                unsafe { DebugUtilsData::new(loader, &x) }
            })
            .transpose()?;

        let position = scene.camera.view.inverse().col(3).truncate();
        let direction = scene.camera.view.inverse().col(2).truncate();

        Ok(MeshApp {
            renderer: None,
            window: None,
            allocator: None,
            debug_data,
            device: None,
            physical_device: None,
            instance: instance.undefer(),
            vk_lib,
            scene,
            position,
            direction,
            view_updated: false,
            w_down: false,
            a_down: false,
            s_down: false,
            d_down: false,
            shift_down: false,
            space_down: false,
        })
    }

    fn is_vk_debug_supported(vk_lib: &Entry) -> Result<bool> {
        let available_layers = unsafe { vk_lib.enumerate_instance_layer_properties()? };
        let supported_extensions = unsafe { vk_lib.enumerate_instance_extension_properties(None)? };

        // technically we can short circuit this but it really doesnt matter
        // its more readable like this :)
        let validation_layer_supported = available_layers
            .iter()
            .any(|x| unsafe { CStr::from_ptr(x.layer_name.as_ptr()) == VALIDATION_LAYER });
        let debug_extensions_supported = supported_extensions
            .iter()
            .any(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) == EXT_DEBUG_UTILS_NAME });

        Ok(validation_layer_supported && debug_extensions_supported)
    }

    fn get_layers_and_extensions(
        event_loop: &EventLoop<()>,
        use_debug_layers: bool,
    ) -> Result<(Vec<*const c_char>, Vec<*const c_char>)> {
        let mut layers = Vec::new();
        let mut extensions = Vec::new();

        if use_debug_layers {
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(EXT_DEBUG_UTILS_NAME.as_ptr());
        }

        let display_handle = event_loop.owned_display_handle();
        let raw_display_handle = display_handle.display_handle()?.as_raw();
        let required_extensions = ash_window::enumerate_required_extensions(raw_display_handle)?;
        let required_renderer_extensions = R::required_instance_extensions();

        // could check if extensions are supported to print exactly the extensions that arent supported
        // but eh, im lazy

        extensions.extend_from_slice(required_extensions);
        extensions.extend_from_slice(required_renderer_extensions);

        Ok((layers, extensions))
    }

    fn create_instance(
        vk_lib: &Entry,
        event_loop: &EventLoop<()>,
        debug_utils_info: Option<&mut DebugUtilsMessengerCreateInfoEXT>,
        validation_features: Option<&mut vk::ValidationFeaturesEXT>,
    ) -> Result<Instance> {
        let (layers, extensions) =
            Self::get_layers_and_extensions(event_loop, debug_utils_info.is_some())?;

        let app_info = vk::ApplicationInfo {
            p_application_name: APPLICATION_NAME.as_ptr() as *const c_char,
            application_version: vk::make_api_version(
                0,
                env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
            ),
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let mut create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };

        if let Some(debug_utils_info) = debug_utils_info {
            create_info = create_info.push_next(debug_utils_info);
        }

        if let Some(validation_features) = validation_features {
            create_info = create_info.push_next(validation_features);
        }

        unsafe { Ok(vk_lib.create_instance(&create_info, None)?) }
    }

    fn is_device_suitable(
        &self,
        device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<bool> {
        // check compatibility of device with window and renderer
        let required_renderer_extensions = R::required_device_extensions();
        let required_window_extensions = WindowData::required_device_extensions();
        let required_extensions =
            [required_renderer_extensions, required_window_extensions].concat();
        let required_features = R::required_features();

        let supported_extensions = unsafe {
            self.instance
                .enumerate_device_extension_properties(device)?
        };

        // check that all required extensions and features are supported (i.e. required is a subset of supported)
        for ext in required_extensions {
            let ext_name = unsafe { CStr::from_ptr(ext) };
            if !supported_extensions
                .iter()
                .any(|x| x.extension_name_as_c_str().unwrap() == ext_name)
            {
                return Ok(false);
            }
        }

        if !required_features.supported(&self.instance, device) {
            return Ok(false);
        }

        if !WindowData::is_device_suitable(&self.vk_lib, &self.instance, device, surface)? {
            return Ok(false);
        }

        let queue_family_info =
            utils::query_queue_families(&self.vk_lib, &self.instance, device, surface)?;
        Ok(R::has_required_queue_families(&queue_family_info))
    }

    fn pick_physical_device(
        &self,
        devices: impl Iterator<Item = vk::PhysicalDevice>,
    ) -> Option<vk::PhysicalDevice> {
        // could make a smarter device scoring system, but let's just take either the first discrete GPU device
        // or the first device that works if there is no discrete GPU
        // in the future could expand this to have the renderer score devices based on what would be best for it
        let mut devices = devices.peekable();
        let first = devices.peek().cloned();

        for device in devices {
            let properties = unsafe { self.instance.get_physical_device_properties(device) };
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                return Some(device);
            }
        }

        first
    }

    fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_info: &QueueFamilyInfo,
    ) -> Result<Device> {
        let enabled_extensions = [
            R::required_device_extensions(),
            WindowData::required_device_extensions(),
        ]
        .concat();
        let enabled_features = R::required_features();

        let queue_info = R::get_queue_info(queue_family_info);

        let create_info = vk::DeviceCreateInfo {
            p_next: enabled_features.get() as *const _ as *const c_void,
            queue_create_info_count: queue_info.len() as u32,
            p_queue_create_infos: queue_info.as_ptr(),
            enabled_extension_count: enabled_extensions.len() as u32,
            pp_enabled_extension_names: enabled_extensions.as_ptr(),
            p_enabled_features: ptr::null(),
            ..Default::default()
        };
        let device = unsafe {
            self.instance
                .create_device(physical_device, &create_info, None)
        }?;

        Ok(device)
    }
}

impl<R> Drop for MeshApp<R> {
    fn drop(&mut self) {
        drop(self.renderer.take());
        drop(self.window.take());
        self.device
            .take()
            .map(|x| unsafe { x.destroy_device(None) });
        drop(self.debug_data.take());
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl<R> ApplicationHandler for MeshApp<R>
where
    R: Renderer<MeshScene, WindowData>,
    MeshScene: Scene,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        debug!("App resuming...");
        if self.window.is_none() {
            let surface_loader = khr::surface::Instance::new(&self.vk_lib, &self.instance);

            let window = event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_inner_size(PhysicalSize::new(800, 800))
                        .with_title("kubgrupp"),
                )
                .unwrap();
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))
                .expect("could not confine cursor");
            window.set_cursor_visible(false);

            let display_handle = window.display_handle().unwrap();
            let window_handle = window.window_handle().unwrap();
            let surface = unsafe {
                ash_window::create_surface(
                    &self.vk_lib,
                    &self.instance,
                    display_handle.as_raw(),
                    window_handle.as_raw(),
                    None,
                )
            }
            .unwrap()
            .defer(|x| unsafe { surface_loader.destroy_surface(*x, None) });
            debug!("Created window: {:?}", window.title());

            // surface created - now we pick physical device
            // we start by checking if the device works for the application
            // we then let the renderer pick the optimal device out of this selection
            let devices = unsafe {
                self.instance
                    .enumerate_physical_devices()
                    .expect("failed to enumerate physical devices")
            };

            let valid_devices = devices.into_iter().filter(|device| {
                // skip and log if check function returns Err
                self.is_device_suitable(*device, *surface)
                    .unwrap_or_else(|e| {
                        warn!("failed to check if device was suitable: {}", e);
                        false
                    })
            });

            let physical_device = self
                .pick_physical_device(valid_devices)
                .expect("failed to find compatible physical device");

            let queue_family_info =
                query_queue_families(&self.vk_lib, &self.instance, physical_device, *surface)
                    .expect("failed to find queue family info");
            let device = self
                .create_device(physical_device, &queue_family_info)
                .expect("failed to create device");

            self.allocator = Some(
                Allocator::new(&AllocatorCreateDesc {
                    instance: self.instance.clone(),
                    device: device.clone(),
                    physical_device,
                    debug_settings: Default::default(),
                    buffer_device_address: true,
                    allocation_sizes: Default::default(),
                })
                .expect("failed to create allocator"),
            );

            self.window = Some(
                WindowData::new(
                    &self.vk_lib,
                    &self.instance,
                    &device,
                    physical_device,
                    *surface,
                    window,
                )
                .expect("swapchain creation failed"),
            );
            surface.undefer();

            self.physical_device = Some(physical_device);
            self.device = Some(device.clone());
            self.renderer = Some(
                R::new(
                    &self.vk_lib,
                    &self.instance,
                    &device,
                    physical_device,
                    &queue_family_info,
                    self.window.as_ref().unwrap(),
                    self.allocator.as_mut().unwrap(),
                )
                .expect("failed to create renderer"),
            );

            // this is where we load the initial scene into the renderer
            // future updates come through the event loop through the render function
            self.renderer
                .as_mut()
                .unwrap()
                .ingest_scene(&self.scene, self.allocator.as_mut().unwrap())
                .expect("failed to ingest scene");
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                debug!("Closing window...");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                device_id: _device_id,
                event: input_event,
                is_synthetic: _is_synthetic,
            } => {
                match input_event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        self.w_down = input_event.state.is_pressed()
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        self.a_down = input_event.state.is_pressed()
                    }
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        self.s_down = input_event.state.is_pressed()
                    }
                    PhysicalKey::Code(KeyCode::KeyD) => {
                        self.d_down = input_event.state.is_pressed()
                    }
                    PhysicalKey::Code(KeyCode::ShiftLeft) => {
                        self.shift_down = input_event.state.is_pressed()
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.space_down = input_event.state.is_pressed()
                    }
                    _ => (),
                }
                match input_event.key_without_modifiers().as_ref() {
                    Key::Character("w") => {
                        self.position += self.direction * 0.05f32;
                        self.view_updated = true;
                    }
                    _ => (),
                }
            }
            WindowEvent::RedrawRequested => {
                const SPEED: f32 = 0.005f32;
                let horiz_dir: Vec3 =
                    Vec3::new(-self.direction.y, self.direction.x, 0f32).normalize();
                let vert_dir: Vec3 = self.direction.cross(horiz_dir);
                if self.w_down {
                    self.position += SPEED * self.direction;
                    self.view_updated = true;
                }
                if self.s_down {
                    self.position -= SPEED * self.direction;
                    self.view_updated = true;
                }
                if self.a_down {
                    self.position -= SPEED * horiz_dir;
                    self.view_updated = true;
                }
                if self.d_down {
                    self.position += SPEED * horiz_dir;
                    self.view_updated = true;
                }
                if self.shift_down {
                    self.position -= SPEED * vert_dir;
                    self.view_updated = true;
                }
                if self.space_down {
                    self.position += SPEED * vert_dir;
                    self.view_updated = true;
                }

                let updates = if self.view_updated {
                    vec![MeshSceneUpdate::NewView(Mat4::look_to_lh(
                        self.position,
                        self.direction,
                        Vec3::new(0f32, 0f32, 1f32),
                    ))]
                } else {
                    vec![]
                };

                self.renderer
                    .as_mut()
                    .unwrap()
                    .render_to(&updates, self.window.as_mut().unwrap())
                    .expect("failed to render to target");

                self.view_updated = false;

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                let (sx, sy) = self.window.as_ref().unwrap().get_size();
                let ry_axis = Vec3::new(-self.direction.y, self.direction.x, 0f32);
                let rx_axis = Vec3::new(0f32, 0f32, 1f32);
                let rx = (dx / sx as f64) as f32;
                let ry = (dy / sy as f64) as f32;

                let rotation = Mat3::from_axis_angle(rx_axis, rx)
                    * Mat3::from_axis_angle(ry_axis.normalize(), ry);

                let new_direction = rotation * self.direction;

                if new_direction.truncate().dot(self.direction.truncate()) >= 0f32 {
                    self.direction = new_direction.normalize();
                }

                self.view_updated = true;
            }
            _ => (),
        }
    }
}

fn main() {
    Builder::new()
        .filter_level(LevelFilter::Debug)
        .parse_default_env()
        .init();

    let event_loop = EventLoop::new().unwrap();

    let file = File::open("resources/scenes/cubes.toml").expect("scene file does not exist");
    let scene = MeshScene::load_from(file).expect("scene could not be loaded");
    let mut app: MeshApp<RaytraceRenderer> = MeshApp::new(&event_loop, scene, DEBUG_MODE).unwrap();
    event_loop.run_app(&mut app).unwrap();
}
