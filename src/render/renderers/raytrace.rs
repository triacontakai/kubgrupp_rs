use std::{ffi::c_char, sync::LazyLock};

use ash::{khr, vk, Device, Entry, Instance};

use crate::{
    features::{vk_features, VkFeatureGuard, VkFeatures},
    render::Renderer,
    utils::{QueueFamilyInfo, QueueInfo},
    window::WindowData,
};

pub struct RaytraceRenderer {
    vk_lib: Entry,
    instance: Instance,
    device: Device,
    acceleration_structure: vk::AccelerationStructureKHR,
    pipeline: vk::Pipeline,
}

impl Renderer<(), WindowData> for RaytraceRenderer {
    type Error = anyhow::Error;

    fn new(vk_lib: &Entry, instance: &Instance, device: &Device, queue_info: QueueInfo) -> Self {
        RaytraceRenderer {
            vk_lib: vk_lib.clone(),
            instance: instance.clone(),
            device: device.clone(),
            acceleration_structure: Default::default(),
            pipeline: Default::default(),
        }
    }

    fn ingest_scene(&mut self, _scene: &()) {
        let vertices = [
            [-0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, -0.5, 0.0],
        ];
        let vertex_count = vertices.len();
        let vertex_stride = std::mem::size_of_val(&vertices[0]);

    }

    fn render_to(&mut self, _updates: (), target: &mut WindowData) -> Result<(), Self::Error> {
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

