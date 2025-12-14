use std::{
    alloc::{self, alloc_zeroed, dealloc, Layout},
    ptr,
};

use ash::vk::{self, StructureType};

macro_rules! vk_features {
    (
        $first_struct:ty {
            $($base_feature:ident),* $(,)?
        }
        $(,
            $feature_struct:ty {
                $($feature:ident),* $(,)?
            }
        )*
        $(,)?
    ) => {
        {
            use $crate::features::{VkFeatures, EnabledFeatures};
            use ::std::alloc::Layout;
            use ::ash::vk::{self, TaggedStructure, ExtendsPhysicalDeviceFeatures2};

            #[allow(unused_imports)]
            use ::std::mem::offset_of;

            // check that first_struct is a vk::PhysicalDeviceFeatures
            struct MatchingType<T>(T);
            #[allow(dead_code, unreachable_patterns)]
            fn assert_type_eq(mine: MatchingType<$first_struct>) {
                match mine {
                    MatchingType::<vk::PhysicalDeviceFeatures>(_) => ()
                }
            }

            // check that all feature structs are actually feature structs
            #[allow(dead_code)]
            const fn assert_feature_struct<T: ExtendsPhysicalDeviceFeatures2>() {}
            $(assert_feature_struct::<$feature_struct>();)*

            // check that none of the types provided are duplicates
            // this works by triggering the unreachable patterns lint due to
            // duplicate STRUCTURE_TYPE match arms if a struct is duplicated
            #[allow(dead_code)]
            #[deny(unreachable_patterns, reason = "no duplicate feature structs allowed")]
            fn assert_types_neq() {
                match <vk::PhysicalDeviceFeatures2 as TaggedStructure>::STRUCTURE_TYPE {
                    $(
                        <$feature_struct as TaggedStructure>::STRUCTURE_TYPE => (),
                    )*
                    _ => (),
                }
            }

            unsafe {
                let mut features = Vec::new();
                features.push(EnabledFeatures::new(
                    vk::PhysicalDeviceFeatures2::STRUCTURE_TYPE,
                    Layout::new::<vk::PhysicalDeviceFeatures2>(),
                    vec![$(
                        offset_of!(vk::PhysicalDeviceFeatures2, features) + offset_of!($first_struct, $base_feature)
                    ),*],
                ));

                $(
                    features.push(EnabledFeatures::new(
                        <$feature_struct as TaggedStructure>::STRUCTURE_TYPE,
                        Layout::new::<$feature_struct>(),
                        vec![$(
                            offset_of!($feature_struct, $feature)
                        ),*]
                    ));
                )*

                VkFeatures::new(features)
            }
        }
    }
}

pub(crate) use vk_features;

#[derive(Debug)]
pub struct VkFeatureGuard<'a> {
    // static lifetime because this lives for as long as the struct does
    head: *mut vk::PhysicalDeviceFeatures2<'static>,
    parent: &'a VkFeatures,
}

#[derive(Debug)]
pub struct EnabledFeatures {
    s_type: StructureType,
    layout: Layout,

    // field offsets of enabled features
    offsets: Vec<usize>,
}

#[derive(Debug)]
pub struct VkFeatures {
    features: Vec<EnabledFeatures>,
}

impl EnabledFeatures {
    /// Create a new `EnabledFeatures`
    ///
    /// This should never be manually called - use the `vk_features!` macro instead.
    pub unsafe fn new(s_type: StructureType, layout: Layout, offsets: Vec<usize>) -> Self {
        Self {
            s_type,
            layout,
            offsets,
        }
    }
}

impl VkFeatures {
    /// Create a new `VkFeatures`
    ///
    /// This should never be manually called - use the `vk_features!` macro instead.
    pub unsafe fn new(features: Vec<EnabledFeatures>) -> Self {
        Self { features }
    }

    pub fn get_list(&self) -> VkFeatureGuard<'_> {
        VkFeatureGuard::new(self)
    }
}

impl<'a> VkFeatureGuard<'a> {
    fn new(parent: &'a VkFeatures) -> Self {
        // loop through each layout and create p_next linked list
        let mut head: *mut vk::PhysicalDeviceFeatures2 = ptr::null_mut();
        let mut curr = ptr::addr_of_mut!(head) as *mut *mut vk::BaseOutStructure;

        for feature in &parent.features {
            unsafe {
                let layout = feature.layout;
                let mem = alloc_zeroed(layout);
                if mem.is_null() {
                    alloc::handle_alloc_error(layout);
                }

                for &offset in &feature.offsets {
                    let feature_ptr = mem.add(offset) as *mut vk::Bool32;
                    feature_ptr.write(vk::TRUE);
                }

                let struct_ptr = mem as *mut vk::BaseOutStructure;
                (*struct_ptr).s_type = feature.s_type;
                *curr = struct_ptr;
                curr = ptr::addr_of_mut!((*struct_ptr).p_next);
            }
        }

        // head should never be null
        // this is due to invariant that first layout is *always* vk::PhysicalDeviceFeatures2
        assert!(!head.is_null());

        VkFeatureGuard { head, parent }
    }

    /// Gets an immutable reference to the underlying `PhysicalDeviceFeatures2` struct
    pub fn get(&self) -> &vk::PhysicalDeviceFeatures2<'_> {
        unsafe { &*self.head }
    }

    /// Returns `true` if all features are supported, false otherwise
    pub fn supported(&self, instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
        // create copy of features list
        // this copy will be mutated, which breaks the invariant,
        // so we must make sure the user never sees it
        let copy = self.clone();

        // populate feature list with supported features
        unsafe { instance.get_physical_device_features2(device, &mut *copy.head) };

        // check if all requested features are in the list
        let mut curr = copy.head as *mut vk::BaseOutStructure;
        let mut all_features = self.parent.features.iter();
        while !curr.is_null() {
            let features = all_features.next().unwrap();

            for &offset in &features.offsets {
                let feature_ptr = unsafe { curr.byte_add(offset) } as *mut vk::Bool32;
                let supported = unsafe { feature_ptr.read() };
                if supported == vk::FALSE {
                    return false;
                }
            }

            curr = unsafe { (*curr).p_next };
        }

        // we should have gone through all features while iterating
        assert!(all_features.next().is_none());
        true
    }
}

impl Clone for VkFeatureGuard<'_> {
    fn clone(&self) -> Self {
        // initializing new from scratch is fine since we have invariant that user cannot modify anything in the chain
        // therefore all features in the set (and only those features) should be set in this clone
        Self::new(self.parent)
    }
}

impl Drop for VkFeatureGuard<'_> {
    fn drop(&mut self) {
        // iterate through p_next chain and drop everything
        let mut curr = self.head as *mut vk::BaseOutStructure;
        let mut features = self.parent.features.iter();
        while !curr.is_null() {
            let next = unsafe { (*curr).p_next };

            let layout = features.next().unwrap().layout;
            unsafe { dealloc(curr as *mut u8, layout) };

            curr = next;
        }

        // we should have gone through all the features in the process of freeing the list
        assert!(features.next().is_none())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_vkfeatures() {
        let features = vk_features! {
            vk::PhysicalDeviceFeatures {
                alpha_to_one,
            },
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
                acceleration_structure, descriptor_binding_acceleration_structure_update_after_bind,
            },
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
                ray_tracing_pipeline,
            },
        };

        let list = features.get_list();
        let features_to_pass_to_func = list.get();

        let cloned = list.clone();
        println!("{:?}", features_to_pass_to_func);
        println!("{:?}", cloned);
    }
}
