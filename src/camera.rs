use std::collections::BTreeMap;
use std::f32::consts::PI;
use std::fmt::Debug;

use glam::{Mat3, Mat4, Vec3};
use winit::keyboard::KeyCode;

use crate::window::WindowData;

#[derive(Debug, Copy, Clone)]
enum Direction {
    None = 0,
    Forward = 0x1,
    Backward = 0x2,
    Left = 0x4,
    Right = 0x8,
    Up = 0x10,
    Down = 0x20,
}

pub struct Camera {
    // matrix from world space to camera space
    view: Mat4,
    // matrix from camera to clip space
    perspective: Mat4,

    fov: f32,

    position: Vec3,
    direction: Vec3,

    key_movements: BTreeMap<KeyCode, (Direction, Box<dyn Fn(&Vec3) -> Vec3>)>,
    movement_direction: u32,
    updated_view: bool,

    speed_modifier: f32,
}

impl Debug for Camera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Camera")
            .field("view", &self.view)
            .field("perspective", &self.perspective)
            .field("fov", &self.fov)
            .field("position", &self.position)
            .field("direction", &self.direction)
            .finish_non_exhaustive()
    }
}

impl Camera {
    const SPEED: f32 = 5f32;

    pub fn new(view: Mat4, fov: f32) -> Camera {
        let mut key_movements: BTreeMap<KeyCode, (Direction, Box<dyn Fn(&Vec3) -> Vec3>)> =
            BTreeMap::new();

        key_movements.insert(
            KeyCode::KeyW,
            (Direction::Forward, Box::new(|dir: &Vec3| *dir)),
        );
        key_movements.insert(
            KeyCode::KeyS,
            (Direction::Backward, Box::new(|dir: &Vec3| -*dir)),
        );
        key_movements.insert(
            KeyCode::KeyA,
            (
                Direction::Left,
                Box::new(|dir: &Vec3| Vec3::new(dir.y, -dir.x, 0f32).normalize()),
            ),
        );
        key_movements.insert(
            KeyCode::KeyD,
            (
                Direction::Right,
                Box::new(|dir: &Vec3| Vec3::new(-dir.y, dir.x, 0f32).normalize()),
            ),
        );
        key_movements.insert(
            KeyCode::ControlLeft,
            (
                Direction::Down,
                Box::new(|dir: &Vec3| dir.cross(Vec3::new(dir.y, -dir.x, 0f32).normalize())),
            ),
        );
        key_movements.insert(
            KeyCode::Space,
            (
                Direction::Up,
                Box::new(|dir: &Vec3| dir.cross(Vec3::new(-dir.y, dir.x, 0f32).normalize())),
            ),
        );

        let fov_radians = fov * PI / 180f32;
        let mut perspective = Mat4::perspective_lh(
            fov_radians,
            WindowData::DEFAULT_WIDTH as f32 / WindowData::DEFAULT_HEIGHT as f32,
            0.1f32,
            1000f32,
        );
        perspective.y_axis = -perspective.y_axis;

        Camera {
            view,
            perspective,
            fov,
            position: view.inverse().col(3).truncate(),
            direction: view.inverse().col(2).truncate(),
            key_movements,
            movement_direction: Direction::None as u32,
            updated_view: false,
            speed_modifier: 1f32,
        }
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) {
        let fov_radians = self.fov * PI / 180f32;
        self.perspective =
            Mat4::perspective_lh(fov_radians, width as f32 / height as f32, 0.1f32, 1000f32);
        self.perspective.y_axis = -self.perspective.y_axis;
    }

    pub fn handle_key_input(&mut self, key: KeyCode, pressed: bool) {
        if let Some((direction, _)) = self.key_movements.get(&key) {
            if pressed {
                self.movement_direction |= *direction as u32;
            } else {
                self.movement_direction &= !(*direction as u32);
            }
        }
        if key == KeyCode::AltLeft {
            self.speed_modifier = 1.0f32 - 0.5f32 * (pressed as u32 as f32);
        }
    }

    pub fn handle_mouse_input(&mut self, rx: f32, ry: f32) {
        let ry_axis = Vec3::new(-self.direction.y, self.direction.x, 0f32);
        let rx_axis = Vec3::new(0f32, 0f32, 1f32);

        let rot_x = Mat3::from_axis_angle(rx_axis, rx);
        let rot_y = Mat3::from_axis_angle(ry_axis.normalize(), ry);

        let new_direction = rot_x * rot_y * self.direction;

        if new_direction.truncate().dot(self.direction.truncate()) < 0f32 {
            self.direction = (rot_x * self.direction).normalize();
        } else {
            self.direction = new_direction.normalize();
        }

        self.updated_view = true;
    }

    pub fn handle_movement(&mut self, dt: f32) {
        for (d, movement_fn) in self.key_movements.values() {
            if self.movement_direction & (*d as u32) == (*d as u32) {
                self.position +=
                    Camera::SPEED * dt * self.speed_modifier * movement_fn(&self.direction);
                self.updated_view = true;
            }
        }
    }

    pub fn update_view(&mut self) -> Option<Mat4> {
        if !self.updated_view {
            return None;
        }

        self.view = Mat4::look_to_lh(self.position, self.direction, Vec3::new(0f32, 0f32, 1f32));
        self.updated_view = false;

        Some(self.view)
    }

    pub fn view(&self) -> Mat4 {
        self.view
    }

    pub fn perspective(&self) -> Mat4 {
        self.perspective
    }
}
