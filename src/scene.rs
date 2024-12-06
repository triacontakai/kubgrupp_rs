pub mod scenes;
mod type_lexer;

pub trait Scene {
    type Updates: ?Sized;
}
