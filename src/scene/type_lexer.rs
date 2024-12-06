pub enum Token<'a> {
    LSqBracket,
    RSqBracket,
    Semicolon,
    Identifier(&'a str),
    Integer(usize),
}

pub struct TokenIter<'a> {
    remaining: &'a str
}

impl<'a> TokenIter<'a> {
    pub fn new(str: &'a str) -> Self {
        Self {
            remaining: str.trim_start()
        }
    }
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // whitespace should have been trimmed from last iteration
        // match directly on token

        todo!()
    }
}
