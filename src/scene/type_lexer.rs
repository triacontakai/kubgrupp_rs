enum Token<'a> {
    LSqBracket,
    RSqBracket,
    Semicolon,
    Identifier(&'a str),
    Integer(usize),
}

pub struct TokenIter {

}
