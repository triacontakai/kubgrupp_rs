use anyhow::anyhow;

#[derive(Debug)]
pub enum Token<'a> {
    LSqBracket,
    RSqBracket,
    Semicolon,
    Typename(&'a str),
    Integer(u64),
    LexerError(anyhow::Error),
}

impl<'a> PartialEq for Token<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Typename(l0), Self::Typename(r0)) => l0 == r0,
            (Self::Integer(l0), Self::Integer(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

pub struct TokenIter<'a> {
    remaining: &'a str,
}

impl<'a> TokenIter<'a> {
    pub fn new(str: &'a str) -> Self {
        Self {
            remaining: str.trim_start(),
        }
    }
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.remaining;

        // whitespace should have been trimmed from last iteration
        // therefore we should be right up against the next token or empty
        if remaining.is_empty() {
            return None;
        }

        let mut chars = remaining.chars();
        Some(match chars.next().unwrap() {
            '[' => {
                self.remaining = &chars.as_str().trim_start();
                Token::LSqBracket
            }
            ']' => {
                self.remaining = &chars.as_str().trim_start();
                Token::RSqBracket
            }
            ';' => {
                self.remaining = &chars.as_str().trim_start();
                Token::Semicolon
            }
            c if c.is_ascii_alphabetic() => {
                // get slice first non-alphanumeric character to get identifier name
                let end = remaining
                    .find(|c: char| !c.is_ascii_alphanumeric())
                    .unwrap_or(remaining.len());
                let id = &remaining[..end];
                self.remaining = &remaining[end..];
                Token::Typename(id)
            }
            c if c.is_ascii_digit() => {
                let end = remaining
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(remaining.len());
                let num = remaining[..end].parse().unwrap();
                self.remaining = &remaining[end..];
                Token::Integer(num)
            }
            x => Token::LexerError(anyhow!(
                "invalid start of token found: {} (remaining: {:?})",
                x,
                remaining
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Token, TokenIter};

    #[test]
    fn lex_all() {
        let iter = TokenIter::new("  [[vec3;   5];1] ");
        let tokens: Vec<_> = iter.collect();

        assert_eq!(
            &tokens[..],
            &[
                Token::LSqBracket,
                Token::LSqBracket,
                Token::Typename("vec3"),
                Token::Semicolon,
                Token::Integer(5),
                Token::RSqBracket,
                Token::Semicolon,
                Token::Integer(1),
                Token::RSqBracket,
            ]
        );
    }
}
