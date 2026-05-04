use std::collections::BTreeMap;

use crate::ir::{Edge, Group, ModelInfo, ModelIr, Node, TensorInfo};

#[derive(Debug, Clone)]
enum JsonValue {
    Null,
    Bool,
    Number(u64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

pub fn parse_model_ir(source: &str) -> Result<ModelIr, String> {
    let mut parser = Parser::new(source);
    let value = parser.parse()?;
    let root = expect_object(&value, "root")?;

    let model = expect_object(get(root, "model")?, "model")?;
    let inputs = get_array(root, "inputs")?
        .iter()
        .map(parse_tensor_info)
        .collect::<Result<Vec<_>, _>>()?;
    let nodes = get_array(root, "nodes")?
        .iter()
        .map(parse_node)
        .collect::<Result<Vec<_>, _>>()?;
    let edges = get_array(root, "edges")?
        .iter()
        .map(parse_edge)
        .collect::<Result<Vec<_>, _>>()?;
    let groups = get_array(root, "groups")?
        .iter()
        .map(parse_group)
        .collect::<Result<Vec<_>, _>>()?;
    let warnings = get_array(root, "warnings")?
        .iter()
        .map(|value| expect_string(value, "warnings[]").cloned())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ModelIr {
        schema_version: string_field(root, "schema_version")?,
        model: ModelInfo {
            name: string_field(model, "name")?,
            source: string_field(model, "source")?,
            total_params: number_field(model, "total_params")?,
            trainable_params: number_field(model, "trainable_params")?,
        },
        inputs,
        nodes,
        edges,
        groups,
        warnings,
    })
}

fn parse_tensor_info(value: &JsonValue) -> Result<TensorInfo, String> {
    let object = expect_object(value, "inputs[]")?;
    Ok(TensorInfo {
        name: string_field(object, "name")?,
        dtype: string_field(object, "dtype")?,
        shape: stringish_array_field(object, "shape")?,
    })
}

fn parse_node(value: &JsonValue) -> Result<Node, String> {
    let object = expect_object(value, "nodes[]")?;
    let attributes = match object.get("attributes") {
        Some(JsonValue::Object(values)) => values
            .iter()
            .map(|(key, value)| Ok((key.clone(), value_to_label(value))))
            .collect::<Result<BTreeMap<_, _>, String>>()?,
        _ => BTreeMap::new(),
    };

    Ok(Node {
        id: string_field(object, "id")?,
        label: string_field(object, "label")?,
        kind: string_field(object, "kind")?,
        module_path: string_field(object, "module_path")?,
        params: number_field(object, "params")?,
        trainable_params: number_field(object, "trainable_params")?,
        input_shapes: stringish_array_field(object, "input_shapes")?,
        output_shapes: stringish_array_field(object, "output_shapes")?,
        attributes,
        style: string_field(object, "style")?,
        repeated: number_field(object, "repeated")?,
        depth: number_field(object, "depth")?,
    })
}

fn parse_edge(value: &JsonValue) -> Result<Edge, String> {
    let object = expect_object(value, "edges[]")?;
    Ok(Edge {
        from: string_field(object, "from")?,
        to: string_field(object, "to")?,
        kind: string_field(object, "kind")?,
    })
}

fn parse_group(value: &JsonValue) -> Result<Group, String> {
    let object = expect_object(value, "groups[]")?;
    Ok(Group {
        id: string_field(object, "id")?,
        label: string_field(object, "label")?,
        children: stringish_array_field(object, "children")?,
        repeated: number_field(object, "repeated")?,
    })
}

fn get<'a>(object: &'a BTreeMap<String, JsonValue>, key: &str) -> Result<&'a JsonValue, String> {
    object
        .get(key)
        .ok_or_else(|| format!("missing JSON field {key:?}"))
}

fn get_array<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<&'a [JsonValue], String> {
    match get(object, key)? {
        JsonValue::Array(values) => Ok(values),
        _ => Err(format!("field {key:?} must be an array")),
    }
}

fn expect_object<'a>(
    value: &'a JsonValue,
    label: &str,
) -> Result<&'a BTreeMap<String, JsonValue>, String> {
    match value {
        JsonValue::Object(object) => Ok(object),
        _ => Err(format!("{label} must be an object")),
    }
}

fn expect_string<'a>(value: &'a JsonValue, label: &str) -> Result<&'a String, String> {
    match value {
        JsonValue::String(value) => Ok(value),
        _ => Err(format!("{label} must be a string")),
    }
}

fn string_field(object: &BTreeMap<String, JsonValue>, key: &str) -> Result<String, String> {
    Ok(expect_string(get(object, key)?, key)?.clone())
}

fn number_field(object: &BTreeMap<String, JsonValue>, key: &str) -> Result<u64, String> {
    match get(object, key)? {
        JsonValue::Number(value) => Ok(*value),
        _ => Err(format!("field {key:?} must be a non-negative integer")),
    }
}

fn stringish_array_field(
    object: &BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<Vec<String>, String> {
    get_array(object, key)?
        .iter()
        .map(|value| Ok(value_to_label(value)))
        .collect()
}

fn value_to_label(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool => "bool".to_string(),
        JsonValue::Number(number) => number.to_string(),
        JsonValue::String(value) => value.clone(),
        JsonValue::Array(values) => values
            .iter()
            .map(value_to_label)
            .collect::<Vec<_>>()
            .join("x"),
        JsonValue::Object(_) => "{...}".to_string(),
    }
}

struct Parser<'a> {
    input: &'a [u8],
    index: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            index: 0,
        }
    }

    fn parse(&mut self) -> Result<JsonValue, String> {
        let value = self.parse_value()?;
        self.skip_ws();
        if self.index != self.input.len() {
            return Err(format!("unexpected trailing JSON at byte {}", self.index));
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'0'..=b'9') => self.parse_number().map(JsonValue::Number),
            Some(b't') => {
                self.expect_literal("true")?;
                Ok(JsonValue::Bool)
            }
            Some(b'f') => {
                self.expect_literal("false")?;
                Ok(JsonValue::Bool)
            }
            Some(b'n') => {
                self.expect_literal("null")?;
                Ok(JsonValue::Null)
            }
            Some(byte) => Err(format!(
                "unexpected byte {:?} at JSON byte {}",
                byte as char, self.index
            )),
            None => Err("unexpected end of JSON".to_string()),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect(b'{')?;
        let mut object = BTreeMap::new();
        loop {
            self.skip_ws();
            if self.consume(b'}') {
                break;
            }

            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.parse_value()?;
            object.insert(key, value);

            self.skip_ws();
            if self.consume(b'}') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(JsonValue::Object(object))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect(b'[')?;
        let mut values = Vec::new();
        loop {
            self.skip_ws();
            if self.consume(b']') {
                break;
            }
            values.push(self.parse_value()?);
            self.skip_ws();
            if self.consume(b']') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(JsonValue::Array(values))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut value = Vec::new();
        while let Some(byte) = self.next() {
            match byte {
                b'"' => {
                    return String::from_utf8(value)
                        .map_err(|error| format!("invalid UTF-8 in string: {error}"));
                }
                b'\\' => {
                    let escaped = self.parse_escape()?;
                    let mut buffer = [0; 4];
                    value.extend_from_slice(escaped.encode_utf8(&mut buffer).as_bytes());
                }
                byte if byte.is_ascii_control() => {
                    return Err(format!("control byte in string at byte {}", self.index));
                }
                byte => value.push(byte),
            }
        }
        Err("unterminated string".to_string())
    }

    fn parse_escape(&mut self) -> Result<char, String> {
        match self.next() {
            Some(b'"') => Ok('"'),
            Some(b'\\') => Ok('\\'),
            Some(b'/') => Ok('/'),
            Some(b'b') => Ok('\u{0008}'),
            Some(b'f') => Ok('\u{000c}'),
            Some(b'n') => Ok('\n'),
            Some(b'r') => Ok('\r'),
            Some(b't') => Ok('\t'),
            Some(b'u') => {
                let mut code = 0u32;
                for _ in 0..4 {
                    let byte = self
                        .next()
                        .ok_or_else(|| "incomplete unicode escape".to_string())?;
                    code = code * 16
                        + (byte as char)
                            .to_digit(16)
                            .ok_or_else(|| "invalid unicode escape".to_string())?;
                }
                char::from_u32(code).ok_or_else(|| "invalid unicode scalar".to_string())
            }
            Some(byte) => Err(format!("invalid escape \\{}", byte as char)),
            None => Err("incomplete escape".to_string()),
        }
    }

    fn parse_number(&mut self) -> Result<u64, String> {
        let start = self.index;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.index += 1;
        }
        let raw = std::str::from_utf8(&self.input[start..self.index])
            .map_err(|error| format!("invalid number bytes: {error}"))?;
        raw.parse::<u64>()
            .map_err(|error| format!("invalid number {raw:?}: {error}"))
    }

    fn expect_literal(&mut self, literal: &str) -> Result<(), String> {
        for expected in literal.as_bytes() {
            self.expect(*expected)?;
        }
        Ok(())
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.index += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        match self.next() {
            Some(byte) if byte == expected => Ok(()),
            Some(byte) => Err(format!(
                "expected {:?} at byte {}, found {:?}",
                expected as char,
                self.index.saturating_sub(1),
                byte as char
            )),
            None => Err(format!(
                "expected {:?}, found end of JSON",
                expected as char
            )),
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.index).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.index += 1;
        Some(byte)
    }
}
