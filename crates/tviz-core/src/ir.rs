use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct ModelIr {
    pub schema_version: String,
    pub model: ModelInfo,
    pub inputs: Vec<TensorInfo>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub groups: Vec<Group>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub name: String,
    pub source: String,
    pub total_params: u64,
    pub trainable_params: u64,
}

#[derive(Debug, Clone, Default)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub module_path: String,
    pub params: u64,
    pub trainable_params: u64,
    pub input_shapes: Vec<String>,
    pub output_shapes: Vec<String>,
    pub attributes: BTreeMap<String, String>,
    pub style: String,
    pub repeated: u64,
    pub depth: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub kind: String,
}

#[derive(Debug, Clone, Default)]
pub struct Group {
    pub id: String,
    pub label: String,
    pub children: Vec<String>,
    pub repeated: u64,
}
