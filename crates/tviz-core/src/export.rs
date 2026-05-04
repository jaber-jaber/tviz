use std::collections::BTreeSet;

use crate::ir::ModelIr;

pub fn export_dot(ir: &ModelIr) -> String {
    let mut out = String::new();
    out.push_str("digraph tviz {\n");
    out.push_str("  graph [rankdir=TB, bgcolor=\"#0b1020\", pad=\"0.4\", nodesep=\"0.6\", ranksep=\"0.8\"];\n");
    out.push_str("  node [shape=box, style=\"rounded,filled\", fontname=\"Inter\", fontsize=10, color=\"#7dd3fc\", fontcolor=\"#e5edff\", fillcolor=\"#172033\"];\n");
    out.push_str("  edge [color=\"#8aa3b6\", arrowsize=0.7, fontcolor=\"#9fb3c8\", fontname=\"Inter\", fontsize=9];\n");

    for group in &ir.groups {
        out.push_str(&format!("  subgraph cluster_{} {{\n", dot_id(&group.id)));
        out.push_str(&format!(
            "    label=\"{}{}\";\n    color=\"#facc15\";\n",
            escape_dot(&group.label),
            if group.repeated > 1 {
                format!(" x{}", group.repeated)
            } else {
                String::new()
            }
        ));
        let children = group.children.iter().collect::<BTreeSet<_>>();
        for node in &ir.nodes {
            if children.contains(&node.id) {
                out.push_str(&format!("    {};\n", dot_id(&node.id)));
            }
        }
        out.push_str("  }\n");
    }

    for node in &ir.nodes {
        out.push_str(&format!(
            "  {} [label=\"{}\", fillcolor=\"{}\"];\n",
            dot_id(&node.id),
            escape_dot(&node_label(node.label.as_str(), node.kind.as_str())),
            style_fill(node.style.as_str())
        ));
    }

    for edge in &ir.edges {
        out.push_str(&format!(
            "  {} -> {} [label=\"{}\", color=\"{}\"];\n",
            dot_id(&edge.from),
            dot_id(&edge.to),
            escape_dot(&edge.kind),
            edge_color(&edge.kind)
        ));
    }

    out.push_str("}\n");
    out
}

pub fn export_svg(ir: &ModelIr) -> String {
    let dot = escape_xml(&export_dot(ir));
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="920" height="620" viewBox="0 0 920 620">
  <rect width="920" height="620" fill="#0b1020"/>
  <text x="28" y="44" fill="#e5edff" font-family="monospace" font-size="22">{}</text>
  <text x="28" y="72" fill="#9fb3c8" font-family="monospace" font-size="13">DOT source embedded below. Render with: dot -Tsvg file.dot -o graph.svg</text>
  <foreignObject x="28" y="100" width="864" height="492">
    <pre xmlns="http://www.w3.org/1999/xhtml" style="color:#d8e2f0;font:12px monospace;white-space:pre-wrap;margin:0">{}</pre>
  </foreignObject>
</svg>
"##,
        escape_xml(&ir.model.name),
        dot
    )
}

fn node_label(label: &str, kind: &str) -> String {
    if label == kind {
        label.to_string()
    } else {
        format!("{label}\\n{kind}")
    }
}

fn dot_id(id: &str) -> String {
    let mut out = String::from("n_");
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

fn escape_dot(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn style_fill(style: &str) -> &'static str {
    match style {
        "input" => "#164e63",
        "conv" => "#065f46",
        "norm" => "#334155",
        "activation" => "#713f12",
        "attention" => "#166534",
        "mlp" => "#7c2d12",
        "embedding" => "#3730a3",
        "pooling" => "#155e75",
        "output" => "#14532d",
        _ => "#172033",
    }
}

fn edge_color(kind: &str) -> &'static str {
    match kind {
        "skip" | "residual" => "#c084fc",
        "branch" => "#f472b6",
        "join" => "#a78bfa",
        _ => "#8aa3b6",
    }
}
