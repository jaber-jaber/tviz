use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::ir::{ModelIr, Node};

pub struct RenderOptions {
    pub color: bool,
    pub granularity: u8,
    pub compact: bool,
    pub max_lines: Option<usize>,
}

pub fn render_model(ir: &ModelIr, options: RenderOptions) -> String {
    let theme = Theme {
        color: options.color,
    };
    let mut out = String::new();

    out.push_str(&header(ir, &theme));
    out.push('\n');
    out.push_str(&overview(ir, &theme));
    out.push('\n');
    if !options.compact {
        out.push_str(&legend(&theme));
        out.push('\n');
    }
    out.push_str(&graph(ir, &theme, options.granularity));
    out.push('\n');
    if !options.compact {
        out.push_str(&details(ir, &theme));
    }

    if !options.compact && !ir.warnings.is_empty() {
        out.push('\n');
        out.push_str(&warnings(ir, &theme));
    }

    if let Some(max_lines) = options.max_lines {
        out = limit_lines(&out, max_lines, &theme);
    }

    out
}

pub fn render_model_fit(ir: &ModelIr, color: bool, terminal_height: usize) -> String {
    let usable_height = terminal_height.saturating_sub(1).max(8);
    for (granularity, compact) in [(2, false), (1, false), (0, false), (0, true)] {
        let output = render_model(
            ir,
            RenderOptions {
                color,
                granularity,
                compact,
                max_lines: None,
            },
        );
        if output.lines().count() <= usable_height {
            return output;
        }
    }

    render_model(
        ir,
        RenderOptions {
            color,
            granularity: 0,
            compact: true,
            max_lines: Some(usable_height),
        },
    )
}

fn header(ir: &ModelIr, theme: &Theme) -> String {
    let title = format!(
        " tviz  {}  {} params ",
        ir.model.name,
        format_count(ir.model.total_params)
    );
    let subtitle = format!("source: {} | schema {}", ir.model.source, ir.schema_version);
    let width = visible_width(&title).max(visible_width(&subtitle)) + 4;
    let mut lines = Vec::new();
    lines.push(format!("╭{}╮", "─".repeat(width)));
    lines.push(format!(
        "│ {}{}{} │",
        theme.paint(&title, Style::Title),
        " ".repeat(width.saturating_sub(visible_width(&title))),
        ""
    ));
    lines.push(format!(
        "│ {}{} │",
        theme.paint(&subtitle, Style::Muted),
        " ".repeat(width.saturating_sub(visible_width(&subtitle)))
    ));
    lines.push(format!("╰{}╯", "─".repeat(width)));
    lines.join("\n")
}

fn overview(ir: &ModelIr, theme: &Theme) -> String {
    let inputs = ir
        .inputs
        .iter()
        .map(|input| format!("{}:{}[{}]", input.name, input.dtype, input.shape.join(",")))
        .collect::<Vec<_>>()
        .join("  ");
    let summary = format!(
        "nodes {}   edges {}   groups {}   trainable {}   inputs {}",
        ir.nodes.len(),
        ir.edges.len(),
        ir.groups.len(),
        format_count(ir.model.trainable_params),
        if inputs.is_empty() { "none" } else { &inputs }
    );
    format!("{}\n", theme.paint(&summary, Style::Muted))
}

fn legend(theme: &Theme) -> String {
    let items = [
        ("conv", Style::Conv),
        ("norm", Style::Norm),
        ("act", Style::Activation),
        ("attention", Style::Attention),
        ("mlp", Style::Mlp),
        ("embedding", Style::Embedding),
        ("output", Style::Output),
        ("custom", Style::Custom),
    ];
    let labels = items
        .iter()
        .map(|(label, style)| theme.paint(&format!(" {label} "), *style))
        .collect::<Vec<_>>()
        .join(" ");
    format!("legend  {labels}\n")
}

fn graph(ir: &ModelIr, theme: &Theme, granularity: u8) -> String {
    let mut out = String::new();
    let nodes = visible_nodes(ir, granularity);
    out.push_str(&theme.paint("architecture", Style::Section));
    out.push('\n');
    out.push_str(&dag_canvas(ir, &nodes, theme, granularity));
    out
}

#[derive(Debug, Clone)]
struct DrawNode<'a> {
    node: &'a Node,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

#[derive(Debug, Clone)]
struct DrawGroup {
    label: String,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

#[derive(Debug, Clone)]
struct VisibleEdge {
    from: String,
    to: String,
    kind: String,
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    ch: char,
    style: Style,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            style: Style::Faint,
        }
    }
}

struct Canvas {
    cells: Vec<Vec<Cell>>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![vec![Cell::default(); width]; height],
        }
    }

    fn put(&mut self, x: usize, y: usize, ch: char, style: Style) {
        if let Some(row) = self.cells.get_mut(y) {
            if let Some(cell) = row.get_mut(x) {
                *cell = Cell { ch, style };
            }
        }
    }

    fn put_merge(&mut self, x: usize, y: usize, ch: char, style: Style) {
        let existing = self
            .cells
            .get(y)
            .and_then(|row| row.get(x))
            .map(|cell| cell.ch)
            .unwrap_or(' ');
        let merged = match (existing, ch) {
            (' ', next) => next,
            ('│', '─') | ('─', '│') | ('┼', _) | (_, '┼') => '┼',
            ('╭', _) | ('╮', _) | ('╰', _) | ('╯', _) => existing,
            ('▼', _) => '▼',
            (same, next) if same == next => same,
            (_, next) => next,
        };
        self.put(x, y, merged, style);
    }

    fn text(&mut self, x: usize, y: usize, text: &str, style: Style) {
        for (offset, ch) in text.chars().enumerate() {
            self.put(x + offset, y, ch, style);
        }
    }

    fn hline(&mut self, x1: usize, x2: usize, y: usize, style: Style) {
        for x in x1.min(x2)..=x1.max(x2) {
            self.put_merge(x, y, '─', style);
        }
    }

    fn vline(&mut self, x: usize, y1: usize, y2: usize, style: Style) {
        for y in y1.min(y2)..=y1.max(y2) {
            self.put_merge(x, y, '│', style);
        }
    }

    fn render(&self, theme: &Theme) -> String {
        let mut out = String::new();
        let last_row = self
            .cells
            .iter()
            .rposition(|row| row.iter().any(|cell| cell.ch != ' '))
            .map(|index| index + 1)
            .unwrap_or(0);
        for row in self.cells.iter().take(last_row) {
            let last = row
                .iter()
                .rposition(|cell| cell.ch != ' ')
                .map(|index| index + 1)
                .unwrap_or(0);
            for cell in row.iter().take(last) {
                out.push_str(&theme.paint(&cell.ch.to_string(), cell.style));
            }
            out.push('\n');
        }
        out
    }
}

fn dag_canvas(ir: &ModelIr, nodes: &[&Node], theme: &Theme, granularity: u8) -> String {
    if nodes.is_empty() {
        return "  no visible nodes\n".to_string();
    }

    let visible_ids = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    let edges = visible_edges(ir, &visible_ids, granularity);
    let ranks = rank_nodes(nodes, &edges);
    let mut layers: BTreeMap<usize, Vec<&Node>> = BTreeMap::new();
    for node in nodes {
        layers
            .entry(*ranks.get(&node.id).unwrap_or(&0))
            .or_default()
            .push(*node);
    }

    let box_w = nodes
        .iter()
        .map(|node| node_width(node))
        .max()
        .unwrap_or(28)
        .clamp(24, 42);
    let box_h = 5;
    let x_gap = 8;
    let y_gap = 5;
    let margin_x = 8;
    let margin_y = 4;
    let max_layer = layers.values().map(Vec::len).max().unwrap_or(1);
    let width = margin_x * 2 + max_layer * box_w + max_layer.saturating_sub(1) * x_gap + 28;
    let height = margin_y * 2 + layers.len() * box_h + layers.len().saturating_sub(1) * y_gap + 4;

    let mut canvas = Canvas::new(width.max(70), height.max(12));
    let mut drawn = BTreeMap::new();
    for (rank, layer) in &layers {
        let layer_width = layer.len() * box_w + layer.len().saturating_sub(1) * x_gap;
        let start_x = margin_x + width.saturating_sub(layer_width) / 2;
        let y = margin_y + rank * (box_h + y_gap);
        for (index, node) in layer.iter().enumerate() {
            let draw = DrawNode {
                node,
                x: start_x + index * (box_w + x_gap),
                y,
                w: box_w,
                h: box_h,
            };
            drawn.insert(node.id.clone(), draw);
        }
    }

    let groups = group_boxes(ir, &drawn);
    for group in &groups {
        draw_group_box(&mut canvas, group);
    }

    for edge in edges {
        if let (Some(from), Some(to)) = (drawn.get(&edge.from), drawn.get(&edge.to)) {
            route_edge(&mut canvas, from, to, &edge.kind);
        }
    }

    for draw in drawn.values() {
        draw_box(&mut canvas, draw);
    }

    canvas.render(theme)
}

fn visible_edges(
    ir: &ModelIr,
    visible_ids: &BTreeSet<String>,
    granularity: u8,
) -> Vec<VisibleEdge> {
    let mut edges = Vec::new();
    let mut seen = BTreeSet::new();
    let mut outgoing = BTreeMap::<String, Vec<&crate::ir::Edge>>::new();

    for edge in &ir.edges {
        if edge.from == edge.to || (edge.kind == "summary" && granularity > 2) {
            continue;
        }

        if edge.kind != "summary" {
            outgoing.entry(edge.from.clone()).or_default().push(edge);
        }

        if visible_ids.contains(&edge.from) && visible_ids.contains(&edge.to) {
            push_visible_edge(&mut edges, &mut seen, &edge.from, &edge.to, &edge.kind);
        }
    }

    for source in visible_ids {
        let mut queue = VecDeque::new();
        let mut visited = BTreeSet::new();
        for edge in outgoing.get(source).into_iter().flatten() {
            if !visible_ids.contains(&edge.to) {
                queue.push_back(edge.to.clone());
            }
        }

        while let Some(hidden) = queue.pop_front() {
            if !visited.insert(hidden.clone()) {
                continue;
            }

            for edge in outgoing.get(&hidden).into_iter().flatten() {
                if edge.to == *source {
                    continue;
                }
                if visible_ids.contains(&edge.to) {
                    push_visible_edge(&mut edges, &mut seen, source, &edge.to, "summary");
                } else {
                    queue.push_back(edge.to.clone());
                }
            }
        }
    }

    edges
}

fn push_visible_edge(
    edges: &mut Vec<VisibleEdge>,
    seen: &mut BTreeSet<(String, String, String)>,
    from: &str,
    to: &str,
    kind: &str,
) {
    if from == to {
        return;
    }

    let key = (from.to_string(), to.to_string(), kind.to_string());
    if seen.insert(key) {
        edges.push(VisibleEdge {
            from: from.to_string(),
            to: to.to_string(),
            kind: kind.to_string(),
        });
    }
}

fn group_boxes(ir: &ModelIr, drawn: &BTreeMap<String, DrawNode<'_>>) -> Vec<DrawGroup> {
    let mut groups = Vec::new();
    for group in &ir.groups {
        let children = group
            .children
            .iter()
            .filter_map(|child| drawn.get(child))
            .collect::<Vec<_>>();
        if children.len() < 2 {
            continue;
        }

        let min_x = children.iter().map(|child| child.x).min().unwrap_or(0);
        let min_y = children.iter().map(|child| child.y).min().unwrap_or(0);
        let max_x = children
            .iter()
            .map(|child| child.x + child.w)
            .max()
            .unwrap_or(0);
        let max_y = children
            .iter()
            .map(|child| child.y + child.h)
            .max()
            .unwrap_or(0);

        let x = min_x.saturating_sub(4);
        let y = min_y.saturating_sub(2);
        let w = max_x.saturating_sub(x).saturating_add(10);
        let h = max_y.saturating_sub(y).saturating_add(3);
        groups.push(DrawGroup {
            label: group_label(&group.label, group.repeated),
            x,
            y,
            w,
            h,
        });
    }

    groups.sort_by_key(|group| std::cmp::Reverse(group.w * group.h));
    groups
}

fn draw_group_box(canvas: &mut Canvas, group: &DrawGroup) {
    if group.w < 8 || group.h < 4 {
        return;
    }

    let x = group.x;
    let y = group.y;
    let right = group.x + group.w - 1;
    let bottom = group.y + group.h - 1;
    let style = Style::Group;

    canvas.put(x, y, '┌', style);
    canvas.put(right, y, '┐', style);
    canvas.put(x, bottom, '└', style);
    canvas.put(right, bottom, '┘', style);
    canvas.hline(x + 1, right.saturating_sub(1), y, style);
    canvas.hline(x + 1, right.saturating_sub(1), bottom, style);
    canvas.vline(x, y + 1, bottom.saturating_sub(1), style);
    canvas.vline(right, y + 1, bottom.saturating_sub(1), style);

    let label = format!(" {} ", truncate(&group.label, group.w.saturating_sub(4)));
    canvas.text(x + 2, y, &label, style);
}

fn rank_nodes(nodes: &[&Node], edges: &[VisibleEdge]) -> BTreeMap<String, usize> {
    let ids = nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut incoming_count = nodes
        .iter()
        .map(|node| (node.id.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<String, Vec<String>>::new();

    for edge in edges {
        if !ids.contains(edge.from.as_str()) || !ids.contains(edge.to.as_str()) {
            continue;
        }
        *incoming_count.entry(edge.to.clone()).or_default() += 1;
        outgoing
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
    }

    let order = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut queue = incoming_count
        .iter()
        .filter(|(_id, count)| **count == 0)
        .map(|(id, _count)| id.clone())
        .collect::<Vec<_>>();
    queue.sort_by_key(|id| order.get(id).copied().unwrap_or(usize::MAX));
    let mut queue = VecDeque::from(queue);
    let mut ranks = nodes
        .iter()
        .map(|node| (node.id.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();

    while let Some(id) = queue.pop_front() {
        seen.insert(id.clone());
        let source_rank = *ranks.get(&id).unwrap_or(&0);
        for target in outgoing.get(&id).into_iter().flatten() {
            let next_rank = source_rank + 1;
            if ranks.get(target).copied().unwrap_or(0) < next_rank {
                ranks.insert(target.clone(), next_rank);
            }
            if let Some(count) = incoming_count.get_mut(target) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    queue.push_back(target.clone());
                }
            }
        }
    }

    if seen.len() != nodes.len() {
        for (index, node) in nodes.iter().enumerate() {
            ranks.entry(node.id.clone()).or_insert(index);
        }
    }

    if edges.is_empty() {
        for (index, node) in nodes.iter().enumerate() {
            ranks.insert(node.id.clone(), index);
        }
    }

    ranks
}

fn node_width(node: &Node) -> usize {
    let label = display_label(node);
    let badge = node_badge(node);
    let attrs = important_attrs(node);
    visible_width(&label)
        .max(visible_width(&badge))
        .max(visible_width(&attrs))
        .saturating_add(4)
}

fn draw_box(canvas: &mut Canvas, draw: &DrawNode<'_>) {
    let style = style_for(draw.node);
    let x = draw.x;
    let y = draw.y;
    let w = draw.w;
    let h = draw.h;

    canvas.put(x, y, '╭', style);
    canvas.put(x + w - 1, y, '╮', style);
    canvas.put(x, y + h - 1, '╰', style);
    canvas.put(x + w - 1, y + h - 1, '╯', style);
    canvas.hline(x + 1, x + w - 2, y, style);
    canvas.hline(x + 1, x + w - 2, y + h - 1, style);
    canvas.vline(x, y + 1, y + h - 2, style);
    canvas.vline(x + w - 1, y + 1, y + h - 2, style);

    canvas.text(
        x + 2,
        y + 1,
        &truncate(&display_label(draw.node), w - 4),
        style,
    );
    canvas.text(
        x + 2,
        y + 2,
        &truncate(&node_badge(draw.node), w - 4),
        Style::Muted,
    );
    canvas.text(
        x + 2,
        y + 3,
        &truncate(&important_attrs(draw.node), w - 4),
        Style::Faint,
    );
}

fn route_edge(canvas: &mut Canvas, from: &DrawNode<'_>, to: &DrawNode<'_>, kind: &str) {
    let style = edge_style(kind);
    let sx = from.x + from.w / 2;
    let sy = from.y + from.h;
    let tx = to.x + to.w / 2;
    let ty = to.y.saturating_sub(1);

    if ty <= sy {
        let lane = from.x.max(to.x) + from.w + 2;
        canvas.hline(sx, lane, sy, style);
        canvas.vline(lane, sy, ty, style);
        canvas.hline(lane, tx, ty, style);
        canvas.put(tx, ty, '▼', style);
        return;
    }

    let mid = sy + (ty - sy) / 2;
    if matches!(kind, "skip" | "residual") || ty.saturating_sub(sy) > 8 {
        let lane = from.x.max(to.x) + from.w + 2;
        canvas.hline(sx, lane, sy, style);
        canvas.vline(lane, sy, ty, style);
        canvas.hline(lane, tx, ty, style);
    } else {
        canvas.vline(sx, sy, mid, style);
        canvas.hline(sx, tx, mid, style);
        canvas.vline(tx, mid, ty, style);
    }
    canvas.put(tx, ty, '▼', style);
}

fn display_label(node: &Node) -> String {
    if node.repeated > 1 {
        format!("{} x{}", node.label, node.repeated)
    } else {
        node.label.clone()
    }
}

fn node_badge(node: &Node) -> String {
    let shape = node
        .output_shapes
        .first()
        .cloned()
        .unwrap_or_else(|| "?".to_string());
    format!("{} · {} · {}", node.kind, format_count(node.params), shape)
}

fn edge_style(kind: &str) -> Style {
    match kind {
        "skip" | "residual" => Style::SkipEdge,
        "branch" | "join" => Style::BranchEdge,
        _ => Style::Edge,
    }
}

fn details(ir: &ModelIr, theme: &Theme) -> String {
    let mut out = String::new();
    out.push_str(&theme.paint("reference", Style::Section));
    out.push('\n');
    let rows = ir
        .nodes
        .iter()
        .take(12)
        .map(|node| {
            format!(
                "  {:<28} {:<18} {:>9}  {}",
                truncate(&node.module_path, 28),
                truncate(&node.kind, 18),
                format_count(node.params),
                node.output_shapes
                    .first()
                    .map(String::as_str)
                    .unwrap_or("?")
            )
        })
        .collect::<Vec<_>>();
    out.push_str(&theme.paint(
        "  module path                  kind                  params  output",
        Style::Muted,
    ));
    out.push('\n');
    out.push_str(&rows.join("\n"));
    out.push('\n');
    out
}

fn warnings(ir: &ModelIr, theme: &Theme) -> String {
    let mut out = String::new();
    out.push_str(&theme.paint("warnings", Style::Warning));
    out.push('\n');
    for warning in &ir.warnings {
        out.push_str(&format!("  {}\n", theme.paint(warning, Style::Warning)));
    }
    out
}

fn visible_nodes(ir: &ModelIr, granularity: u8) -> Vec<&Node> {
    if granularity >= 4 {
        return ir.nodes.iter().collect();
    }

    let base_depth = ir
        .nodes
        .iter()
        .filter(|node| node.style != "input" && node.style != "output")
        .map(|node| node.depth)
        .min()
        .unwrap_or(0);
    let max_depth = base_depth + u64::from(granularity);

    ir.nodes
        .iter()
        .filter(|node| node.depth <= max_depth || node.style == "input" || node.style == "output")
        .collect()
}

fn group_label(label: &str, repeated: u64) -> String {
    if repeated > 1 {
        format!("{label} x{repeated}")
    } else {
        label.to_string()
    }
}

fn important_attrs(node: &Node) -> String {
    let mut attrs = node
        .attributes
        .iter()
        .take(3)
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>();
    if attrs.is_empty() {
        attrs.push(node.module_path.clone());
    }
    truncate(&attrs.join("  "), 64)
}

fn style_for(node: &Node) -> Style {
    match node.style.as_str() {
        "conv" => Style::Conv,
        "norm" => Style::Norm,
        "activation" => Style::Activation,
        "attention" => Style::Attention,
        "mlp" => Style::Mlp,
        "embedding" => Style::Embedding,
        "pooling" => Style::Pooling,
        "output" => Style::Output,
        "input" => Style::Input,
        _ => Style::Custom,
    }
}

fn format_count(value: u64) -> String {
    if value >= 1_000_000_000 {
        format!("{:.1}B", value as f64 / 1_000_000_000.0)
    } else if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn truncate(value: &str, max: usize) -> String {
    if visible_width(value) <= max {
        value.to_string()
    } else if max <= 1 {
        "…".to_string()
    } else {
        format!("{}…", value.chars().take(max - 1).collect::<String>())
    }
}

fn visible_width(value: &str) -> usize {
    value.chars().count()
}

fn limit_lines(output: &str, max_lines: usize, theme: &Theme) -> String {
    let mut lines = output.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        return output.to_string();
    }

    let notice = theme.paint(
        "… clipped to fit terminal; use --print for scrollback",
        Style::Warning,
    );
    if max_lines == 0 {
        return notice;
    }
    lines.truncate(max_lines.saturating_sub(1));
    let mut clipped = lines.join("\n");
    if !clipped.is_empty() {
        clipped.push('\n');
    }
    clipped.push_str(&notice);
    clipped.push('\n');
    clipped
}

#[derive(Debug, Clone, Copy)]
enum Style {
    Title,
    Section,
    Muted,
    Faint,
    Warning,
    Edge,
    SkipEdge,
    BranchEdge,
    Group,
    Input,
    Conv,
    Norm,
    Activation,
    Attention,
    Mlp,
    Embedding,
    Pooling,
    Output,
    Custom,
}

struct Theme {
    color: bool,
}

impl Theme {
    fn paint(&self, text: &str, style: Style) -> String {
        if !self.color {
            return text.to_string();
        }
        let code = match style {
            Style::Title => "1;38;5;117",
            Style::Section => "1;38;5;229",
            Style::Muted => "38;5;103",
            Style::Faint => "38;5;245",
            Style::Warning => "38;5;214",
            Style::Edge => "38;5;75",
            Style::SkipEdge => "38;5;177",
            Style::BranchEdge => "38;5;211",
            Style::Group => "38;5;220",
            Style::Input => "38;5;81",
            Style::Conv => "38;5;79",
            Style::Norm => "38;5;147",
            Style::Activation => "38;5;221",
            Style::Attention => "38;5;219",
            Style::Mlp => "38;5;208",
            Style::Embedding => "38;5;111",
            Style::Pooling => "38;5;115",
            Style::Output => "38;5;120",
            Style::Custom => "38;5;252",
        };
        format!("\x1b[{code}m{text}\x1b[0m")
    }
}
