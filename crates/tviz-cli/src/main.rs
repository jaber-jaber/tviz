use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tviz_core::export::{export_dot, export_svg};
use tviz_core::json::parse_model_ir;
use tviz_core::render::{RenderOptions, render_model, render_model_fit};

const EMBEDDED_PROBE: &str = include_str!("../../../python/tviz_probe/probe.py");
const EMBEDDED_HF_PROBE: &str = include_str!("../../../python/tviz_probe/hf_probe.py");

#[derive(Debug, Clone)]
struct Cli {
    input: String,
    factory: Option<String>,
    python: String,
    probe: Option<String>,
    revision: String,
    input_spec: Option<String>,
    trace: String,
    granularity: u8,
    no_color: bool,
    json: bool,
    export: Option<String>,
    view: ViewMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Auto,
    Print,
    Screen,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tviz: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", help());
        return Ok(());
    }
    let cli = parse_args(args)?;
    let json = load_ir_json(&cli)?;

    let ir = parse_model_ir(&json)?;

    if let Some(path) = &cli.export {
        write_export(path, &json, &ir)?;
    }

    if cli.json {
        println!("{json}");
        return Ok(());
    }

    match resolved_view(cli.view) {
        ViewMode::Screen => show_screen(&ir, !cli.no_color)?,
        _ => {
            let output = render_model(
                &ir,
                RenderOptions {
                    color: !cli.no_color,
                    granularity: cli.granularity,
                    compact: false,
                    max_lines: None,
                },
            );
            print!("{output}");
        }
    }
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Cli, String> {
    if args.is_empty() {
        return Err(help());
    }

    let mut input = None;
    let mut factory = None;
    let mut python = env::var("TVIZ_PYTHON").unwrap_or_else(|_| "python".to_string());
    let mut probe = None;
    let mut revision = "main".to_string();
    let mut input_spec = None;
    let mut trace = "hook".to_string();
    let mut granularity = 4;
    let mut no_color = false;
    let mut json = false;
    let mut export = None;
    let mut view = ViewMode::Auto;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--factory" => {
                index += 1;
                factory = Some(value_after(&args, index, "--factory")?);
            }
            "--python" => {
                index += 1;
                python = value_after(&args, index, "--python")?;
            }
            "--probe" => {
                index += 1;
                probe = Some(value_after(&args, index, "--probe")?);
            }
            "--revision" => {
                index += 1;
                revision = value_after(&args, index, "--revision")?;
            }
            "--input" => {
                index += 1;
                input_spec = Some(value_after(&args, index, "--input")?);
            }
            "--trace" => {
                index += 1;
                trace = value_after(&args, index, "--trace")?;
            }
            "--granularity" | "-g" => {
                index += 1;
                let raw = value_after(&args, index, "--granularity")?;
                granularity = raw
                    .parse::<u8>()
                    .map_err(|_| format!("invalid granularity {raw:?}; expected 0-4"))?;
                if granularity > 4 {
                    return Err(format!("invalid granularity {granularity}; expected 0-4"));
                }
            }
            "--no-color" => no_color = true,
            "--json" => json = true,
            "--export" => {
                index += 1;
                export = Some(value_after(&args, index, "--export")?);
            }
            "--screen" | "--fullscreen" => view = ViewMode::Screen,
            "--print" => view = ViewMode::Print,
            value if value.starts_with('-') => {
                return Err(format!("unknown option {value}\n\n{}", help()));
            }
            value => {
                if input.is_some() {
                    return Err(format!("unexpected extra argument {value:?}\n\n{}", help()));
                }
                input = Some(value.to_string());
            }
        }
        index += 1;
    }

    Ok(Cli {
        input: input.ok_or_else(help)?,
        factory,
        python,
        probe,
        revision,
        input_spec,
        trace,
        granularity,
        no_color,
        json,
        export,
        view,
    })
}

fn value_after(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn load_ir_json(cli: &Cli) -> Result<String, String> {
    let path = Path::new(&cli.input);
    if path.extension() == Some(OsStr::new("json")) {
        return fs::read_to_string(path)
            .map_err(|error| format!("failed to read IR JSON {}: {error}", path.display()));
    }

    if looks_like_hf(&cli.input) {
        return run_hf_probe(cli);
    }

    run_python_probe(cli)
}

fn run_hf_probe(cli: &Cli) -> Result<String, String> {
    let mut command = Command::new(&cli.python);
    command
        .arg("-c")
        .arg(EMBEDDED_HF_PROBE)
        .arg(cli.input.as_str())
        .arg("--revision")
        .arg(cli.revision.as_str())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = command
        .output()
        .map_err(|error| format!("failed to start HF probe: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("HF probe failed:\n{stderr}"));
    }

    String::from_utf8(output.stdout)
        .map_err(|error| format!("HF probe returned invalid UTF-8: {error}"))
}

fn run_python_probe(cli: &Cli) -> Result<String, String> {
    let mut command = Command::new(&cli.python);

    if let Some(probe) = &cli.probe {
        command.arg(probe);
    } else {
        command.arg("-c").arg(EMBEDDED_PROBE);
    }

    command
        .arg(cli.input.as_str())
        .arg("--trace")
        .arg(cli.trace.as_str())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(factory) = &cli.factory {
        command.arg("--factory").arg(factory);
    }

    if let Some(input_spec) = &cli.input_spec {
        command.arg("--input").arg(input_spec);
    }

    let output = command
        .output()
        .map_err(|error| format!("failed to start Python probe: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python probe failed:\n{stderr}"));
    }

    String::from_utf8(output.stdout)
        .map_err(|error| format!("probe returned invalid UTF-8: {error}"))
}

fn resolved_view(view: ViewMode) -> ViewMode {
    match view {
        ViewMode::Auto => ViewMode::Print,
        other => other,
    }
}

fn write_export(path: &str, json: &str, ir: &tviz_core::ir::ModelIr) -> Result<(), String> {
    let output = match Path::new(path).extension().and_then(OsStr::to_str) {
        Some("json") => json.to_string(),
        Some("dot") => export_dot(ir),
        Some("svg") => export_svg(ir),
        Some(other) => {
            return Err(format!(
                "unsupported export extension {other:?}; use .json, .dot, or .svg"
            ));
        }
        None => return Err("export path needs an extension: .json, .dot, or .svg".to_string()),
    };

    fs::write(path, output).map_err(|error| format!("failed to write export {path}: {error}"))
}

fn show_screen(ir: &tviz_core::ir::ModelIr, color: bool) -> Result<(), String> {
    let height = terminal_height().unwrap_or(32);
    let output = render_model_fit(ir, color, height);
    let mut stdout = io::stdout();

    write!(stdout, "\x1b[?1049h\x1b[2J\x1b[H{output}")
        .map_err(|error| format!("failed to draw screen: {error}"))?;
    writeln!(
        stdout,
        "\n{}",
        if color {
            "\x1b[38;5;245mpress Enter to close · use --print for scrollback\x1b[0m"
        } else {
            "press Enter to close · use --print for scrollback"
        }
    )
    .map_err(|error| format!("failed to draw screen footer: {error}"))?;
    stdout
        .flush()
        .map_err(|error| format!("failed to flush screen: {error}"))?;

    let mut hold = String::new();
    let _ = io::stdin().read_line(&mut hold);

    write!(io::stdout(), "\x1b[?1049l")
        .map_err(|error| format!("failed to restore terminal: {error}"))?;
    Ok(())
}

fn terminal_height() -> Option<usize> {
    if let Ok(lines) = env::var("LINES") {
        if let Ok(value) = lines.parse::<usize>() {
            return Some(value);
        }
    }

    let output = Command::new("stty").arg("size").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    raw.split_whitespace().next()?.parse::<usize>().ok()
}

fn looks_like_hf(input: &str) -> bool {
    if PathBuf::from(input).exists() {
        return false;
    }
    input.starts_with("https://huggingface.co/")
        || input.starts_with("huggingface.co/")
        || input.matches('/').count() == 1
        || (!input.contains('/') && !input.ends_with(".py") && !input.ends_with(".json"))
}

fn help() -> String {
    "usage: tviz <MODEL.py|MODEL_IR.json|HF_REPO> [--factory NAME] [--input SPEC] [--trace structure|hook] [--revision REV] [--granularity 0-4] [--screen] [--export PATH] [--json] [--no-color]\n\nexamples:\n  tviz examples/models.py\n  tviz examples/models.py --factory messy_research_model\n  tviz examples/models.py --factory tiny_convnet --input 'float32[1,3,64,64]'\n  tviz Qwen/Qwen3.5-0.8B\n  tviz https://huggingface.co/google/vit-base-patch16-224\n  tviz fixtures/tiny_convnet.json --export architecture.dot\n\nnotes:\n  default output is scrollback-friendly print mode with full execution detail\n  use --granularity 0, 1, or 2 for a more compact print\n  --screen uses an alternate-screen fit view and waits for Enter\n  --export supports .json, .dot, and .svg\n  --revision selects a Hugging Face branch, tag, or commit\n  set HF_TOKEN for gated/private HF repos\n  set TVIZ_PYTHON or pass --python to choose the PyTorch/Python environment\n".to_string()
}
