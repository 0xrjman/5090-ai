#!/usr/bin/env python3
"""
config_loader.py — Declarative config loader for local-ai.

Reads YAML configs from configs/ and provides:
  - list_configs()     → list of active (non-deprecated) configs
  - load_config()      → load a single config by internal_name
  - render_compose_env() → expand config into KEY=VALUE lines for compose/.env
  - CLI interface: --list, --load, --render-env, --write-compose-env

Usage:
  python3 lib/config_loader.py --list                    # JSON list of configs
  python3 lib/config_loader.py --list --format shell      # shell-friendly list
  python3 lib/config_loader.py --load vision-mtp          # JSON config detail
  python3 lib/config_loader.py --render-env vision-mtp    # KEY=VALUE export lines
  python3 lib/config_loader.py --write-compose-env vision-mtp  # full compose/.env content
"""

import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # Fallback: use a minimal YAML parser if pyyaml is not installed.
    # For our simple schema, we can parse with a basic approach.
    yaml = None


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
GENESIS_FLAGS = [
    "prealloc_v2", "p5b", "p67", "pn8", "pn34", "p82", "p98", "pn59", "pn54", "pn32",
]


def _parse_yaml_simple(text):
    """Minimal YAML parser for our flat config schema (no pyyaml needed)."""
    result = {}
    current_section = None
    for line in text.split("\n"):
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key (no leading whitespace)
        if not line[0].isspace():
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip()
                if val:
                    result[key] = _parse_value(val)
                    current_section = None
                else:
                    result[key] = {}
                    current_section = key
            continue

        # Nested key (indented)
        if current_section and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val:
                result[current_section][key] = _parse_value(val)

    return result


def _parse_value(val):
    """Parse a YAML scalar value."""
    if val.startswith("[") and val.endswith("]"):
        # Inline list
        items = val[1:-1].split(",")
        return [_parse_value(i.strip()) for i in items if i.strip()]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    # Strip quotes
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def _load_yaml(path):
    """Load a YAML file, using pyyaml if available, else simple parser."""
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text)
    return _parse_yaml_simple(text)


def list_configs(configs_dir=None, include_deprecated=False):
    """List all config files, optionally excluding deprecated ones."""
    if configs_dir is None:
        configs_dir = CONFIGS_DIR

    configs = []
    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        cfg = _load_yaml(yaml_file)
        deprecated = cfg.get("deprecated", False)
        if not include_deprecated and deprecated:
            continue
        cfg["_file"] = str(yaml_file.name)
        configs.append(cfg)
    return configs


def load_config(internal_name, configs_dir=None):
    """Load a single config by its internal_name (ENGINE value)."""
    configs = list_configs(configs_dir, include_deprecated=True)
    for cfg in configs:
        if cfg.get("internal_name") == internal_name:
            return cfg
    return None


def render_env_vars(config):
    """Expand config into a dict of environment variables for compose."""
    env = {}

    # Core env vars
    env["CONTAINER_NAME"] = config["container_name"]
    env["MODEL_SUBDIR"] = config["model"]["weights_subdir"]
    env["QUANT_MODE"] = config["model"]["format"]
    env["MODALITY"] = config["env"].get("MODALITY", "vision")
    env["SPEC_CONFIG"] = config["env"].get("SPEC_CONFIG", "")
    env["CHAT_TEMPLATE_PATH"] = config["model"].get("chat_template", "")
    env["MAX_MODEL_LEN"] = config["env"].get("MAX_MODEL_LEN", str(config["params"]["context"]))
    env["GPU_MEMORY_UTIL"] = config["env"].get("GPU_MEMORY_UTIL", "0.94")
    env["MAX_NUM_SEQS"] = config["env"].get("MAX_NUM_SEQS", "8")
    env["MAX_NUM_BATCHED"] = config["env"].get("MAX_NUM_BATCHED", "4096")
    env["KV_CACHE_DTYPE"] = config["params"].get("kv_cache", "fp8_e4m3")

    # Genesis flags
    genesis = config.get("genesis", {})
    for flag in GENESIS_FLAGS:
        env[f"GENESIS_{flag.upper()}"] = int(genesis.get(flag, 0))

    return env


def render_compose_env(config, model_dir=None):
    """Render full compose/.env content for a config."""
    env = render_env_vars(config)
    env["MODEL_DIR"] = model_dir or os.environ.get("MODEL_DIR", str(ROOT_DIR / "models"))

    lines = [
        "# Generated by local-ai.sh — do not edit manually",
    ]
    for key, val in env.items():
        lines.append(f"{key}={val}")
    return "\n".join(lines) + "\n"


def render_export_lines(config):
    """Render shell export lines for eval."""
    env = render_env_vars(config)
    lines = []
    for key, val in env.items():
        # Quote values that might contain spaces/special chars
        if " " in str(val) or "'" in str(val):
            lines.append(f"export {key}='{val}'")
        else:
            lines.append(f"export {key}={val}")
    return "\n".join(lines)


def format_tui_menu(configs_dir=None):
    """Format configs for TUI menu output (internal_name|name|description|is_production)."""
    configs = list_configs(configs_dir, include_deprecated=True)
    for cfg in configs:
        internal = cfg["internal_name"]
        name = cfg.get("name", internal)
        desc = cfg.get("description", "")
        deprecated = cfg.get("deprecated", False)
        production = "★" if not deprecated and "production" in cfg.get("labels", []) else ""
        print(f"{internal}|{name}|{desc}|{production}|{deprecated}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="local-ai config loader")
    parser.add_argument("--list", action="store_true", help="List all configs")
    parser.add_argument("--load", type=str, help="Load config by internal_name")
    parser.add_argument("--render-env", type=str, help="Render shell export lines")
    parser.add_argument("--write-compose-env", type=str, help="Write compose/.env content")
    parser.add_argument("--format", choices=["json", "shell"], default="json",
                        help="Output format for --list")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model directory for compose env")
    parser.add_argument("--label", action="store_true", help="Output config labels (for TUI)")
    parser.add_argument("--tui-list", action="store_true", help="TUI menu format: internal|name|desc|★|deprecated")
    args = parser.parse_args()

    if args.tui_list:
        format_tui_menu()
        return

    if args.list:
        configs = list_configs(include_deprecated=False)
        if args.format == "shell":
            for cfg in configs:
                print(f"{cfg['internal_name']}|{cfg['name']}")
        else:
            print(json.dumps(configs, indent=2, default=str))

    elif args.load:
        cfg = load_config(args.load)
        if cfg:
            print(json.dumps(cfg, indent=2, default=str))
        else:
            print(f"Config not found: {args.load}", file=sys.stderr)
            sys.exit(1)

    elif args.render_env:
        cfg = load_config(args.render_env)
        if cfg:
            print(render_export_lines(cfg))
        else:
            print(f"Config not found: {args.render_env}", file=sys.stderr)
            sys.exit(1)

    elif args.write_compose_env:
        cfg = load_config(args.write_compose_env)
        if cfg:
            print(render_compose_env(cfg, args.model_dir))
        else:
            print(f"Config not found: {args.write_compose_env}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()