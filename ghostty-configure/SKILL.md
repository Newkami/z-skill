---
name: ghostty-configure
description: Configure Ghostty terminal by writing the embedded canonical config to ~/.config/ghostty/config. Use this skill whenever the user wants to set up, reset, or reapply their Ghostty configuration. Trigger for phrases like "配置ghostty", "ghostty config", "ghostty 配置", "ghostty 终端配置", "apply ghostty config", or any request to configure or reconfigure Ghostty.
---

# Ghostty Terminal Configuration

Your job is to write the canonical Ghostty config (embedded below) to `~/.config/ghostty/config`. Follow these steps exactly.

---

## Step 1 — Ensure the config directory exists

```bash
mkdir -p ~/.config/ghostty
```

---

## Step 2 — Write the config

Read the file `{base_dir}/config` (provided as "Base directory for this skill" at the top of the skill invocation), then write its content verbatim to `~/.config/ghostty/config`, overwriting any existing file.

---

## Step 3 — Verify

Read back `~/.config/ghostty/config` and confirm it was written correctly. Report a brief summary of key settings applied.

---

## Step 4 — Reload instructions

Tell the user how to make the config take effect:

- **Hot reload** (Ghostty already running): press `Cmd+Shift+,`
- **Full restart**: quit and reopen Ghostty

---

## Notes

- To view all available options: `ghostty +show-config --default --docs`
- If the font "Maple Mono NF CN" is not installed, Ghostty will fall back to its default font — install it from Nerd Fonts if needed.
