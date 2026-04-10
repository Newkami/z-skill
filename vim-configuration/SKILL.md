---
name: vim-configuration
description: Generate ready-to-paste .vimrc configuration for macOS personal Vim setup. Use this skill whenever the user wants to configure Vim — including syntax highlighting, auto-format (clang-format, black/Python), format keybindings, insert-mode word movement with Option+arrow keys, or any .vimrc setup task. Trigger for phrases like "配置vim", "vimrc", "vim 快捷键", "vim format", "vim 自动格式化", "vim 光标移动", or any request to set up, customize, or configure Vim.
---

# Vim Configuration (macOS Personal Setup)

Your job is to output a ready-to-paste `.vimrc` configuration snippet. Structure the response as:
1. A short prerequisites block (install commands)
2. The `.vimrc` snippet itself
3. How to apply it
4. An Option+arrow troubleshooting note

---

## Prerequisites

Output this block first:

```bash
# Install formatters (run once)
brew install clang-format      # C / C++ / ObjC / Proto
pip install black              # Python
```

---

## The .vimrc Snippet

Output this complete, self-contained block. Adapt the comments to match what the user asked for.

```vim
" ─── Display & syntax ─────────────────────────────────────────────────────────
" syntax enable respects any colors already set (preferred over syntax on)
syntax enable
" Auto-detect filetype, load matching syntax rules and indent settings
filetype plugin indent on
" Dark background improves contrast for most color schemes
set background=dark
" Built-in theme that looks good in terminal without plugins
colorscheme desert

" ─── Auto-format  (\f  i.e. <leader>f in Normal mode) ───────────────────────
" Formats the current file in place, restoring cursor position.
" Supported filetypes: C / C++ / ObjC / Proto (clang-format)  ·  Python (black)
" Prerequisites: brew install clang-format  &&  pip install black
" Default leader is \  so the shortcut is \f.  To change leader add:
"   let mapleader = ","   (or <Space>, etc.) before this block.
function! AutoFormat()
  let l:view = winsaveview()

  if &filetype =~# '\v^(c|cpp|objc|objcpp|proto)$'
    let l:result = system('clang-format', join(getline(1, '$'), "\n"))
  elseif &filetype ==# 'python'
    let l:result = system('black -q -', join(getline(1, '$'), "\n"))
  else
    echo '[fmt] no formatter configured for filetype: ' . &filetype
    return
  endif

  if v:shell_error
    echohl ErrorMsg
    echo '[fmt] formatter exited with error — buffer unchanged'
    echohl None
    return
  endif

  " Replace buffer contents without touching the undo history entry
  let l:lines = split(l:result, "\n", 1)
  " Trim the trailing empty string that split() adds for a final newline
  if !empty(l:lines) && l:lines[-1] ==# ''
    call remove(l:lines, -1)
  endif
  silent! 1,$delete _
  call setline(1, l:lines)
  call winrestview(l:view)
endfunction

nnoremap <leader>f :call AutoFormat()<CR>

" ─── Insert mode: line-boundary shortcuts (Ctrl+A / Ctrl+E) ─────────────────
" Mirrors shell/readline behaviour: C-a = start of line, C-e = end of line
inoremap <C-a> <C-o>0
inoremap <C-e> <C-o>$

" ─── Insert mode: word-level cursor movement (Option + ← / →) ────────────────
" Ghostty sends xterm modifier-3 sequences by default.
" Terminal.app sends modifier-9 sequences by default.
" Both variants are mapped here so the same .vimrc works in either terminal.
" If neither works see the Troubleshooting section below.
if !has('gui_running')
  " Option+Left  (Ghostty default: mod-3 / Terminal.app default: mod-9)
  inoremap <Esc>[1;3D <C-o>b
  inoremap <Esc>[1;9D <C-o>b
  " Option+Right
  inoremap <Esc>[1;3C <C-o>w
  inoremap <Esc>[1;9C <C-o>w
  " Terminal.app with "Use Option as Meta key" enabled (sends ESC + letter)
  inoremap <Esc>b <C-o>b
  inoremap <Esc>f <C-o>w
endif
```

---

## How to apply

Tell the user:

1. Open (or create) `~/.vimrc`
2. Paste the snippet above
3. Either restart Vim or run `:source ~/.vimrc` inside a running session
4. Test: open a `.c` or `.py` file, make a change, press `\f` in normal mode

---

## Option+arrow troubleshooting

If the cursor movement doesn't work, the terminal is sending a sequence not covered above. To find the actual sequence:

1. Run `cat -v` in a fresh terminal
2. Press `Option+Left` — you'll see something like `^[[1;3D` or `^[b`
3. Map that exact sequence:
   - `^[[1;3D`  →  add `inoremap <Esc>[1;3D <C-o>b` (already included)
   - `^[b`      →  add `inoremap <Esc>b <C-o>b` (already included)
   - `^[[1;5D`  →  add `inoremap <Esc>[1;5D <C-o>b` (not included — paste and adjust)

Note on `\f` mapping: `\` is Vim's default leader key and has no built-in meaning by itself, so `\f` is conflict-free. If you change leader (e.g. `let mapleader = ","`) the shortcut becomes `,f` instead.
