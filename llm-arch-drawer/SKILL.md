---
name: llm-arch
description: Generate a detailed computational dataflow diagram as a draw.io XML file
  for any LLM model by reading its Python implementation. ALWAYS trigger when user says
  "draw architecture", "generate diagram", "visualize model", "draw dataflow", or references
  a model .py file with "architecture/diagram/dataflow". Also trigger for "/llm-arch".
  Reads actual model source code to extract class hierarchy and forward() data flow.
  Output is a .drawio XML file showing tensors as dashed-border rounded rectangles,
  operations as solid-border nodes, and directed edges with tensor shape labels.
---

## When to Use

Trigger on any of:
- `/llm-arch <path>`
- "draw [model name] architecture"
- "visualize [model] dataflow"
- "generate diagram from [file.py]"
- "draw architecture from vllm/model_executor/models/..."

## How to Invoke

```
/llm-arch vllm/model_executor/models/qwen3_moe.py
draw qwen3_moe architecture diagram
generate Mixtral dataflow diagram from vllm/model_executor/models/mixtral.py
```

## Workflow

1. **Extract model file path** from user message. If not specified, ask the user.
2. **Ask the user for the output directory** before generating. Say: "Where should I save the diagram? (press Enter for current directory, or paste an absolute path)". Use their answer as `{OUTPUT_DIR}`. Default to the current working directory if they skip.
3. **Launch Discovery Agent** (general-purpose) with Agent Template A below.
   Pass it: `{MODEL_FILE_PATH}`.
   → The agent reads the model code and returns a structured **COMPONENT INVENTORY** as text.
4. **Present the inventory to the user.** Say:
   > Here is everything I found in the model. Does this look complete?
   > Any components to skip or rename? Reply "ok" to proceed.
5. **Launch Generation Agent** (general-purpose) with Agent Template B below.
   Pass it: `{MODEL_FILE_PATH}`, `{OUTPUT_DIR}/{MODEL_NAME}-architecture.drawio`, and the **full inventory text** from step 3, plus any user modifications (skips / renames).
6. Report the output file path to the user.

---

## Agent Template A — Discovery

When launching the Discovery Agent, use this prompt (fill in `{MODEL_FILE_PATH}`):

```
You are a model analysis agent. Your task is to read a PyTorch model implementation
and output a COMPONENT INVENTORY — a structured text list of every component found.
Do NOT generate any draw.io XML. Output only the inventory.

## Input File
{MODEL_FILE_PATH}

## Step 1: Read the Model File

Read the Python file completely. Extract:

### A) All nn.Module Classes
For each class that extends nn.Module:
- Record the class name
- List every `self.X = SomeClass(...)` assignment in `__init__`
- Note `make_layers()`, `nn.ModuleList([...] * N)` (these become ×N groups)
- Note ALL conditional assignments:
    if ...: self.X = A(...) else: self.X = B(...)
  → Record BOTH branches. Never omit a branch.

### B) Forward() Call Sequences
For each class, trace `forward()`:
- What does each `self.X(...)` call produce?
- Where are split operations (`x.split`, `x.chunk`)?
- Where are residual adds (`out = x + residual`)?
- What is the execution order?

### C) Config Parameters
Extract concrete values for the legend:
- hidden_size / d_model
- num_attention_heads / num_heads
- num_key_value_heads
- head_dim
- intermediate_size / d_ff
- moe_intermediate_size / d_ff_moe (if MoE)
- vocab_size
- num_hidden_layers
- num_experts / num_experts_per_tok (if MoE)
- Any other architecture-specific params

## Step 2: Classify Every Component

For every `self.X = ClassName(...)` found, apply these inference rules to produce
an inferred type, a human-readable label, and dimension annotation.

### Rule A: Type from Class Name (apply in order, first match wins)

| Pattern (case-insensitive) | Inferred type |
|---|---|
| `.*Linear.*` or `.*Proj.*` | Projection |
| `.*Norm.*` | Normalization |
| `SiLU\|GELU\|ReLU\|Swish\|Mish\|.*Act.*\|SiluAndMul` | Activation |
| `.*Attn.*` or `.*Attention.*` | Attention-candidate → refine from forward() |
| `.*Conv.*` | Convolution |
| `.*Mixer.*` or `.*Mamba.*` or `.*SSM.*` | Recurrent-candidate → refine from forward() |
| `.*MLP.*` or `.*FFN.*` or `.*FeedForward.*` | FFN-block |
| `.*Embed.*` | Embedding |
| `.*Rout.*` or `.*Gate.*` (in routing context) | Router |
| `.*Expert.*` | Expert-FFN |
| anything else | Generic → read forward() for refinement |

**Refinement for Attention-candidate** (inspect this class's forward() method):
- Contains `softmax` → type = Standard-Attention
- Contains `chunk_*`, `selective_scan`, `recurrent_`, state update (h_t = f(h_{t-1},...)) → type = Recurrent-op
- Contains `F.conv1d` as the core computation → type = Conv-op

**Refinement for Recurrent-candidate** (inspect this class's forward()):
- Contains selective state update (gating + state matrix) → type = Recurrent-op (SSM)
- Contains delta rule update (`chunk_gated_delta_rule`, `fused_recurrent_*`) → type = Recurrent-op (Linear-Attn)

**Refinement for Generic** (read its forward()):
- matmul + activation without state → likely Projection
- state update pattern → Recurrent-op
- truly opaque (custom CUDA kernel, no readable Python logic) → type = Custom-op

### Rule B: Label from Attr Name (tokenize snake_case)

Split the attr name on `_` into tokens. Map each token using the table below, then
reconstruct a Title Case label.

**Token map:**
```
q / query       → Q
k / key         → K
v / value       → V
z               → Z
g / gate        → Gate
up              → Up
down            → Down
in / input      → Input
out / output    → Output
proj / projection → Projection
linear          → Linear
norm            → Norm
embed           → Embedding
head / lm       → LM (for lm_head), Head (otherwise)
mixer           → Mixer
conv            → Conv
correction      → Correction
prediction      → Prediction
coefs / coef    → Coefficients
a / b           → A / B  (SSM state params)
dt              → Dt
bias            → Bias
router          → Router
expert          → Expert
shared          → Shared
```

**Joining rules:**
- Multiple role tokens at the same position (q, k, v, z) → join with "/" → "Q/K/V/Z"
- `gate` + `up` → "Gate+Up"
- Leading `in_` → treat as "Input" prefix: `in_proj_qkvz` → "Q/K/V/Z Input Projection"
- Leading `out_` or trailing `out_proj` → treat as "Output Projection"

**Examples:**
```
in_proj_qkvz    → "Q/K/V/Z Input Projection"
gate_up_proj    → "Gate+Up Projection"
down_proj       → "Down Projection"
o_proj          → "Output Projection"
qkv_proj        → "QKV Projection"
correction_coefs → "Correction Coefficients"
a_log           → "A Log"
dt_bias         → "Dt Bias"
b_proj          → "B Projection"
shared_expert   → "Shared Expert"
my_weird_linear → "My Weird Linear"  (fallback: Title Case)
```

**Fallback:** snake_case → Title Case (e.g. `residual_multiplier` → "Residual Multiplier")

### Rule C: Dimension from Constructor Args

Look at the class's `__init__` arguments and map to symbolic names:

| Arg name seen | Symbolic label |
|---|---|
| `hidden_size` | `d_model` |
| `intermediate_size` | `d_ff` |
| `moe_intermediate_size` | `d_ff_moe` |
| `num_attention_heads` | `H` |
| `num_key_value_heads` | `H_kv` |
| `head_dim` | `d_head` |
| `vocab_size` | `vocab` |

For Linear-type ops: look for first two positional args or `in_features`/`out_features`.
Express as `[K → N]` using symbolic names. If not resolvable, write `(dims unknown)`.

## Step 3: Output the COMPONENT INVENTORY

Use exactly this format:

```
COMPONENT INVENTORY
===================
Model: <ModelName>
File: <path>

## Top-level structure
  <TopLevelClass>ForCausalLM
  └── model: <ModelClass>
       └── embed_tokens: Token Embedding  [vocab → d_model]
       └── layers: Decoder Block ×<N>
            └── (see layer types below)
       └── norm: RMSNorm
  └── lm_head: LM Head  [d_model → vocab]

## Decoder Layer Type(s)

### <LayerClassName1>  (applies to: all layers)
  ─ OR, for hybrid ─
### <LayerClassName1>  (applies to: layers where config.X == 'Y', e.g. mamba layers)
### <LayerClassName2>  (applies to: layers where config.X == 'Z', e.g. attention layers)

For each layer class, list every self.X:
  self.<attr>  (<ClassName>)
  → Type: <Projection | Normalization | Standard-Attention | Recurrent-op | Conv | FFN-block | Activation | Router | Expert-FFN | Embedding | Custom-op | Generic>
  → Label: "<human readable label>"
  → Dims: [K → N]  or  (dims unknown)
  → Notes: <anything unusual — conditional, gated, new shape, etc.>

## Conditional / Hybrid Structure
  <Describe any if/else in __init__ or forward() that changes the computation graph>
  <For hybrid models: which layer indices use which class>

## Novel / Unknown Components
  <List any self.X where type = Generic or Custom-op>
  <State: "will draw as gray rounded rect" or "will draw as gray hexagon (custom op)">

## Config Legend Values
  d_model = <value>
  H = <num_heads>,  H_kv = <num_kv_heads>
  d_head = <head_dim>
  d_ff = <intermediate_size>  (if applicable)
  d_ff_moe = <moe_intermediate_size>  (if applicable)
  vocab = <vocab_size>
  N = <num_hidden_layers> layers
  E = <num_experts> experts, K = <num_experts_per_tok> active  (if MoE)
  <other relevant params>
```

After outputting the inventory, add this line:
```
---
Inventory complete. Awaiting user confirmation before generating draw.io XML.
```
```

---

## Agent Template B — Generation

When launching the Generation Agent, use this prompt (fill in `{MODEL_FILE_PATH}`, `{OUTPUT_PATH}`, and `{INVENTORY}`):

```
You are a diagram generation agent. Your task is to generate a draw.io XML file
for a PyTorch model. The component analysis has already been done — use it directly.

## Input File
{MODEL_FILE_PATH}

## Output File
Write to: {OUTPUT_PATH}

## Confirmed Component Inventory
{INVENTORY}

## Step 1: Map Inventory Types to Visual Styles

For every component in the inventory, select its draw.io style using this table:

| Inventory type | draw.io style |
|---|---|
| Projection / Embedding / LM Head | `rounded=1;whiteSpace=wrap;html=1;arcSize=20;fillColor=#fff2cc;strokeColor=#d6b656;` |
| Normalization | `rounded=0;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;` |
| Standard-Attention | `shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;` |
| Recurrent-op (SSM / linear-attn) | `shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#f0e6ff;strokeColor=#9673a6;` |
| Conv-op | `shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;` |
| Activation | `rounded=1;whiteSpace=wrap;html=1;arcSize=50;fillColor=#d5e8d4;strokeColor=#82b366;` |
| FFN-block (swimlane container) | `swimlane;fontStyle=1;fillColor=#d5e8d4;strokeColor=#82b366;` |
| Router | `rhombus;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;` |
| Expert-FFN (swimlane) | `swimlane;fontStyle=1;fillColor=#ffe6cc;strokeColor=#d79b00;` |
| Generic | `rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontColor=#333333;` |
| Custom-op | `shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;` |

**For Hybrid models** (inventory listed multiple layer classes):
Place ALL layer variants **side by side** as sub-swimlanes inside the Decoder Block swimlane.
- Each sub-swimlane gets its own full internal flow (do not collapse or simplify)
- Label each sub-swimlane: `<TypeName> Block&#xa;(<applies-to> layers)`
- Label the outer Decoder Block: `Decoder Block ×N&#xa;(e.g. K×type1 + M×type2)`
- This WILL produce a wide diagram — that is correct

## Step 2: Build Node Label Strings

Use the inventory labels exactly. Format for nodes:
- Operation node: `<Label>&#xa;[K → N]` (use `&#xa;` for line break, `→` as `&#x2192;`, `×` as `&#xD7;`)
- Tensor node: `<name>&#xa;[B, T, d]`
- Swimlane: `<Group Name>` (no dims in header)

Keep all dimensions **symbolic** (d_model, d_ff, H, etc.). Never substitute actual numbers.

## Step 3: Layout Calculation — CRITICAL, DO THIS BEFORE WRITING ANY XML

**BEFORE WRITING ANY XML**, do a full layout calculation pass:
1. List all nodes top-to-bottom in execution order
2. For each node: `Y_node = Y_previous_node + H_previous_node + gap`
3. Only then write the XML — **never guess or estimate coordinates**

### Standard node dimensions

| Node type | Width | Height |
|---|---|---|
| Operation (Projection, Norm, Activation) | 220 | 70 |
| Tensor (dashed rounded rect) | 200 | 65 |
| Hexagon (Attention, Recurrent, Conv, Custom) | 220 | 80 |
| Split diamond | 80 | 65 |
| Add ⊕ circle | 50 | 50 |
| Swimlane header | — | 35 |
| Small tensor (inside parallel branch) | 120 | 50 |
| Small operation (inside parallel branch) | 120 | 60 |

**Vertical gap between siblings: 35px. Extra 20px gap before/after swimlane entry/exit.**

### Single-column main flow

Center X for the main column: **X = 450** (with swimlane-relative offsets inside containers).
Nodes in the main column: left edge = X_center − width/2.

### Parallel branches (q / k / v fan-out) — inside a full-width swimlane

When a swimlane is ≥ 800px wide:
- q column: center X = 200 (left edge = 100), use standard widths
- k column: center X = 460 (left edge = 350), use standard widths
- v column: center X = 720 (left edge = 610), use standard widths
- Q Norm / K Norm / RoPE: place directly below q / k / v in the same column
- Attention hexagon: center X = 410, width = 220, below all three columns
- All coordinates are **swimlane-relative** (relative to swimlane's top-left)

When a swimlane is narrow (< 500px, e.g. a sub-swimlane in a hybrid layout):
- Use **small** node dimensions (120×50 for tensors, 120×60 for ops)
- q column: center X = 60 (left edge = 0), width = 120
- k column: center X = 170 (left edge = 110), width = 120
- v column: center X = 280 (left edge = 220), width = 120
- Q Norm / K Norm below q / k in same column; v goes straight to attention
- Attention hexagon: center X = 170, width = 220 (may overflow slightly — OK)

### Hybrid layout (two sub-swimlanes side by side)

Left sub-swimlane: X=20, width=400
Right sub-swimlane: X=440, width=400
Outer Decoder Block: width = 20 + 400 + 20 + 400 + 20 = 860px

Inside each 400px sub-swimlane:
- Single-column nodes: center X = 200, width = 340, left edge = 30
- Parallel q/k/v (narrow rules above): q center=70, k center=200, v center=330, width=120

### Swimlane height — COMPUTE EXPLICITLY

Example for a full-attention sub-swimlane (compute the same way for every swimlane):
```
35 (header) + 40 (top pad)
+ 70 (Input Layer Norm) + 35
+ 70 (QKV Projection) + 35
+ 65 (Split diamond) + 35
+ 50 (q/k/v tensors — parallel, use max height = 50) + 35
+ 60 (Q Norm / K Norm — parallel) + 35
+ 60 (RoPE — parallel) + 35
+ 80 (Attention hexagon) + 35
+ 70 (Output Projection) + 35
+ 50 (⊕ Add) + 35
+ 70 (Post Attn Layer Norm) + 35
+ 70 (Gate+Up Projection) + 35
+ 70 (SiLU×Gate) + 35
+ 70 (Down Projection) + 35
+ 50 (⊕ Add)
+ 40 (bottom pad)
= approximately 1340px
```
Apply the same explicit sum to every swimlane. **Never guess — always sum.**

### MoE fan-out layout

- Router diamond: centered in block
- TopK node: below router, centered
- Expert 0 FFN: X=30
- Expert 1 FFN: X=280
- `×K` label (text node): X=530
- Shared Expert: X=780
- Weighted Sum ⊕: centered below all experts

### Canvas size

Use `pageWidth="1654" pageHeight="2339"` (A1) for tall single-column diagrams.
For hybrid/wide diagrams (two sub-swimlanes): `pageWidth="2339" pageHeight="3308"` (A0).

## Step 4: Add Legend Box

Place a legend box outside all swimlanes (top-right), `parent="1"`.

Style: `text;html=1;strokeColor=#666666;fillColor=#f9f9f9;align=left;verticalAlign=top;whiteSpace=wrap;rounded=1;`

Content: use the "Config Legend Values" section from the inventory verbatim,
formatted with `&#xa;` between lines.

Size: 260×200px. Position: X = main_diagram_right_edge + 40, Y = 40.

## Step 5: Generate draw.io XML

Use this structure. Assign sequential integer IDs starting from 2.

```xml
<mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1654" pageHeight="2339" math="0" shadow="0">
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
    <!-- nodes and edges as children of id="1" or inside swimlanes -->
  </root>
</mxGraphModel>
```

### Tensor node (dashed rounded rect — for named intermediate tensors)
```xml
<mxCell id="N" value="name&#xa;[B, T, d]" style="rounded=1;whiteSpace=wrap;html=1;arcSize=15;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=2;dashed=1;dashPattern=8 4;fontStyle=3;" vertex="1" parent="PARENT_ID">
  <mxGeometry x="X" y="Y" width="200" height="65" as="geometry" />
</mxCell>
```

### Operation node
```xml
<mxCell id="N" value="Label&#xa;[K &#x2192; N]" style="STYLE" vertex="1" parent="PARENT_ID">
  <mxGeometry x="X" y="Y" width="220" height="70" as="geometry" />
</mxCell>
```

### Swimlane
```xml
<mxCell id="N" value="Group Label" style="swimlane;fontStyle=1;align=center;verticalAlign=top;startSize=35;fillColor=#f5f5f5;strokeColor=#666666;" vertex="1" parent="1">
  <mxGeometry x="X" y="Y" width="W" height="H" as="geometry" />
</mxCell>
```

### Edge
```xml
<mxCell id="N" value="[B, T, d]" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" source="SRC" target="TGT" parent="PARENT_ID">
  <mxGeometry relative="1" as="geometry" />
</mxCell>
```

### Residual bypass edge (curved, left side)
```xml
<mxCell id="N" value="" style="edgeStyle=elbowEdgeStyle;elbow=vertical;exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;rounded=1;" edge="1" source="SRC" target="ADD_ID" parent="PARENT_ID">
  <mxGeometry relative="1" as="geometry">
    <Array as="points">
      <mxPoint x="LEFT_X" y="SRC_MID_Y" />
      <mxPoint x="LEFT_X" y="ADD_MID_Y" />
    </Array>
  </mxGeometry>
</mxCell>
```

## Step 6: Tensor Annotations on Edges

Add shape labels on edges between key nodes. Use symbolic names only.

- input_ids → Embedding: `[B, T]`
- Embedding → hidden: `[B, T, d_model]`
- normed → first projection: `[B, T, d_model]`
- QKV / Q/K/V/Z split output: `[B, T, d_q]` / `[B, T, d_kv]` (etc.)
- Attention/Recurrent-op output: `[B, T, d_model]`
- Gate+Up Proj output: `[B, T, 2&#xD7;d_ff]`
- Down Proj output: `[B, T, d_model]`
- LM Head output: `[B, T, vocab]`
- For SSM/recurrent ops: annotate state tensors as `[B, H, d_state]` if visible in code

## XML Well-Formedness Rules

- NO XML comments (`<!-- ... -->` are forbidden inside mxCell blocks)
- All attribute values must be XML-escaped: `&amp;` for &, `&lt;` for <, `&gt;` for >, `&quot;` for "
- Use `&#x2192;` for →, `&#xD7;` for ×, `&#xa;` for newline inside attribute values
- Every vertex mxCell must have `vertex="1"`
- Every edge mxCell must have `edge="1"` plus `source` and `target`
- All cells except id="0" and id="1" must have a `parent` attribute
- `mxGeometry` must be the only child of `mxCell`
- IDs must be unique integers

## Output

Write the complete, valid draw.io XML to: `{OUTPUT_PATH}`

After writing, confirm the file was written and report the absolute path.
```

---

## Color and Style Reference

| Component | Fill | Stroke | Shape |
|---|---|---|---|
| Tensor (dashed) | `#dae8fc` | `#6c8ebf` | dashed rounded rect |
| Projection / Embedding | `#fff2cc` | `#d6b656` | rounded rect |
| Normalization | `#e1d5e7` | `#9673a6` | rect |
| Activation | `#d5e8d4` | `#82b366` | rounded rect |
| Standard Attention | `#fff2cc` | `#d6b656` | hexagon |
| Recurrent op (SSM / linear-attn) | `#f0e6ff` | `#9673a6` | hexagon |
| Convolution op | `#d5e8d4` | `#82b366` | hexagon |
| Add ⊕ | `#ffffff` | `#000000` | ellipse |
| Split | `#f5f5f5` | `#666666` | diamond |
| Router | `#f8cecc` | `#b85450` | diamond |
| Expert FFN | `#ffe6cc` | `#d79b00` | swimlane |
| Group/Swimlane | `#f5f5f5` | `#666666` | swimlane |
| Generic / unknown | `#f5f5f5` | `#666666` | rounded rect |
| Custom op | `#f5f5f5` | `#666666` | hexagon |

---

## Example: Minimal Dense Transformer Block (reference XML skeleton)

This example shows the structural pattern for ONE transformer block (no MoE). Use it as a reference for XML structure and positioning only — always derive actual components from the model code and inventory.

```xml
<mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />

    <mxCell id="2" value="input_ids&#xa;[B, T]" style="rounded=1;whiteSpace=wrap;html=1;arcSize=15;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=2;dashed=1;dashPattern=8 4;fontStyle=3;" vertex="1" parent="1">
      <mxGeometry x="400" y="40" width="160" height="60" as="geometry" />
    </mxCell>

    <mxCell id="3" value="Token Embedding&#xa;[vocab &#x2192; d_model]" style="rounded=1;whiteSpace=wrap;html=1;arcSize=20;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
      <mxGeometry x="360" y="140" width="240" height="60" as="geometry" />
    </mxCell>

    <mxCell id="4" value="hidden&#xa;[B, T, d_model]" style="rounded=1;whiteSpace=wrap;html=1;arcSize=15;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=2;dashed=1;dashPattern=8 4;fontStyle=3;" vertex="1" parent="1">
      <mxGeometry x="380" y="240" width="200" height="60" as="geometry" />
    </mxCell>

    <mxCell id="5" value="Decoder Block ×N" style="swimlane;fontStyle=1;align=center;verticalAlign=top;startSize=35;fillColor=#f5f5f5;strokeColor=#666666;" vertex="1" parent="1">
      <mxGeometry x="200" y="340" width="560" height="700" as="geometry" />
    </mxCell>

    <mxCell id="6" value="input_layernorm&#xa;RMSNorm" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;" vertex="1" parent="5">
      <mxGeometry x="180" y="50" width="200" height="70" as="geometry" />
    </mxCell>

    <mxCell id="7" value="normed&#xa;[B, T, d_model]" style="rounded=1;whiteSpace=wrap;html=1;arcSize=15;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=2;dashed=1;dashPattern=8 4;fontStyle=3;" vertex="1" parent="5">
      <mxGeometry x="180" y="155" width="200" height="65" as="geometry" />
    </mxCell>

    <mxCell id="8" value="QKV Projection&#xa;[d_model &#x2192; d_q + 2&#xD7;d_kv]" style="rounded=1;whiteSpace=wrap;html=1;arcSize=20;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="5">
      <mxGeometry x="180" y="255" width="200" height="70" as="geometry" />
    </mxCell>

    <mxCell id="9" value="Split" style="rhombus;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;" vertex="1" parent="5">
      <mxGeometry x="240" y="360" width="80" height="65" as="geometry" />
    </mxCell>

    <mxCell id="13" value="Scaled Dot-Product&#xa;Attention" style="shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="5">
      <mxGeometry x="180" y="480" width="200" height="80" as="geometry" />
    </mxCell>

    <mxCell id="14" value="Output Projection&#xa;[d_q &#x2192; d_model]" style="rounded=1;whiteSpace=wrap;html=1;arcSize=20;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="5">
      <mxGeometry x="180" y="595" width="200" height="70" as="geometry" />
    </mxCell>

    <mxCell id="15" value="&#x2295;" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffffff;strokeColor=#000000;fontSize=18;fontStyle=1;" vertex="1" parent="5">
      <mxGeometry x="255" y="700" width="50" height="50" as="geometry" />
    </mxCell>

    <mxCell id="30" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="2" target="3" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="31" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="3" target="4" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="32" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="4" target="6" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="33" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="6" target="7" parent="5"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="34" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="7" target="8" parent="5"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="35" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="8" target="9" parent="5"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="39" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="9" target="13" parent="5"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="42" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="13" target="14" parent="5"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="43" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="14" target="15" parent="5"><mxGeometry relative="1" as="geometry"/></mxCell>

    <mxCell id="50" value="" style="edgeStyle=elbowEdgeStyle;elbow=vertical;exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;rounded=1;" edge="1" source="6" target="15" parent="5">
      <mxGeometry relative="1" as="geometry">
        <Array as="points">
          <mxPoint x="30" y="85" />
          <mxPoint x="30" y="725" />
        </Array>
      </mxGeometry>
    </mxCell>

  </root>
</mxGraphModel>
```

This skeleton shows structure only. For actual generation: derive all components from the confirmed inventory, compute Y coordinates from height sums, expand MoE fan-outs, and add hybrid sub-swimlanes for hybrid architectures.
