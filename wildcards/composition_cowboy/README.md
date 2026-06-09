# Composition Cowboy wildcards

13 creative wildcard files (1000 lines each), one per text input on the
**🤠 Composition Cowboy** node — `title`, `subtitle`, `hero`, `body`, `brand`,
`extra`, `high_level_description`, `background`, `style_detail`, `aesthetics`,
`lighting`, `medium`, `style_palette`.

## Install

These are plain Mikey-style wildcard files (one option per line). Copy this folder
into your ComfyUI **user wildcards** directory so a wildcard node can find it:

```
ComfyUI/user/wildcards/composition_cowboy/
```

## Use

Feed a Mikey **Wildcard Processor** node (one per input) with the matching ref and
wire its STRING output into the corresponding Composition Cowboy socket:

```
__composition_cowboy/hero__
__composition_cowboy/title__
__composition_cowboy/style_palette__
...
```

`../../example_workflows/composition_cowboy_wildcards.json` has all 13 pre-wired.

Random pick per run: set the Wildcard Processor seed control to `randomize`, or use
the `__*composition_cowboy/<name>__` form. Pull several lines at once with
`3$$__composition_cowboy/body__`.

## Regenerate

Re-roll the contents (or change the seed / widen the vocab banks) with:

```
python ../../tools/gen_composition_wildcards.py
```
