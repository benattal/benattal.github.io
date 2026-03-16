# LaTeX / GitHub Markdown Rendering Caveats

GitHub uses KaTeX/MathJax for math rendering but runs its Markdown parser **before** handing off to the math renderer in some cases. This causes subtle rendering failures that are hard to debug.

---

## The `\text{XXX}_Y` problem (most common)

**Do not write** `$\text{label}_subscript$` in inline math.

When `_` is preceded by `}` (not an alphanumeric), GitHub's CommonMark parser can treat it as an emphasis marker `_`. This consumes the `_`, breaks the math expression, and — because it also unbalances the `$` delimiter count — corrupts all subsequent inline math on the same line.

**Bad:**
```
$\text{gen}_a$, $\text{kv}_0$, $\hat{x}_t$, $\bar{g}_a$
```

**Good:** ensure `_` is always preceded by a letter or digit:
```
$g_a$, $v_0$, $x_t$, $G_a$
```

This applies to any `\command{...}_subscript` pattern, including:
- `\text{foo}_x`
- `\hat{x}_t`, `\bar{g}_a`, `\tilde{z}_k` (the `_` is after the closing `}`)
- `\mathbf{v}_i`, `\mathcal{L}_t`

**Rule of thumb:** if the character immediately before `_` in the raw LaTeX source is `}`, rename the variable to a single letter or move the subscript inside the braces if the semantics allow it.

---

## The `$(...)$` problem

**Do not start** inline math with `$(`:

**Bad:**
```
the pair $(y_{t-1}, y_{t+1})$
```

GitHub may not recognise `$(` as opening a math expression (looks like a dollar amount). Rewrite to avoid it:

**Good:**
```
the pair $y_{t-1}$ and $y_{t+1}$
```

---

## Spacing commands: `\!`, `\;`

- `\!` (negative thin space) — avoid in inline math; can confuse some renderers.
- `\;` (thick space) — safe in display math `$$...$$`, but can fail in inline `$...$`. Use `,` or `,\,` instead.

---

## Display math: always surround with blank lines

```markdown
Preceding text.

$$x_t = t \cdot x_0 + (1-t) \cdot \varepsilon$$

Following text.
```

A `$$` block without a blank line before or after may render inline instead of as a display equation.

---

## Images: add a gitignore exception

`*.png` / `*.jpg` are ignored by default. To commit images in a checkins folder, add:

```
!checkins/images/*.png
!checkins/images/*.jpg
```

to `.gitignore`. Store all checkin images in `checkins/images/` with descriptive names.
