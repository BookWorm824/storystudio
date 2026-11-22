"""Does every question StoryStudio asks the AI also work with ChatGPT?

    python tests/check_chatgpt.py [--live]

Collects every answer form (JSON schema) StoryStudio sends: module-level schemas, the schema
functions, and the forms written inline in each C.ask(...) call (read from the source). Each one
is checked against what ChatGPT's structured answers allow; with --live, each is also sent to
ChatGPT with a tiny "fill this in" question to prove ChatGPT accepts it (uses a little of the
ChatGPT plan, no money).
"""
import ast
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
MODULES = ["ss_claude", "ss_script", "ss_clips", "ss_place", "ss_feedback", "ss_audit",
           "ss_change", "ss_karaoke", "storysound"]
NOT_ALLOWED = {"minItems", "maxItems", "minimum", "maximum", "pattern", "format", "default",
               "oneOf", "allOf", "not", "minLength", "maxLength", "uniqueItems", "const",
               "patternProperties", "if", "then", "else"}


def collect():
    import importlib
    found = {}
    for name in MODULES:
        mod = importlib.import_module(name)
        for k, v in vars(mod).items():                 # module-level forms
            if isinstance(v, dict) and v.get("type") == "object" and "properties" in v:
                found[f"{name}.{k}"] = v
        built = {"shots_schema": lambda m: m.shots_schema(),
                 "cartoon_schema": lambda m: m.cartoon_schema(m.shots_schema()),
                 "with_type(REDO_SHOT)": lambda m: m.with_type(m.REDO_SHOT, "picture")}
        for k, make in built.items():                  # forms the script builder puts together
            if hasattr(mod, k.split("(")[0]):
                try:
                    found[f"{name}.{k}"] = make(mod)
                except Exception as e:
                    found[f"{name}.{k}"] = e
        for k in ("fix_schema", "note_schema", "compare_schema", "ask_schema", "redo_schema"):
            if callable(getattr(mod, k, None)):
                try:
                    found[f"{name}.{k}()"] = getattr(mod, k)()
                except Exception as e:
                    found[f"{name}.{k}()"] = e
        src = (REPO / f"{name}.py").read_text(encoding="utf8")
        tree = ast.parse(src)
        env = {k: v for k, v in vars(mod).items() if not k.startswith("__")}
        for node in ast.walk(tree):                    # forms written inside the call
            if not (isinstance(node, ast.Call) and len(node.args) >= 2):
                continue
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if fname != "ask":
                continue
            arg = node.args[1]
            if isinstance(arg, ast.Dict):
                try:
                    v = eval(compile(ast.Expression(arg), name, "eval"), env)
                    found[f"{name}:{node.lineno}"] = v
                except Exception as e:
                    found[f"{name}:{node.lineno}"] = e
    return found


def problems(schema, path="$"):
    out = []
    if not isinstance(schema, dict):
        return [f"{path}: not a form ({schema!r})"]
    for k in schema:
        if k in NOT_ALLOWED:
            out.append(f"{path}: '{k}' isn't allowed by ChatGPT")
    t = schema.get("type")
    if isinstance(t, list):
        out.append(f"{path}: a list of types")
    if t == "object":
        if "properties" not in schema:
            out.append(f"{path}: an object with no fields listed")
        for k, v in (schema.get("properties") or {}).items():
            out += problems(v, f"{path}.{k}")
    if t == "array":
        if "items" not in schema:
            out.append(f"{path}: a list with no item form")
        else:
            out += problems(schema["items"], f"{path}[]")
    if "enum" in schema:
        kinds = {type(x).__name__ for x in schema["enum"]}
        if len(kinds) > 1:
            out.append(f"{path}: mixed choices {kinds}")
    return out


def depth(s, d=1):
    if not isinstance(s, dict):
        return d
    kids = list((s.get("properties") or {}).values()) + ([s["items"]] if "items" in s else [])
    return max([depth(k, d + (1 if s.get("type") == "object" else 0)) for k in kids] or [d])


def main():
    live = "--live" in sys.argv
    import ss_gpt as G
    forms = collect()
    bad = 0
    print(f"{len(forms)} answer forms found")
    for k, s in sorted(forms.items()):
        if isinstance(s, Exception):
            print(f"  ?    {k}: couldn't be read ({s})")
            continue
        p = problems(G.strict(s))
        if depth(s) > 10:
            p.append(f"nested {depth(s)} deep (ChatGPT allows 10)")
        if p:
            bad += 1
            print(f"  FAIL {k}: " + "; ".join(p[:4]))
    print(f"form check: {len(forms) - bad} fine, {bad} with problems")
    if not live:
        return 1 if bad else 0
    if G.status() != "ready":
        print("ChatGPT isn't ready on this computer; skipping the live check")
        return 1
    uniq = {}
    for k, s in forms.items():
        if isinstance(s, dict):
            uniq.setdefault(json.dumps(s, sort_keys=True), k)

    def one(item):
        key, name = item
        t0 = time.time()
        try:
            G.ask("This is only a test of the answer form. Fill in every field with a short "
                  "placeholder answer (one or two words, numbers like 1, and at most one item in "
                  "each list).", json.loads(key), timeout=300)
            return name, "ok", time.time() - t0
        except Exception as e:
            return name, f"{type(e).__name__}: {e}", time.time() - t0
    print(f"asking ChatGPT with {len(uniq)} different forms…", flush=True)
    fails = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name, res, sec in ex.map(one, uniq.items()):
            print(f"  {'ok  ' if res == 'ok' else 'FAIL'} {name} ({sec:.0f}s){'' if res == 'ok' else ': ' + res}",
                  flush=True)
            fails += res != "ok"
    print(f"live check: {len(uniq) - fails} accepted by ChatGPT, {fails} refused")
    return 1 if fails or bad else 0


if __name__ == "__main__":
    sys.exit(main())
