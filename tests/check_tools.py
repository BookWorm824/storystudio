"""Runs every StoryStudio tool start to finish on a COPY of a real project, in a test home.

    python tests/check_tools.py "<project.prproj>" [test home folder]

Nothing is written to the user's own folders: STORYSTUDIO_HOME points at a test home, copies of
projects go there, the AI and Fal are stand-ins (no credit is spent, answers are made up from
each request's schema), and every project a tool writes is checked: it opens, no reference
points at nothing, no two clips share a moment on one track, every sound file exists.
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REAL_HOME = Path(os.environ.get("STORYSTUDIO_REAL_HOME") or Path.home() / "StoryStudio")
SRC = Path(sys.argv[1])
HOME = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "tests" / "_home"
SRC2 = Path(sys.argv[3]) if len(sys.argv) > 3 else None   # the first "(with sound)" copy
SRC3 = Path(sys.argv[4]) if len(sys.argv) > 4 else None   # the project before any sounds
HOME = HOME / time.strftime("run_%H%M%S")
HOME.mkdir(parents=True)
os.environ["STORYSTUDIO_HOME"] = str(HOME)
os.environ["STORYSTUDIO_NO_WINDOW"] = "1"
os.environ["PATH"] = str(REAL_HOME / "bin") + os.pathsep + os.environ["PATH"]
sys.path.insert(0, str(REPO))

import ss_core as core                                    # noqa: E402
core.save_settings = lambda s: None
import ss_media as M                                      # noqa: E402
M.MODELS = REAL_HOME / "models"
import storysound as E                                    # noqa: E402
import ss_premiere as P                                   # noqa: E402
import ss_prwrite as W                                    # noqa: E402
import ss_claude as C                                     # noqa: E402
import ss_fal                                             # noqa: E402
import ss_sounds as SS                                    # noqa: E402
import ss_server as S                                     # noqa: E402

T = core.TICKS
FAKE = {"t": 13.0}                # where made-up answers put their times
RESULTS = []


# ---- stand-ins ---------------------------------------------------------------------------
def fake(schema, key=""):
    t = schema.get("type")
    if "enum" in schema:
        vals = [v for v in schema["enum"] if v != ""] or schema["enum"]
        return vals[0]
    if t == "object":
        return {k: fake(v, k) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        return [fake(schema["items"], key), fake(schema["items"], key)] if "items" in schema else []
    if t in ("number", "integer"):
        k = key.lower()
        if k in ("start", "time", "at", "from", "t"):
            v = FAKE["t"]
        elif k in ("end", "to", "until"):
            v = FAKE["t"] + 1.5
        elif k in ("note", "id", "shot", "n"):
            v = 1
        else:
            v = 1
        return int(v) if t == "integer" else float(v)
    if t == "boolean":
        return True
    return f"test {key or 'words'}"


C.ask = lambda prompt, schema, system=None, images=None, timeout=0: fake(schema)
C.provider = lambda: "claude"
C.ai_name = lambda: "Claude"
STAND_IN = next(iter(sorted((SRC.parent / "StoryStudio" / "sounds").rglob("*.wav"))), None)
SPENT = []


def fake_fal(jobs, key, progress=None, cancel=None):
    for j in jobs:
        SPENT.append(j["out"])
        for f in [j["out"]] + list(j.get("cands") or []):
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(STAND_IN, f)
    return {}, {"seconds": {}, "cost": 0.0}


def fake_pick(groups, progress=None, cancel=None, scores=None):
    for g in groups:
        if g.get("cands") and Path(g["cands"][0]).exists():
            shutil.copyfile(g["cands"][0], g["out"])
    return {}


ss_fal.run_jobs = fake_fal
SS.run_pick = fake_pick
SS.run_score = lambda groups, cancel=None: [0.5 for _ in groups]
SS.local_ready = lambda: False
os.startfile = lambda p: None
S.premiere_exe = lambda: None                            # never open Premiere
S.S.settings.update({"fal_key": "test-key", "sound_source": "auto", "best_of": 1,
                     "busy": "cartoon", "add_music": False})


# ---- the real sound runs for this timeline, so line-up has what it needs -----------------
def bring_runs(uid):
    n = 0
    for f in list((REAL_HOME / "work").glob("*/state.json")) + \
            list((REAL_HOME / "work").glob("*/state_sounds.json")) + \
            list((REAL_HOME / "work").glob("*/changes/*/state.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if d.get("seq_uid") != uid or not any(c.get("file") for c in d.get("cues") or []):
            continue
        dst = E.WORK / f.relative_to(REAL_HOME / "work")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    return n


# ---- checking a written project ----------------------------------------------------------
def problems(path, uid, base=None):
    out = []
    try:
        w = W.Writer(path)
        seq = w.sequence(uid)
    except Exception as e:
        return [f"doesn't open: {e}"]
    dangling = set()
    for el in w.root.iter():
        for a in ("ObjectRef", "ObjectURef"):
            k = el.get(a)
            if k and k not in w.objs:
                dangling.add(k)
    if base is not None:
        bw = W.Writer(base)
        dangling -= {el.get(a) for el in bw.root.iter() for a in ("ObjectRef", "ObjectURef")
                     if el.get(a) and el.get(a) not in bw.objs}
    if dangling:
        out.append(f"{len(dangling)} references point at nothing: {sorted(dangling)[:5]}")
    items = W._all_items(w, seq)
    tracks = {}
    for i in items:
        tracks.setdefault((i["kind"], id(i["track"]), i["tname"]), []).append(i)
    for (kind, _, tname), its in tracks.items():
        its.sort(key=lambda i: i["start"])
        for a, b in zip(its, its[1:]):
            if a["end"] > b["start"] + T // 1000:
                out.append(f"overlap on {kind} track '{tname}': {a['name'][:30]} "
                           f"({a['end'] / T:.2f}) over {b['name'][:30]} ({b['start'] / T:.2f})")
    for i in items:
        if i["end"] <= i["start"]:
            out.append(f"clip with no length: {i['name'][:40]}")
        if i["kind"] == "audio" and i["file"] and not os.path.exists(i["file"]):
            out.append(f"sound file missing: {i['file'][-70:]}")
    for g in seq.findall("TrackGroups/TrackGroup"):
        grp = w.get(g, "Second")
        for tr in w.refs(grp, "TrackGroup/Tracks/Track") if grp is not None else []:
            lst = tr.find("ClipTrack/ClipItems/TrackItems")
            if lst is not None:
                idx = [e.get("Index") for e in lst]
                if idx != [str(k) for k in range(len(idx))]:
                    out.append(f"track list out of order: {idx[:8]}")
    try:
        if not P.Project(path).sequences():
            out.append("StoryStudio's reader finds no sequence")
    except Exception as e:
        out.append(f"StoryStudio's reader can't open it: {e}")
    return out


def sounds_of(path, uid):
    w = W.Writer(path)
    return [i for i in W._all_items(w, w.sequence(uid)) if i["kind"] == "audio"]


# ---- running steps -----------------------------------------------------------------------
def wait_job(label, limit=900):
    t0 = time.time()
    while S.S.job and not S.S.job.done:
        if time.time() - t0 > limit:
            raise RuntimeError(f"{label}: still running after {limit}s")
        time.sleep(0.3)
    j = S.S.job
    if j and j.failed:
        raise RuntimeError(f"{label}: job failed: {j.error or '(see log)'}")


def api(route, **a):
    r = S.api(route, a)
    if isinstance(r, dict) and r.get("job"):
        wait_job(route)
    return r


def step(name, fn):
    t0 = time.time()
    try:
        note = fn()
        RESULTS.append((name, "ok", note or "", time.time() - t0))
        print(f"  ok   {name}  {note or ''}  ({time.time() - t0:.0f}s)", flush=True)
        return True
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}", time.time() - t0))
        print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def need_clean(path, uid, base=None):
    p = problems(path, uid, base)
    assert not p, f"{Path(path).name}: " + "; ".join(p[:6])


def main():
    proj = HOME / "projects" / "Chester test.prproj"
    proj.parent.mkdir(parents=True)
    shutil.copy2(SRC, proj)
    w = W.Writer(proj)
    uid = (w.sequence().get("ObjectUID") or w.sequence().get("ObjectID"))
    print("test home:", HOME, "| runs brought:", bring_runs(uid), flush=True)
    before = sounds_of(proj, uid)

    def t_open():
        r = api("open_project", path=str(proj))
        assert r["project"]["with_sound"], "project not seen as having StoryStudio sounds"
        need_clean(proj, uid, proj)
        return f"{len(before)} sound clips on the timeline"
    step("open the project", t_open)

    def t_prompts():
        r = api("sound_prompts")
        on = [x for x in r["rows"] if x["time"]]
        assert len(r["rows"]) > 20 and on, "prompt list is empty"
        assert all(x["sound"] for x in r["rows"]), "a sound has no prompt"
        p = api("save_prompts")["path"]
        assert Path(p).exists() and str(HOME) in p, f"prompt file went to {p}"
        return f"{len(r['rows'])} sounds, {len(on)} on the timeline"
    step("my sound prompts", t_prompts)

    def t_change():
        fx = next(i for i in before if i["tname"].startswith("StoryStudio effects")
                  and i["start"] > 30 * T)
        t = fx["start"] / T + 0.3
        r = api("sounds_at", time=f"{t:.2f}")
        pick = next((x for x in r["sounds"] if x["id"] == fx["id"]), None)
        assert pick, "the sound playing there isn't listed"

        def do(x, action, at, note=""):
            return api("change_sound", id=x["id"], name=x["raw"], start=x["start"],
                       action=action, t=at, note=note)
        r1 = do(pick, "remove", r["t"], "test")
        assert str(HOME) in r1["project"], "the change went outside the test home"
        need_clean(r1["project"], uid, proj)
        names = [i["name"] for i in sounds_of(r1["project"], uid)]
        assert len(names) == len(before) - 1, f"remove: {len(before)} -> {len(names)} clips"
        assert all(x["id"] != fx["id"] or x["raw"] != fx["name"]
                   for x in api("sounds_at", time=f"{t:.2f}")["sounds"]), \
            "the removed sound is still listed"
        os.utime(proj, None)                              # she presses Ctrl+S in Premiere
        api("open_project", path=str(proj))
        later = api("sounds_at", time=f"{t + 6:.2f}")
        fx2 = next(x for x in later["sounds"] if x["kind"] in ("effect", "character"))
        r2 = do(fx2, "remove", later["t"])
        names2 = [i["name"] for i in sounds_of(r2["project"], uid)]
        assert names2.count(fx["name"]) == names.count(fx["name"]), \
            "the first change was lost after saving in Premiere"
        assert len(names2) == len(before) - 2, f"two removes: {len(before)} -> {len(names2)}"
        # a stale or wrong id never touches another clip
        vid = next(i for i in W._all_items(W.Writer(proj), W.Writer(proj).sequence(uid))
                   if i["kind"] == "video")
        try:
            api("change_sound", id=vid["id"], name="", start=None, action="remove", t=1.0)
            raise AssertionError("it removed a clip that isn't a StoryStudio sound")
        except Exception as e:
            assert "StoryError" in type(e).__name__ or "changed on the timeline" in str(e), e
        bed = next((x for x in api("sounds_at", time=f"{t:.2f}")["sounds"] if x["long"]), None)
        if bed and bed["start"] + 1 < r["t"] < bed["end"] - 1:
            r3 = do(bed, "stop", r["t"])
            need_clean(r3["project"], uid, proj)
            b2 = next(i for i in sounds_of(r3["project"], uid) if i["name"] == bed["raw"])
            assert abs(b2["end"] / T - r["t"]) < 0.06, "stop: the sound doesn't end there"
        FAKE["t"] = t
        fx3 = next(x for x in api("sounds_at", time=f"{t + 12:.2f}")["sounds"]
                   if x["kind"] in ("effect", "character"))
        n0 = len(SPENT)
        do(fx3, "remake", t + 12, "softer")
        last = api("last_change")
        assert last.get("project"), "remake gave no result"
        need_clean(last["project"], uid, proj)
        assert len(SPENT) > n0, "remake made no sound"
        assert not any(str(SRC.parent) in f for f in SPENT), "a sound was written in her folder"
        final = [i["name"] for i in sounds_of(last["project"], uid)]
        assert fx3["raw"] not in final or final.count(fx3["raw"]) < names2.count(fx3["raw"]), \
            "the remade sound's old version is still there"
        assert len(final) == len(before) - 2, \
            f"after remove, remove, stop, remake: {len(before)} -> {len(final)} clips"
        (Path(S.S.st["work"]) / "sound_changes.json").unlink()   # the next tests start clean
        return "remove, save in Premiere, remove, wrong clip refused, stop, remake"
    step("change a sound I don't like", t_change)

    def t_lineup():
        api("open_project", path=str(proj))
        r = api("line_up_sounds")
        need_clean(r["project"], uid, proj)
        res = sounds_of(r["project"], uid)
        assert len(res) >= len(before) - r["removed"], "line-up lost sounds it didn't report"
        if r["changed"]:
            api("open_project", path=r["project"])       # a second pass must change nothing
            r2 = api("line_up_sounds")
            assert not r2["changed"], f"lining up twice still changes things: " \
                f"moved {r2['moved']} removed {r2['removed']} restored {r2['restored']}"
        note = f"moved {r['moved']}, removed {r['removed']}, {len(r['missing'])} reported missing"
        if r["missing"]:                                  # putting them back, then stable
            api("open_project", path=str(proj))
            r3 = api("line_up_sounds", restore=[[m["name"], m["at"]] for m in r["missing"]])
            need_clean(r3["project"], uid, proj)
            assert r3["restored"] == len(r["missing"]), \
                f"asked for {len(r['missing'])} back, got {r3['restored']}"
            api("open_project", path=r3["project"])
            r4 = api("line_up_sounds")
            assert not r4["changed"] and not r4["missing"], \
                f"after putting sounds back it still wants changes: {r4['moved']} " \
                f"{r4['removed']} missing {len(r4['missing'])}"
            note += f"; put back {r3['restored']}; second pass changes nothing"
        return note
    step("line up my sounds again", t_lineup)

    def t_lineup_edits():
        """Move, trim and delete shots by hand in a copy (sounds left behind), line up, and
        check every effect is on the right moment of the right shot."""
        if not SRC2 or not SRC2.exists():
            return "no '(with sound)' original given (skipped)"
        p2 = HOME / "projects" / "Edited (with sound).prproj"
        shutil.copy2(SRC2, p2)
        w0 = W.Writer(p2)
        its = W._all_items(w0, w0.sequence(uid))
        vids = sorted([i for i in its if i["kind"] == "video"], key=lambda i: i["start"])
        fx0 = [i for i in its if i["kind"] == "audio" and i["tname"].startswith("StoryStudio effects")]
        assert len(vids) > 30 and len(fx0) > 50, "not the project this test expects"

        def moment(snd, videos):        # (shot file, seconds into that shot's own clip)
            v = W._anchor(snd, videos)
            return v["file"], (v["in"] + (snd["start"] - v["start"]) * v["speed"]) / T
        truth = {}
        for f in fx0:
            truth.setdefault(f["name"], []).append(moment(f, vids))
        # the edit: swap shots 10 and 11, trim 1 s off the front of shot 20, delete shot 30
        order = list(vids)
        order[9], order[10] = order[10], order[9]
        dead = order.pop(29)
        trim = order[19]
        t = vids[0]["start"]
        for v in order:
            ln = v["end"] - v["start"]
            if v is trim:
                cut = T
                W.Writer.setx(v["c"], "InPoint", v["in"] + int(cut * v["speed"]))
                ln -= cut
            W._set_times(v["cti"], t, t + ln)
            t += ln
        lst = dead["track"].find("ClipTrack/ClipItems/TrackItems")
        for e in list(lst):
            if e.get("ObjectRef") == dead["id"]:
                lst.remove(e)
        items = sorted(list(lst), key=lambda e: W._num(w0.objs[e.get("ObjectRef")],
                                                       "ClipTrackItem/TrackItem/Start"))
        for e in items:
            lst.remove(e)
        for n, e in enumerate(items):
            e.set("Index", str(n))
            lst.append(e)
        w0.save(p2)
        api("open_project", path=str(p2))
        r = api("line_up_sounds")
        assert r["changed"], "nothing was lined up after the edit"
        need_clean(r["project"], uid, p2)
        w1 = W.Writer(r["project"])
        its1 = W._all_items(w1, w1.sequence(uid))
        v1 = [i for i in its1 if i["kind"] == "video"]
        fx1 = [i for i in its1 if i["kind"] == "audio" and i["tname"].startswith("StoryStudio effects")]
        wrong, checked = [], 0
        for f in fx1:
            file, sec = moment(f, v1)
            want = truth.get(f["name"], [])
            head = f["in"] / T          # a sound trimmed at the front starts later in its shot
            if not any(file == wf and (abs(sec - ws) < 0.06 or abs(sec - head - ws) < 0.06)
                       for wf, ws in want):
                wrong.append((f["name"][:40], Path(file).stem, round(sec, 2), want[:1]))
            checked += 1
        assert not wrong, f"{len(wrong)} of {checked} sounds are on the wrong moment: {wrong[:4]}"
        left = [f["name"][:40] for f in fx1 if truth.get(f["name"]) and
                all(wf == dead["file"] for wf, _ in truth[f["name"]])]
        assert not left, f"sounds of the deleted shot are still there: {left[:4]}"
        api("open_project", path=r["project"])
        r2 = api("line_up_sounds")
        assert not r2["changed"], "a second pass still changes things"
        return (f"{checked} sounds checked on their shots after a swap, a trim and a delete; "
                f"moved {r['moved']}, removed {r['removed']}")
    step("line-up after moving, trimming and deleting shots", t_lineup_edits)

    def t_newshots():
        import ss_audit as A
        real_sources = A.sound_sources                    # as it was before sounds were made
        A.sound_sources = lambda st: [x for x in real_sources(st) if not x[0].get("only_shots")
                                      and x[0].get("stage") != "change"]
        try:
            return newshots()
        finally:
            A.sound_sources = real_sources

    def newshots():
        api("open_project", path=str(proj))
        r = api("new_shots")
        if r.get("none_yet") or not r["shots"]:
            return "no new shots on this timeline (skipped)"
        sh = r["shots"][0]
        FAKE["t"] = sh["start"] + 1.0
        n0 = len(SPENT)
        api("start_sounds", only_shots=[[sh["start"], sh["end"]]], busy="cartoon")
        cues = S.S.st["cues"]
        assert cues, "no sounds were planned for the new shot"
        assert all(sh["start"] - 0.3 <= c["start"] < sh["end"] + 0.3 for c in cues), \
            "a planned sound is outside the new shot"
        assert not any(c["kind"] in E.BEDS for c in cues), "a background or music was planned"
        made = len(SPENT) - n0
        api("finish")
        out = S.S.st["outputs"]
        assert out.get("project"), out.get("project_error")
        need_clean(out["project"], uid, proj)
        after = sounds_of(out["project"], uid)
        assert len(after) > len(before) - 3, "existing sounds went missing"
        return f"{len(cues)} sounds planned for '{sh['name']}', {made} made"
    step("sounds for my new shots", t_newshots)

    def t_fix():
        api("open_project", path=str(proj))
        n0 = len(SPENT)
        api("start_fix_audio")
        out = S.S.st["outputs"]
        assert out.get("project"), out.get("project_error")
        need_clean(out["project"], uid, proj)
        a = {i["id"]: i for i in sounds_of(out["project"], uid)}
        b = {i["id"]: i for i in before}
        assert set(a) == set(b), "fix the audio added or removed clips"
        swapped = [b[k]["name"] for k in a if a[k]["file"] != b[k]["file"]]
        assert not swapped, f"fix the audio swapped files: {swapped[:4]}"
        moved = [b[k]["name"] for k in a if a[k]["start"] != b[k]["start"]]
        assert not moved, f"fix the audio moved clips: {moved[:4]}"
        assert len(SPENT) == n0, "fix the audio made sounds without being asked"
        return "; ".join(S.S.st["levels"][1:4])[:150]
    step("fix the audio (volumes only)", t_fix)

    def t_feedback():
        api("open_project", path=str(proj))
        FAKE["t"] = 30.0
        api("start_feedback", kind="cartoon", claude=True, focus="all")
        fb = api("feedback")["feedback"]
        assert fb and fb["notes"] is not None, "no feedback came back"
        if fb["notes"]:
            api("fb_toggle", id=fb["notes"][0]["id"], keep=False)
            api("fb_save")
        out = S.S.st["outputs"]
        if out.get("project"):
            need_clean(out["project"], uid, proj)
        if fb.get("can_retime"):
            r = api("fb_retime")
            need_clean(r["project"], uid, proj)
        return f"{len(fb['notes'])} notes"
    step("edit feedback and retime", t_feedback)

    def t_audit():
        api("open_project", path=str(proj))
        FAKE["t"] = 40.0
        api("start_audit")
        au = api("audit")["audit"]
        assert au and au["checked"], "the story check didn't finish"
        api("audit_ask", text="it looks odd here", at=40.0)
        au = api("audit")["audit"]
        assert au["asks"], "her note wasn't answered"
        if au.get("project"):
            need_clean(au["project"], uid, proj)
        return f"{len(au['notes'])} notes, {len(au['asks'])} answer"
    step("story check and tell StoryStudio", t_audit)

    def t_full():
        """The whole sound-effects tool on a project with no sounds yet, then the rule that a
        copy she has saved in Premiere is never written over."""
        if not SRC3 or not SRC3.exists():
            return "no project without sounds given (skipped)"
        p3 = HOME / "projects" / "Fresh.prproj"
        shutil.copy2(SRC3, p3)
        api("open_project", path=str(p3))
        FAKE["t"] = 20.0
        n0 = len(SPENT)
        api("start_sounds", busy="cartoon")
        rv = api("review")["review"]
        cues = S.S.st["cues"]
        assert cues and any(c.get("file") for c in cues), "no sounds were made"
        made = [c for c in cues if c.get("file")]
        api("toggle", id=made[0]["id"], keep=True)
        api("nudge", id=made[0]["id"], by=0.2)
        old_file = made[0]["file"]
        api("remake", id=made[0]["id"])
        c0 = next(c for c in S.S.st["cues"] if c["id"] == made[0]["id"])
        assert c0.get("file") and Path(c0["file"]).exists(), "the remade sound has no file"
        assert c0["file"] != old_file, "a remade sound was written over the old file"
        assert Path(old_file).exists(), "the old sound file is gone after a remake"
        api("finish")
        out = S.S.st["outputs"]
        assert out.get("project"), out.get("project_error")
        need_clean(out["project"], uid, p3)
        first = Path(out["project"])
        n_first = len(sounds_of(first, uid))
        assert n_first > 0, "the copy has no sounds in it"
        api("finish")                                     # untouched copy: replaced in place
        assert Path(S.S.st["outputs"]["project"]) == first, "an untouched copy wasn't reused"
        time.sleep(2.1)
        os.utime(first, None)                             # she saves that copy in Premiere
        stamp = first.stat().st_mtime
        api("finish")
        second = Path(S.S.st["outputs"]["project"])
        assert second != first, "a copy she saved in Premiere was written over"
        assert first.stat().st_mtime == stamp, "her saved copy was changed"
        return (f"{len(made)} sounds made ({len(SPENT) - n0} files), remade one to a new file; "
                f"her saved copy kept, new copy '{second.name}'")
    step("add sound effects, and never overwrite her copy", t_full)

    def t_update():
        import ss_update
        real = ss_update.check
        ss_update.check = lambda timeout=8: None
        try:
            S.RUN_VERSION = "0.0.1"
            r = api("update_check")
            assert r["update"], "a newer installed version isn't offered"
        finally:
            ss_update.check = real
            S.RUN_VERSION = S._version_here()
        return "offers to restart into the installed version"
    step("update when a newer version is installed", t_update)

    bad = [r for r in RESULTS if r[1] != "ok"]
    touched = [f for f in SRC.parent.glob("*.prproj") if f.stat().st_mtime > START]
    print("\n==== summary ====")
    for name, ok, note, sec in RESULTS:
        print(f"{ok:4} {name}: {note}")
    if touched:
        print("!! files changed in her project folder during the test:", [f.name for f in touched])
    print("log:", HOME / "storysound_log.txt")
    return 1 if bad or touched else 0


START = time.time()
if __name__ == "__main__":
    sys.exit(main())
