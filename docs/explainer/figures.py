"""The Prospecta explainer's figures, one function each, written to diagrams/*.svg by build.py.

They use the house figure primitives (fig.py, fetched unchanged from the Spire explainer builder by build.sh): boxes with wrapped text,
arrows, one palette. A box whose text does not fit makes the build fail. Every example in them is invented: the
diary of Ines, a gardener, traveller and cook who does not exist. The numbers in the charts are real (see README.md).
"""
import math
from pathlib import Path

import fig as _fig
from fig import ACCENT, DETOUR, HIT, INK, INK2, LINE, MISSED, MUTED, RED, SOFT, Fig

OVER = _fig.OVER
_wrap = _fig.wrap
_fig.wrap = lambda text, width, size: _wrap(text, width * 0.9, size)  # Noto Sans runs wider than the primitive's estimate
GREY = "#f1efe9"
W = 1680

# The invented diary: file name, date, text. Used by the figures and, word for word, by the slides.
NOTES = [
    ("seed-order", "18 Apr 2025", "Ordered twelve seed packets from Fenwick Seeds: Black Krim tomatoes, Genovese basil, kale and nine more."),
    ("tomato-planting", "12 May 2025", "Planted the Black Krim tomatoes in bed 3. Soil was 16 degrees by Mr. Okafor's thermometer."),
    ("lisbon-stay", "2 Jun 2025", "A short stay at the Pensão Azul in Lisbon. Best breakfast: custard tarts on Rua Augusta."),
    ("fence-rebuild", "21 Jun 2025", "Mr. Okafor and I rebuilt the back fence. He brought cedar posts; I made lemon rice."),
    ("yellow-leaves", "8 Jul 2025", "Lower tomato leaves yellow with brown spots. Mr. Okafor says early blight. Sprayed copper."),
    ("heat-wave", "19 Jul 2025", "37 degrees. Watered twice. The basil bolted."),
    ("first-harvest", "14 Aug 2025", "First big harvest: nine kilos of Black Krim. Sauce: garlic, basil, red wine."),
    ("compost-skipper", "2 Mar 2025", "Dara turned the compost heap again. Family calls her Dari; I call her Skipper Compost."),
    ("zine-column", "10 Sep 2025", "Dara's first garden zine column ran under the pen name Fern Aldous."),
]
NOTE = {n: (d, t) for n, d, t in NOTES}


class Fig2(Fig):
    """The house Fig, with the type set larger: explainer slides are read from a distance."""
    TS, BS, XS = 1.12, 1.18, 1.15

    def box(self, x, y, w, h, title=None, body=None, **kw):
        if "tsize" in kw:
            kw["tsize"] = round(kw["tsize"] * self.TS)
        elif title:
            kw["tsize"] = round(26 * self.TS)
        if "bsize" in kw:
            kw["bsize"] = round(kw["bsize"] * self.BS)
        elif body:
            kw["bsize"] = round(20 * self.BS)
        return super().box(x, y, w, h, title, body, **kw)

    def text(self, x, y, s, size=22, weight=400, color=INK2, anchor="start", italic=False):
        return super().text(x, y, s, round(size * self.XS) if size < 30 else size, weight, color, anchor, italic)


def F(h=550):
    return Fig2(W, h)


# ---------------------------------------------------------------- small helpers
def cyl(f, x, y, w, h, lines, fill=SOFT, stroke=ACCENT, size=26):
    ry = 26
    f.add(f'<path d="M{x} {y + ry} V{y + h - ry} A{w / 2} {ry} 0 0 0 {x + w} {y + h - ry} V{y + ry}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    f.add(f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="#f4f2ff" stroke="{stroke}" stroke-width="3"/>')
    f.lines(x + w / 2, y + ry * 2 + 34, lines, size, stroke, round(size * 1.3), "middle", 700)


def pill(f, x, y, text, color=ACCENT, size=19, fill=None, tcolor="#fff", anchor="start"):
    es = round(size * 1.15)
    w = round(len(text) * es * 0.6 + 28)
    x0 = x - w if anchor == "end" else (x - w / 2 if anchor == "middle" else x)
    f.add(f'<rect x="{x0:.0f}" y="{y}" width="{w}" height="{es + 15}" rx="{(es + 15) // 2}" fill="{fill or color}"/>')
    f.add(f'<text x="{round(x0 + w / 2)}" y="{y + es + 3}" font-size="{es}" font-weight="700" fill="{tcolor}" text-anchor="middle">{_fig.esc(text)}</text>')
    return w
    return w


def note_card(f, x, y, w, h, name, text=None, date=None, hot=False, tsize=22, bsize=17, stroke=None):
    d, t = NOTE[name]
    f.box(x, y, w, h, f"{name}.md", text if text is not None else t, fill=SOFT if hot else "#fff",
          stroke=stroke or (ACCENT if hot else LINE), tsize=tsize, bsize=bsize, pad=18, sw=3 if hot else 2)
    if date is not False:
        f.text(x + w - 18, y + 18 + tsize * 0.85, date or d, 17, 400, MUTED, "end")


def qbar(f, x, y, w, text, h=70):
    f.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{INK}"/>')
    f.text(x + 22, y + h / 2 + 7, "QUESTION", 17, 700, "#b3a9ff")
    f.text(x + 150, y + h / 2 + 9, text, 26, 600, "#fff")


def good_bad(f, x, y, w, h, good, bad, gsize=19):
    hh = (h - 24) / 2
    f.box(x, y, w, hh, "Good at", good, fill="#fff", stroke=HIT, tcolor=HIT, tsize=24, bsize=gsize, pad=18, sw=3)
    f.box(x, y + hh + 24, w, hh, "Weak at", bad, fill="#fff", stroke=RED, tcolor=RED, tsize=24, bsize=gsize, pad=18, sw=3)


def hbar(f, x, y, w, label, val, vmax, color=ACCENT, lw=360, size=24, fmt="{:.2f}", h=40, bold=False):
    f.text(x + lw - 16, y + h * 0.7, label, size, 700 if bold else 400, INK, "end")
    bw = max((w - lw - 110) * val / vmax, 4)
    f.add(f'<rect x="{x + lw}" y="{y}" width="{bw:.0f}" height="{h}" rx="8" fill="{color}"/>')
    f.text(round(x + lw + bw + 14), y + h * 0.72, fmt.format(val), size, 800, INK)


def vbar(f, x, base, h, w, val, vmax, label, color=ACCENT, vfmt="{:.2f}", lsize=20, sub=None):
    bh = h * val / vmax
    f.add(f'<rect x="{x}" y="{base - bh:.0f}" width="{w}" height="{bh:.0f}" rx="8" fill="{color}"/>')
    f.text(x + w / 2, base - bh - 12, vfmt.format(val), 28, 800, INK, "middle")
    lines = label if isinstance(label, list) else [label]
    f.lines(x + w / 2, base + 30, lines, lsize, INK2, round(lsize * 1.25), "middle", 600)


# ---------------------------------------------------------------- intro
def diary():
    f = F(678)
    for i, (name, d, t) in enumerate(NOTES):
        x, y = (i % 3) * 570, (i // 3) * 228
        note_card(f, x, y, 540, 200, name, tsize=24, bsize=18)
    return f


def memory():
    f = F(550)
    names = [("Ines", "a being who keeps a diary"), ("Odile", "another being, another diary"), ("An agent", "a program with a job")]
    for i, (who, what) in enumerate(names):
        x = 20 + i * 560
        f.box(x, 10, 500, 120, who, what, fill="#fff", stroke=INK, tsize=30, bsize=19, center=True, pad=16)
        f.arrow(x + 140, 136, x + 140, 262, ACCENT, 6)
        f.text(x + 120, 210, "retain", 24, 800, ACCENT, "end")
        f.arrow(x + 360, 262, x + 360, 136, HIT, 6)
        f.text(x + 380, 210, "recall", 24, 800, HIT)
        cyl(f, x, 270, 500, 170, [f"Bank: {who.lower()}"], size=28)
        f.text(x + 250, 424, "its own notes, never mixed with another bank's", 19, 400, INK2, "middle")
    f.box(0, 466, 1680, 68, fill=SOFT, stroke=SOFT, rx=14)
    f.text(840, 508, "Retain: write down what happened.   Recall: bring back what matters to the question asked.", 25, 700, ACCENT, "middle")
    return f


def vectors():
    f = F(550)
    f.box(0, 40, 440, 150, "A note's text", "Planted the Black Krim tomatoes in bed 3.", fill="#fff", stroke=INK, tsize=26, bsize=19, pad=20)
    f.arrow(444, 115, 560, 115, ACCENT, 6)
    f.text(502, 92, "embedding", 20, 700, ACCENT, "middle")
    f.text(502, 148, "model", 20, 700, ACCENT, "middle")
    f.box(566, 20, 400, 190, "A vector", ["[ 0.12, -0.80, 0.33, ... ]", "1,536 numbers: its place on a map of meaning"],
          fill=INK, stroke=INK, tcolor="#fff", bcolor="#c9cbe0", tsize=26, bsize=17, pad=20)
    f.box(0, 250, 966, 270, "Why a place on a map?", ["Texts about the same thing land close together; unrelated texts land far apart.",
          "Searching is then simple: put the question on the map and read off its nearest neighbours.",
          "The map is stored in Postgres, with the pgvector extension that finds the nearest points fast."],
          fill="#fff", stroke=LINE, tsize=26, bsize=19, pad=22)
    # the map
    mx, my, mw, mh = 1010, 0, 670, 540
    f.add(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="18" fill="{GREY}" stroke="{LINE}" stroke-width="2"/>')
    clusters = [("garden", 1200, 170, HIT, [(-60, -30), (-10, 20), (50, -10), (20, -60), (-50, 40)]),
                ("travel", 1500, 130, DETOUR, [(-30, -20), (30, 20), (0, 50)]),
                ("cooking", 1330, 420, MISSED, [(-50, -10), (20, 20), (60, -20), (-10, -50)])]
    for name, cx, cy, col, pts in clusters:
        f.add(f'<circle cx="{cx}" cy="{cy}" r="110" fill="{col}" opacity=".12"/>')
        for dx, dy in pts:
            f.add(f'<circle cx="{cx + dx}" cy="{cy + dy}" r="11" fill="{col}"/>')
        f.text(cx, cy + 108 if name != "garden" else cy - 118, name, 22, 700, col, "middle")
    f.add(f'<circle cx="1215" cy="150" r="17" fill="{ACCENT}" stroke="#fff" stroke-width="4"/>')
    f.arrow(1300, 300, 1230, 170, ACCENT, 4)
    f.text(1330, 322, "the question", 22, 800, ACCENT)
    f.text(1330, 348, "lands among its neighbours", 19, 400, INK2)
    return f


def timeline():
    f = F(550)
    f.box(0, 150, 360, 220, "Hindsight", ["the earlier memory service", "retired"], fill=GREY, stroke=MUTED, tcolor=MUTED, bcolor=MUTED,
          tsize=30, bsize=21, dash="10 8", center=True)
    f.arrow(366, 260, 520, 260, ACCENT, 7)
    f.box(526, 100, 520, 320, "Prospecta", ["one library for long-term memory", "Postgres with pgvector", "one bank per being",
          "write once, recall many ways"], fill=ACCENT, stroke=ACCENT, tcolor="#fff", bcolor="#e4e0ff", tsize=36, bsize=21, center=True)
    users = [("Forge", "moved onto Prospecta: about 72,000 notes", HIT), ("Augur", "moved onto Prospecta: about 3,800 notes", HIT),
             ("Rung provider", "rung-memory-prospecta: a thin adapter for the Rung agent runtime", DETOUR),
             ("Hermes provider", "hermes-prospecta: the plugin Forge and Augur use", DETOUR)]
    for i, (t, b, c) in enumerate(users):
        y = i * 136
        f.box(1200, y, 480, 128, t, b, fill="#fff", stroke=c, tsize=26, bsize=17, pad=16)
        f.arrow(1052, 260, 1194, y + 64, c, 4)
    return f


# ---------------------------------------------------------------- how it works
def retain():
    f = F(550)
    note_card(f, 0, 150, 420, 250, "yellow-leaves", date=False, tsize=26, bsize=20)
    f.text(210, 440, "one note arrives", 22, 700, INK2, "middle")
    f.arrow(424, 275, 520, 275, ACCENT, 7)
    f.box(526, 190, 250, 170, "Retain", "writes five things", fill=ACCENT, stroke=ACCENT, tcolor="#fff", bcolor="#e4e0ff",
          tsize=36, bsize=24, center=True)
    outs = [("1  Chunks", "pieces of at most 1,000 characters, each knowing its parent"),
            ("2  Anticipated questions", "an LLM writes what someone would ask to find this note"),
            ("3  Embeddings", "a 1,536-number vector for every chunk and question"),
            ("4  Entities, aliases, links", "an LLM names who and what appears, and how notes relate"),
            ("5  Metadata", "the date and the person, read from the note's header")]
    for i, (t, b) in enumerate(outs):
        y = i * 107
        f.box(900, y, 780, 112, t, b, fill="#fff", stroke=ACCENT, tsize=24, bsize=17, pad=14)
        f.arrow(780, 275, 894, y + 54, ACCENT, 3)
    return f


def chunks():
    f = F(550)
    f.box(0, 0, 1680, 100, "Parent note: yellow-leaves.md", "a note longer than one chunk is cut at paragraph breaks, with a small overlap so no sentence is lost",
          fill=GREY, stroke=MUTED, tsize=26, bsize=19, pad=20)
    texts = ["The lower tomato leaves have gone yellow with brown spots. Mr. Okafor looked over the fence and said early blight.",
             "I sprayed copper on the lower leaves and pruned the three worst branches, then bagged the cuttings.",
             "Plan: spray again after rain. Water at the root, never on the leaves. Check the neighbour's plants too."]
    for i, t in enumerate(texts):
        x = i * 570
        f.box(x, 190, 540, 190, f"Chunk {i + 1}", t, fill="#fff", stroke=ACCENT, tsize=26, bsize=19, pad=20)
        f.add(f'<path d="M{x + 270} 104 V186" stroke="{MUTED}" stroke-width="3" stroke-dasharray="7 6" fill="none"/>')
        f.text(x + 280, 164, "parent: yellow-leaves", 18, 700, MUTED)
        pill(f, x + 20, 396, "at most 1,000 characters", ACCENT, 18)
    for i in range(2):
        x = (i + 1) * 570 - 30
        f.add(f'<rect x="{x - 4}" y="240" width="38" height="90" rx="8" fill="{MISSED}" opacity=".35"/>')
    f.text(840, 470, "The orange strips are the 100-character overlap shared by neighbouring chunks.", 22, 700, INK2, "middle")
    f.text(840, 508, "A search hits a chunk; the answer cites the parent note.", 22, 400, INK2, "middle")
    return f


def questions():
    f = F(550)
    f.box(0, 140, 470, 250, "A chunk", "Lower tomato leaves yellow with brown spots. Mr. Okafor says early blight. Sprayed copper.",
          fill="#fff", stroke=INK, tsize=28, bsize=21, pad=22)
    f.arrow(474, 265, 600, 265, ACCENT, 7)
    f.box(606, 190, 220, 150, "An LLM", "reads it and asks", fill=INK, stroke=INK, tcolor="#fff", bcolor="#c9cbe0", tsize=30, bsize=21, center=True)
    qs = ["What was wrong with the tomato leaves?", "Who said it was early blight?", "What did I spray on the tomatoes?",
          "Why did the lower leaves turn yellow?"]
    for i, q in enumerate(qs):
        y = i * 100
        f.box(950, y, 730, 84, None, q, fill="#fff", stroke=ACCENT, bsize=20, pad=20, sw=3)
        f.arrow(830, 265, 944, y + 42, ACCENT, 3)
    f.text(1315, 436, "Each question is embedded too and points back to its chunk.", 21, 700, ACCENT, "middle")
    f.text(1315, 470, "The reader's question will be compared with these.", 21, 400, INK2, "middle")
    return f


def entities():
    f = F(678)
    notes = [("compost-skipper", 0), ("zine-column", 1), ("fence-rebuild", 2), ("yellow-leaves", 3), ("tomato-planting", 4)]
    ny = {}
    for n, i in notes:
        y = 10 + i * 124
        f.box(0, y, 420, 100, f"{n}.md", None, fill="#fff", stroke=LINE, tsize=24, pad=18)
        f.text(18, y + 76, NOTE[n][0], 19, 400, MUTED)
        ny[n] = y + 50
    ents = [("Dara", "person", 700, 90, ["Dari", "Skipper Compost", "Fern Aldous"], ["compost-skipper", "zine-column"]),
            ("Mr. Okafor", "person", 700, 400, ["Okafor"], ["fence-rebuild", "yellow-leaves", "tomato-planting"])]
    for name, kind, x, y, aliases, ns in ents:
        f.add(f'<circle cx="{x}" cy="{y + 60}" r="62" fill="{SOFT}" stroke="{ACCENT}" stroke-width="3"/>')
        f.text(x, y + 56, name, 24, 800, ACCENT, "middle")
        f.text(x, y + 84, kind, 17, 400, INK2, "middle")
        for n in ns:
            f.arrow(424, ny[n], x - 66, y + 60, ACCENT, 3)
        for k, a in enumerate(aliases):
            pill(f, x + 90, y - 10 + k * 46, f"alias: {a}", MISSED, 17)
    f.box(1040, 560, 640, 110, None, "Entities and aliases are extracted by an LLM when the note is written.", fill=GREY, stroke=LINE, bsize=19, pad=18)
    links = [("NEXT", "the following chunk of the same note"), ("ENTITY", "two notes name the same person or thing"),
             ("TEMPORALLY_CLOSE", "the same person wrote both close in time"),
             ("SEMANTIC · CAUSAL", "Jev judges: same subject? cause?")]
    f.text(1040, 330, "Links written between notes", 24, 800, INK)
    for i, (t, b) in enumerate(links):
        y = 350 + i * 52
        pw = pill(f, 1040, y, t, ACCENT if i < 3 else HIT, 16)
        f.text(1040 + pw + 14, y + 24, b, 16, 400, INK2)
    return f


def plan():
    f = F(550)
    steps = [("Question", "what the reader asks", "#fff", INK), ("Plan", "who and when it is about", "#fff", ACCENT),
             ("Channels", "five searches at once", ACCENT, ACCENT), ("Fuse", "merge the five lists", "#fff", ACCENT),
             ("Rerank", "an LLM reads the best notes", "#fff", ACCENT), ("Read, hop", "enough? else look again", "#fff", ACCENT),
             ("Answer", "grounded, with citations", INK, INK)]
    for i, (t, b, fill, st) in enumerate(steps):
        x = i * 245
        dark = fill in (ACCENT, INK)
        f.box(x, 60, 210, 200, f"{i + 1} {t}" if i not in (0, 6) else t, b, fill=fill, stroke=st, tcolor="#fff" if dark else INK,
              bcolor="#e4e0ff" if fill == ACCENT else ("#c9cbe0" if fill == INK else INK2), tsize=26, bsize=19, pad=16, center=True)
        if i < 6:
            f.arrow(x + 214, 160, x + 241, 160, ACCENT, 5)
    why = [(2 * 245, "dense, BM25, questions, metadata, graph"), (4 * 245, "Jev gate, then Sonnet"), (6 * 245, "or map-reduce for sets")]
    for x, t in why:
        f.text(min(x + 105, 1560), 296, t, 17, 700, INK2, "middle")
    f.box(0, 340, 820, 180, "Everything before step 5 is cheap", "Search in Postgres costs nothing but a fraction of a second. The language models are what cost money and time.",
          fill="#fff", stroke=LINE, tsize=26, bsize=20, pad=22)
    f.box(860, 340, 820, 180, "Every step is recorded", "What was asked, which channel found what, the reranker's grades and the answer are kept in Postgres tables, so any recall can be replayed.",
          fill="#fff", stroke=LINE, tsize=26, bsize=20, pad=22)
    return f


# ---------------------------------------------------------------- techniques
def dense():
    f = F(550)
    mx, my, mw, mh = 0, 0, 1000, 530
    f.add(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="18" fill="{GREY}" stroke="{LINE}" stroke-width="2"/>')
    pts = {"tomato-planting": (330, 200), "seed-order": (240, 290), "first-harvest": (420, 330), "yellow-leaves": (180, 150), "heat-wave": (110, 240),
           "lisbon-stay": (800, 120), "fence-rebuild": (640, 430), "compost-skipper": (760, 360), "zine-column": (870, 250)}
    for n, (x, y) in pts.items():
        hot = n in ("tomato-planting", "seed-order", "first-harvest")
        f.add(f'<circle cx="{x}" cy="{y}" r="12" fill="{ACCENT if hot else MUTED}"/>')
        f.text(x + 18, y + 6, n, 18, 600 if hot else 400, INK if hot else INK2)
    qx, qy = 300, 235
    f.add(f'<circle cx="{qx}" cy="{qy}" r="150" fill="none" stroke="{ACCENT}" stroke-width="3" stroke-dasharray="9 7"/>')
    f.add(f'<circle cx="{qx}" cy="{qy}" r="14" fill="{RED}" stroke="#fff" stroke-width="4"/>')
    f.text(qx - 150, 36, "Question: when did the dark-fruited heirloom vines go into the ground?", 21, 800, RED)
    f.text(24, 520, "No shared words with the note, yet its point is the nearest.", 20, 700, INK2)
    good_bad(f, 1040, 0, 640, 530, ["Meaning: finds the note even when no word matches.", "Paraphrases, loose descriptions, questions."],
             ["Exact names, numbers, codes: close is not the same as equal.", "Dates and who: a note from the wrong month can sit nearby."])
    return f


def bm25():
    f = F(550)
    f.text(0, 36, "Question: Okafor thermometer, as the notes' own words", 24, 800, INK)
    rows = [("the", 9, 0.06), ("Okafor", 3, 1.05), ("thermometer", 1, 1.90)]
    f.text(20, 90, "word", 20, 700, MUTED)
    f.text(230, 90, "notes holding it", 20, 700, MUTED)
    f.text(480, 90, "weight (rarer is higher)", 20, 700, MUTED)
    for i, (w, n, wt) in enumerate(rows):
        y = 110 + i * 70
        f.text(20, y + 32, w, 26, 700, INK)
        f.text(230, y + 32, f"{n} of 9", 26, 400, INK2)
        f.add(f'<rect x="480" y="{y + 6}" width="{max(wt * 220, 6):.0f}" height="40" rx="8" fill="{ACCENT}"/>')
        f.text(480 + wt * 220 + 14, y + 36, f"{wt:.2f}", 22, 800, INK)
    cards = [("Rare words weigh more", "A word in one note out of nine says a lot. A word in every note says nothing."),
             ("Repeats saturate", "Saying a word ten times counts only a little more than saying it three times."),
             ("Short notes count more", "A hit in a short note is stronger than the same hit lost in a long one.")]
    for i, (t, b) in enumerate(cards):
        f.box(i * 340, 330, 320, 210, t, b, fill="#fff", stroke=LINE, tsize=22, bsize=17, pad=16)
    f.box(0, 330 - 0, 0, 0)
    good_bad(f, 1040, 0, 640, 530, ["Exact names, numbers, codes and rare words.", "Needs no model at all: free and instant."],
             ["Paraphrase: a question with none of the note's words finds nothing.", "Common words mislead it: the word 'first' can pull the wrong note."])
    return f


def qchannel():
    f = F(550)
    f.box(0, 0, 640, 110, "The question", "Why did my tomatoes get sick?", fill=INK, stroke=INK, tcolor="#fff", bcolor="#fff", tsize=22, bsize=26, pad=22)
    rows = [("What was wrong with the tomato leaves?", "yellow-leaves", 0.81, True),
            ("What did I spray on the tomatoes?", "yellow-leaves", 0.66, False),
            ("When were the tomatoes planted?", "tomato-planting", 0.58, False),
            ("How many kilos did the harvest give?", "first-harvest", 0.31, False)]
    f.text(0, 160, "Anticipated questions written at retain time, compared with it", 21, 700, INK2)
    for i, (q, n, s, hot) in enumerate(rows):
        y = 180 + i * 88
        f.box(0, y, 590, 76, None, q, fill=SOFT if hot else "#fff", stroke=ACCENT if hot else LINE, bsize=18, pad=16, sw=3 if hot else 2)
        f.text(604, y + 46, f"{s:.2f}", 24, 800, ACCENT if hot else INK2)
        f.arrow(672, y + 38, 712, y + 38, ACCENT if hot else MUTED, 3)
        f.text(722, y + 46, f"{n}.md", 19, 600 if hot else 400, INK if hot else INK2)
    good_bad(f, 1040, 0, 640, 530, ["The reader's words differ from the writer's.", "Costs nothing at question time: the work was done when the note was written."],
             ["The LLM may anticipate the wrong questions: alone it reaches 0.74 hit@10 against 0.92 for chunks.", "Needs an LLM call for every note written."])
    return f


def meta():
    f = F(550)
    qbar(f, 0, 0, 1000, "What went wrong in the garden in July 2025?", 76)
    f.box(0, 100, 480, 210, "Plan reads it", ["people: none", "date_from: 2025-07-01", "date_to: 2025-07-31", "hard: yes"], fill="#fff", stroke=ACCENT, tsize=24, bsize=17, pad=18)
    f.arrow(484, 205, 560, 205, ACCENT, 5)
    months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    f.text(580, 140, "The diary by month", 21, 700, INK2)
    for i, m in enumerate(months):
        x = 570 + i * 62
        hot = m == "Jul"
        f.add(f'<rect x="{x}" y="160" width="56" height="140" rx="8" fill="{SOFT if hot else "#fff"}" stroke="{ACCENT if hot else LINE}" stroke-width="{3 if hot else 2}"/>')
        f.text(x + 28, 322, m, 18, 700 if hot else 400, ACCENT if hot else INK2, "middle")
    f.text(570 + 4 * 62 + 28, 232, "2", 34, 800, ACCENT, "middle")
    f.text(570 + 4 * 62 + 28, 260, "notes", 14, 700, ACCENT, "middle")
    f.box(0, 360, 1000, 170, "Scope, never exclusion", "The notes inside the filter get a score boost; no note is ever removed. Two of 19 extractions would have thrown away their own answer if used as a hard filter. Promotion is off by default.",
          fill="#fff", stroke=LINE, tsize=24, bsize=19, pad=20)
    good_bad(f, 1040, 0, 640, 530, ["Who and when: scoped questions 0.70 to 1.00 hit@10.", "Free: a column lookup in Postgres."],
             ["Only as good as the extraction: a wrong date or name misleads.", "Needs metadata: notes without a date or person cannot be scoped."])
    return f


def graph():
    f = F(550)
    f.text(0, 30, "Seed: yellow-leaves, found by the other channels (score 1.00)", 21, 800, INK)
    f.add(f'<circle cx="90" cy="265" r="70" fill="{ACCENT}"/>')
    f.text(90, 260, "yellow-", 21, 800, "#fff", "middle")
    f.text(90, 288, "leaves", 21, 800, "#fff", "middle")
    hop1 = [("fence-rebuild", "ENTITY: Mr. Okafor", 80, 0.5 * 0.5), ("tomato-planting", "SEMANTIC", 230, 1.0 * 0.5), ("heat-wave", "TEMPORAL: close in time", 380, 0.5 * 0.5)]
    for n, why, y, sc in hop1:
        f.arrow(164, 265, 236, y + 34, ACCENT, 4)
        f.box(244, y, 300, 66, None, n, fill="#fff", stroke=ACCENT, bsize=20, pad=16, sw=3)
        f.text(560, y + 30, f"hop 1 · {sc:.2f}", 20, 800, ACCENT)
        f.text(560, y + 58, why, 17, 400, INK2)
    f.arrow(700, 270, 770, 300, MUTED, 3)
    f.box(776, 270, 224, 62, None, "first-harvest", fill="#fff", stroke=MUTED, bsize=18, pad=14)
    f.text(776, 360, "hop 2 · 0.12", 17, 400, MUTED)
    f.box(0, 458, 1000, 82, None, "score = seed score x decay 0.5 x link-type weight x link confidence. Weights: SEMANTIC 1.0, CAUSAL 0.8, TEMPORAL 0.5, ENTITY 0.5.", fill=GREY, stroke=LINE, bsize=16, pad=16)
    good_bad(f, 1040, 0, 640, 530, ["Finds neighbours no word or meaning match reaches: the same person, an earlier cause.", "Cheap to run: about 0.2 seconds in the latest build."],
             ["Measured gain: 0.00 to 0.01 hit@10. It found a note nothing else found for 2 of 264.", "Costs an LLM at write time; a hub person links to hundreds of notes."], gsize=18)
    return f


def jev():
    f = F(550)
    qbar(f, 0, 0, 1000, "What did Mr. Okafor say was wrong with the tomatoes?", 80)
    rows = [("yellow-leaves", 3, "Holds the answer"), ("tomato-planting", 1, "Same subject only"), ("lisbon-stay", 0, "Unrelated")]
    for i, (n, s, lab) in enumerate(rows):
        y = 120 + i * 112
        f.box(0, y, 560, 92, f"{n}.md", None, fill="#fff", stroke=LINE, tsize=24, pad=20)
        f.arrow(566, y + 46, 680, y + 46, ACCENT, 4)
        col = HIT if s == 3 else (MISSED if s == 1 else MUTED)
        f.add(f'<circle cx="740" cy="{y + 46}" r="40" fill="{col}"/>')
        f.text(740, y + 58, str(s), 38, 800, "#fff", "middle")
        f.text(800, y + 54, lab, 22, 600, INK)
    f.box(0, 460, 1000, 80, None, "Jev, a small model made to judge relevance, answers a 0 to 3 question per candidate. About $0.0003 a call.", fill=GREY, stroke=LINE, bsize=19, pad=18)
    f.box(1040, 0, 640, 280, "Where Jev works here", ["judges which notes to link at write time", "scores candidates in the reranker (hit@1 0.73 alone)", "gates Sonnet: a top score of 2.95 or more skips it"],
          fill="#fff", stroke=ACCENT, tsize=26, bsize=19, pad=20)
    f.box(1040, 300, 640, 240, "Why not use Jev for everything", ["It reorders what others found; it never invents a candidate.", "As the only reranker it is 40 times cheaper than Sonnet but reaches 0.73 hit@1 against 0.87."],
          fill="#fff", stroke=LINE, tsize=26, bsize=19, pad=20)
    return f


def fusion():
    f = F(550)
    k = 60
    lists = {"N1": (4, 1, 1), "N2": (1, 3, 2), "N6": (2, 2, None), "N5": (3, None, None), "N3": (None, None, 3)}
    names = {"N1": "seed-order", "N2": "tomato-planting", "N6": "first-harvest", "N5": "yellow-leaves", "N3": "fence-rebuild"}
    wts = (4, 1, 1)
    def score(r, w):
        return sum(wi / (k + ri) for wi, ri in zip(w, r) if ri)
    rowsd = []
    for key, r in lists.items():
        rowsd.append((names[key], r, score(r, (1, 1, 1)), score(r, wts)))
    eq_order = [n for n, *_ in sorted(rowsd, key=lambda t: -t[2])]
    wt_order = [n for n, *_ in sorted(rowsd, key=lambda t: -t[3])]
    xs = [0, 330, 450, 570, 740, 940, 1130]
    heads = ["note", "dense", "BM25", "questions", "equal weights", "weights 4, 1, 1"]
    hx = [0, 350, 470, 590, 760, 1000]
    for h, x in zip(heads, hx):
        f.text(x, 30, h, 20, 700, MUTED)
    f.text(350 + 0, 56, "rank in each channel", 16, 400, MUTED)
    for i, (n, r, se, sw) in enumerate(sorted(rowsd, key=lambda t: -t[3])):
        y = 70 + i * 72
        hot = n == "tomato-planting"
        f.add(f'<rect x="-6" y="{y}" width="1330" height="62" rx="12" fill="{SOFT if hot else "#fff"}" stroke="{ACCENT if hot else LINE}" stroke-width="2"/>')
        f.text(12, y + 40, n, 24, 800 if hot else 600, INK)
        for j, rk in enumerate(r):
            f.text(hx[1 + j] + 20, y + 40, f"#{rk}" if rk else "-", 24, 400, INK2 if rk else MUTED)
        f.text(hx[4], y + 40, f"{se:.4f}", 24, 400, INK2)
        f.text(hx[5], y + 40, f"{sw:.4f}", 24, 800, ACCENT)
    f.text(0, 462, "score = sum over channels of weight / (60 + rank).  A rank counts for less the lower it sits.", 22, 700, INK, "start")
    f.text(0, 500, f"Equal weights: {eq_order[0]} and {eq_order[1]} nearly tie, and {eq_order[0]} wins.   Weighted: {wt_order[0]} wins.", 22, 400, INK2, "start")
    f.box(1360, 70, 320, 360, "The weights", ["dense chunks 4", "metadata 3", "BM25 1", "questions 1", "graph 1"], fill="#fff", stroke=ACCENT, tsize=26, bsize=21, pad=20)
    f.box(1360, 450, 320, 90, None, "k = 60", fill=GREY, stroke=LINE, bsize=22, pad=20, center=True)
    return f


def rerank():
    f = F(550)
    f.box(0, 100, 380, 300, "The pool", ["every fused note scoring at least 0.15 times the best", "about 385 notes in the latest round"], fill="#fff", stroke=LINE, tsize=28, bsize=20, pad=22)
    f.arrow(384, 250, 480, 250, ACCENT, 6)
    f.box(486, 60, 470, 380, "Sonnet reads, in one go", ["one best chunk per note, in full, under a header: note name, date, person", "replies in JSON: a grade and a ranking",
          "if it fails, the fused order stands and the reason is recorded"], fill=ACCENT, stroke=ACCENT, tcolor="#fff", bcolor="#e4e0ff", tsize=28, bsize=20, pad=24)
    f.arrow(960, 250, 1040, 250, ACCENT, 6)
    before = ["first-harvest", "seed-order", "tomato-planting"]
    after = ["tomato-planting", "first-harvest", "seed-order"]
    f.text(1050, 60, "Before", 22, 700, MUTED)
    f.text(1370, 60, "After", 22, 700, ACCENT)
    for i, n in enumerate(before):
        f.box(1040, 80 + i * 70, 290, 62, None, f"{i + 1}  {n}", fill="#fff", stroke=LINE, bsize=17, pad=12)
    for i, n in enumerate(after):
        f.box(1370, 80 + i * 70, 290, 62, None, f"{i + 1}  {n}", fill=SOFT if i == 0 else "#fff", stroke=ACCENT if i == 0 else LINE, bsize=17, pad=12, sw=3 if i == 0 else 2)
    f.text(1040, 330, "hit@1: 0.72 to 0.87", 28, 800, ACCENT)
    f.text(1040, 366, "hit@10: 0.93 to 0.97", 28, 800, ACCENT)
    f.text(1040, 410, "$0.021 a question for a pool of 30, about $0.20 now", 20, 400, INK2)
    f.text(1040, 440, "the biggest single step after the embedder", 20, 700, INK2)
    return f


def gate():
    f = F(550)
    f.box(0, 100, 300, 300, "Jev scores", ["the whole pool, in parallel batches", "about half a second"], fill="#fff", stroke=ACCENT, tsize=28, bsize=20, pad=22)
    f.arrow(304, 250, 400, 250, ACCENT, 6)
    f.add(f'<path d="M560 130 L720 250 L560 370 L400 250 Z" fill="{SOFT}" stroke="{ACCENT}" stroke-width="3"/>')
    f.text(560, 240, "top score", 22, 700, ACCENT, "middle")
    f.text(560, 270, "2.95 or more?", 22, 700, ACCENT, "middle")
    f.arrow(720, 250, 900, 250, HIT, 6)
    f.text(810, 232, "yes", 22, 800, HIT, "middle")
    f.box(906, 150, 340, 200, "Keep Jev's order", ["no Sonnet call", "34 of 121 questions"], fill="#fff", stroke=HIT, tcolor=HIT, tsize=28, bsize=21, pad=22)
    f.arrow(560, 374, 560, 450, RED, 6)
    f.text(580, 424, "no", 22, 800, RED)
    f.box(380, 454, 360, 90, "Sonnet reranks", None, fill=ACCENT, stroke=ACCENT, tcolor="#fff", tsize=28, pad=22, center=True)
    f.box(1300, 90, 380, 440, "Why a gate", ["Jev is about 40 times cheaper than Sonnet.", "When Jev is sure, its order matched Sonnet's hit@1.", "The 2.95 threshold was set on these same questions, so the saving is in-sample."], fill="#fff", stroke=LINE, tsize=26, bsize=19, pad=20)
    return f


def blend():
    f = F(550)
    f.text(0, 30, "The blend: score = 0.7 x reranker position + 0.3 x fused score", 24, 800, INK)
    f.box(0, 60, 760, 210, "Tuned for a pool of 30", ["The reranker's first and second choice differ by 0.7 / 29 = 0.024 per place.", "Enough to matter against the fused term."], fill="#fff", stroke=HIT, tcolor=HIT, tsize=24, bsize=20, pad=20)
    f.box(0, 290, 760, 240, "Run on a pool of about 385", ["The same step is 0.7 / 384 = 0.002.", "The fused score now decides, and the reranker only breaks ties.", "Switched off by default; the code stays for tests."], fill="#fff", stroke=RED, tcolor=RED, tsize=24, bsize=20, pad=20)
    base = 480
    f.text(1000, 30, "hit@1 on 120 questions", 22, 800, INK)
    items = [("blend on", 0.72, RED), ("blend off", 0.86, HIT), ("both re-sorts off", 0.86, HIT)]
    # only two bars: with blend and promotion on versus off
    vbar(f, 960, base, 380, 200, 0.72, 1.0, ["blend and", "promotion on"], RED)
    vbar(f, 1260, base, 380, 200, 0.86, 1.0, ["both off", "(now the default)"], HIT)
    return f


def hop():
    f = F(550)
    f.box(0, 120, 330, 170, "Top of the list", "after reranking", fill="#fff", stroke=LINE, tsize=26, bsize=21, pad=20)
    f.arrow(334, 205, 420, 205, ACCENT, 6)
    f.box(426, 40, 340, 330, "The reader asks", ["Do these notes hold enough to answer?", "Answers: sufficient, or one follow-up per missing fact"], fill=ACCENT, stroke=ACCENT, tcolor="#fff", bcolor="#e4e0ff", tsize=26, bsize=20, pad=20)
    f.arrow(770, 150, 900, 90, HIT, 5)
    f.text(800, 100, "sufficient", 20, 800, HIT, "middle")
    f.box(906, 30, 330, 110, "Done", "most questions", fill="#fff", stroke=HIT, tcolor=HIT, tsize=26, bsize=20, pad=18)
    f.arrow(770, 260, 900, 320, RED, 5)
    f.text(830, 346, "missing", 20, 800, RED, "middle")
    f.box(906, 200, 774, 170, "Follow-up search, cheap channels only", "Missing: how much did the harvest give?  Dense, BM25 and question channels run again. New notes join and the joined set is reranked once.", fill="#fff", stroke=RED, tcolor=RED, tsize=24, bsize=19, pad=18)
    f.box(0, 400, 1680, 130, None, "At most one hop. In the latest rounds the reader asked for a follow-up on 9 of 121 ordinary questions and 5 of 21 set questions; the hop lifted hit@1 by about 0.03 and set cover by 0.04.", fill=GREY, stroke=LINE, bsize=22, pad=24)
    return f


def depth():
    f = F(550)
    pts = []
    for i in range(0, 101):
        x = 20 + i * 8.0
        y = 470 - 400 * math.exp(-i / 9.0)
        pts.append((x, y))
    f.add(f'<polyline points="{" ".join(f"{x:.0f},{y:.0f}" for x, y in pts)}" fill="none" stroke="{ACCENT}" stroke-width="5"/>')
    f.add('<line x1="20" y1="470" x2="830" y2="470" stroke="#8a8f9c" stroke-width="2"/>')
    f.text(20, 508, "notes, best fused score first", 19, 400, MUTED)
    for frac, label, col in ((0.15, "standard: 0.15 x best", HIT), (0.05, "deep: 0.05 x best", RED)):
        y = 470 - 400 * frac
        f.add(f'<line x1="20" y1="{y:.0f}" x2="830" y2="{y:.0f}" stroke="{col}" stroke-width="3" stroke-dasharray="9 7"/>')
        f.text(830, y - 10, label, 20, 800, col, "end")
    f.text(300, 120, "Everything above a line goes to the reranker.", 21, 700, INK2)
    rows = [("pool cut", "0.15", "0.05"), ("notes reranked", "about 385", "about 1,100"), ("cost a question", "about $0.22", "about $0.83"),
            ("used for", "automatic recall", "explicit recall, sets")]
    f.text(1250, 40, "standard", 24, 800, HIT, "middle")
    f.text(1520, 40, "deep", 24, 800, RED, "middle")
    for i, (a, b, c) in enumerate(rows):
        y = 60 + i * 110
        f.box(880, y, 800, 94, None, None, fill="#fff", stroke=LINE, sw=2)
        f.text(900, y + 56, a, 22, 700, INK)
        f.text(1250, y + 56, b, 22, 400, INK2, "middle")
        f.text(1520, y + 56, c, 22, 400, INK2, "middle")
    return f


def synth():
    f = F(550)
    qbar(f, 0, 0, 760, "How did the tomatoes do?", 90)
    ev = ["first-harvest", "yellow-leaves", "heat-wave"]
    f.text(0, 168, "Evidence handed over, whole", 21, 700, INK2)
    for i, n in enumerate(ev):
        f.box(0, 186 + i * 112, 500, 110, f"{n}.md", NOTE[n][1][:70] + "...", fill="#fff", stroke=LINE, tsize=22, bsize=16, pad=16)
        f.arrow(504, 234 + i * 112, 600, 290, ACCENT, 3)
    f.box(606, 160, 220, 260, "Sonnet writes", ["from the evidence only", "cites every claim"], fill=ACCENT, stroke=ACCENT, tcolor="#fff", bcolor="#e4e0ff", tsize=26, bsize=19, pad=18, center=True)
    f.arrow(830, 285, 910, 285, ACCENT, 6)
    f.box(916, 90, 764, 300, "Answer", ["Nine kilos of Black Krim came in on 14 August [first-harvest]. Trouble started in July: early blight on the lower leaves, treated with copper [yellow-leaves], and a heat wave that bolted the basil [heat-wave]."],
          fill="#fff", stroke=ACCENT, tsize=26, bsize=21, pad=24)
    f.box(916, 410, 764, 132, None, "If the evidence lacks it, the answer is 'not in memory'. A cited name that is not in the evidence is flagged.", fill=GREY, stroke=LINE, bsize=20, pad=22)
    return f


def mapreduce():
    f = F(550)
    qbar(f, 0, 0, 1680, "What are Dara's nicknames?", 70)
    f.box(0, 110, 330, 250, "1 Resolve", ["Dara is an entity, with aliases Dari, Skipper Compost, Fern Aldous"], fill="#fff", stroke=ACCENT, tsize=26, bsize=19, pad=20)
    f.arrow(334, 225, 400, 225, ACCENT, 5)
    f.box(406, 110, 330, 250, "2 Gather", ["every note tied to Dara, not just the top ten", "read in batches of eight"], fill="#fff", stroke=ACCENT, tsize=26, bsize=19, pad=20)
    f.arrow(740, 225, 806, 225, ACCENT, 5)
    f.box(812, 110, 400, 270, "3 Map", ["each batch: pull out the asked facts, each with its note", "Dari [compost-skipper]", "Fern Aldous [zine-column]"], fill="#fff", stroke=ACCENT, tsize=26, bsize=18, pad=20)
    f.arrow(1216, 225, 1282, 225, ACCENT, 5)
    f.box(1288, 110, 392, 250, "4 Reduce", ["merge, drop repeats, keep citations", "Dari, Skipper Compost, and the pen name Fern Aldous"], fill=ACCENT, stroke=ACCENT, tcolor="#fff", bcolor="#e4e0ff", tsize=26, bsize=19, pad=20)
    f.box(0, 390, 820, 160, "Why not a top-ten list?", "A set has no 'best ten'. Search ranks by closeness, so a note that names a nickname without the word 'nickname' sits far down.", fill="#fff", stroke=LINE, tsize=24, bsize=19, pad=20)
    f.box(860, 390, 820, 160, "On the real notes", "A question like this found all 5 names with citations, for $0.42 and 98 seconds. Map-reduce costs a model call per batch.", fill="#fff", stroke=LINE, tsize=24, bsize=19, pad=20)
    return f


def resolver():
    f = F(550)
    f.text(0, 36, "A name matches when every core word of the shorter name appears in the longer", 23, 800, INK)
    rows = [("Okafor", "Mr. Okafor", True, "'Mr.' is dropped, one word left matches"), ("okafor, mr", "Mr. Okafor", True, "order and case do not matter"),
            ("Oka", "Mr. Okafor", True, "a 3-letter start of a word matches"), ("Dari", "Dara", True, "an alias row points to its person"),
            ("Ofak", "Mr. Okafor", False, "letters out of order: no match")]
    f.text(0, 86, "you search", 18, 700, MUTED)
    f.text(330, 86, "stored as", 18, 700, MUTED)
    for i, (q, s, ok, why) in enumerate(rows):
        y = 100 + i * 78
        f.box(0, y, 300, 62, None, q, fill="#fff", stroke=LINE, bsize=24, pad=18)
        f.arrow(304, y + 31, 380, y + 31, HIT if ok else RED, 4)
        f.box(386, y, 300, 62, None, s, fill="#fff", stroke=LINE, bsize=24, pad=18)
        f.text(710, y + 40, "match" if ok else "no match", 22, 800, HIT if ok else RED)
        f.text(850, y + 40, why, 20, 400, INK2)
    f.box(1260, 100, 420, 390, "On the real notes", ["A question about one person, stored as 'Mr. <surname>', first found 0 candidate notes.", "With the resolver: 22 notes, all 4 answers among them."], fill="#fff", stroke=ACCENT, tsize=26, bsize=20, pad=22)
    return f


# ---------------------------------------------------------------- scoring
def score():
    f = F(550)
    f.text(0, 34, "One question, the top ten notes returned. Gold notes (the right answers) are green.", 23, 700, INK)
    golds = {2, 5, 9}
    for i in range(10):
        x = i * 168
        gold = (i + 1) in golds
        f.box(x, 60, 150, 110, None, None, fill="#e3f4ec" if gold else "#fff", stroke=HIT if gold else LINE, sw=3 if gold else 2)
        f.text(x + 75, 130, str(i + 1), 44, 800, HIT if gold else MUTED, "middle")
    f.text(0, 200, "A fourth gold note sits at place 14: outside the top ten.", 21, 700, RED)
    cards = [("hit@1", "Is a gold note first?", "Here: no. Place 1 is not gold.", "0"),
             ("hit@10", "Is any gold note in the top ten?", "Here: yes. Places 2, 5 and 9.", "1"),
             ("cover@10", "What share of all gold notes is in the top ten?", "Here: 3 of 4 gold notes.", "0.75")]
    for i, (t, q, a, v) in enumerate(cards):
        x = i * 570
        f.box(x, 250, 540, 280, t, [q, a], fill="#fff", stroke=ACCENT, tsize=36, bsize=21, pad=24)
        f.text(x + 500, 310, v, 60, 800, ACCENT, "end")
    return f


def score_example():
    f = F(678)
    hdr = ["question", "gold at places", "hit@1", "hit@10", "cover@10"]
    hx = [0, 560, 900, 1100, 1330]
    for h, x in zip(hdr, hx):
        f.text(x, 30, h, 20, 700, MUTED)
    rows = [("A  When were the tomatoes planted?", "1", 1, 1, "1.00"), ("B  What went wrong in July?", "4", 0, 1, "1.00"),
            ("C  What are Dara's nicknames? (2 gold... 4 here)", "2, 5, 9, 14", 0, 1, "0.75"), ("D  Which kale did I order?", "23", 0, 0, "0.00")]
    rows[2] = ("C  Everything about Mr. Okafor", "2, 5, 9, 14", 0, 1, "0.75")
    for i, (q, g, a, b, c) in enumerate(rows):
        y = 50 + i * 96
        f.add(f'<rect x="-6" y="{y}" width="1690" height="82" rx="12" fill="#fff" stroke="{LINE}" stroke-width="2"/>')
        f.text(12, y + 52, q, 24, 600, INK)
        f.text(hx[1], y + 52, g, 26, 400, INK2)
        f.text(hx[2], y + 52, str(a), 30, 800, HIT if a else RED)
        f.text(hx[3], y + 52, str(b), 30, 800, HIT if b else RED)
        f.text(hx[4], y + 52, c, 30, 800, INK)
    y = 50 + 4 * 96
    f.add(f'<rect x="-6" y="{y}" width="1690" height="82" rx="12" fill="{SOFT}" stroke="{ACCENT}" stroke-width="3"/>')
    f.text(12, y + 52, "Average of the four", 26, 800, ACCENT)
    f.text(hx[2], y + 52, "0.25", 30, 800, ACCENT)
    f.text(hx[3], y + 52, "0.75", 30, 800, ACCENT)
    f.text(hx[4], y + 52, "0.69", 30, 800, ACCENT)
    f.text(0, 540, "hit@1 is 1 of 4. hit@10 is 3 of 4. cover@10 is (1 + 1 + 0.75 + 0) / 4.", 24, 700, INK, "start")
    f.text(0, 580, "A set question (C) has several gold notes, so only cover@10 shows how much of the set came back.", 22, 400, INK2)
    f.text(0, 620, "With 100 questions, one question is 0.01: read only differences of 0.03 or more.", 22, 400, INK2)
    return f


# ---------------------------------------------------------------- walk-throughs
def walk(query, gold, chans, final, note, hot_chan=None, h=678):
    """Five channel strips of ranked notes and the fused list. Gold notes are green."""
    f = F(h)
    qbar(f, 0, 0, 1680, query, 76)
    n = len(chans)
    cw = 310
    gap = (1680 - cw * n) / (n - 1) if n > 1 else 0
    for i, (name, w, items, why) in enumerate(chans):
        x = round(i * (cw + gap))
        hot = name == hot_chan
        f.box(x, 100, cw, 54, None, None, fill=ACCENT if hot else SOFT, stroke=ACCENT, sw=2)
        f.text(x + 16, 136, name, 22, 800, "#fff" if hot else ACCENT)
        f.text(x + cw - 16, 136, f"weight {w}", 16, 400, "#e4e0ff" if hot else INK2, "end")
        if not items:
            f.box(x, 164, cw, 250, None, why, fill="#fff", stroke=LINE, bsize=19, pad=16)
            continue
        for j, it in enumerate(items):
            y = 164 + j * 62
            g = it in gold
            f.box(x, y, cw, 54, None, None, fill="#e3f4ec" if g else "#fff", stroke=HIT if g else LINE, sw=3 if g else 2, rx=10)
            f.text(x + 14, y + 36, f"{j + 1}", 19, 800, HIT if g else MUTED)
            f.text(x + 44, y + 36, it, 19, 700 if g else 400, INK)
        f.text(x + 4, 164 + len(items) * 62 + 24, why, 15, 400, INK2)
    f.arrow(840, 412, 840, 450, ACCENT, 6)
    f.text(870, 440, "fuse, then rerank", 20, 800, ACCENT)
    f.text(0, 516, "Result", 21, 800, INK)
    for j, it in enumerate(final):
        g = it in gold
        x = 120 + j * 330
        f.box(x, 480, 310, 56, None, None, fill="#e3f4ec" if g else "#fff", stroke=HIT if g else LINE, sw=3 if g else 2, rx=10)
        f.text(x + 14, 516, f"{j + 1}", 19, 800, HIT if g else MUTED)
        f.text(x + 44, 516, it, 19, 700 if g else 400, INK)
    f.box(0, 552, 1680, h - 552, None, note, fill=GREY, stroke=LINE, bsize=19, pad=18)
    return f


def q_lookup():
    return walk("What temperature was the soil when I planted the Black Krim tomatoes?", {"tomato-planting"},
                [("Dense", 4, ["tomato-planting", "seed-order", "first-harvest", "yellow-leaves"], "meaning matches"),
                 ("BM25", 1, ["tomato-planting", "seed-order", "first-harvest"], "Black Krim, soil, planted"),
                 ("Questions", 1, ["seed-order", "tomato-planting", "yellow-leaves"], ""),
                 ("Metadata", 3, [], "no person or date in the question: abstains"),
                 ("Graph", 1, ["fence-rebuild", "yellow-leaves"], "neighbours of the seeds")],
                ["tomato-planting", "seed-order", "first-harvest"],
                "Every channel agrees. BM25 and dense both nail it because the question carries the note's own rare words. The answer: 16 degrees [tomato-planting]. Hit at place 1.",
                hot_chan="BM25", h=678)


def q_para():
    return walk("When did the dark-fruited heirloom vines first go into the ground?", {"tomato-planting"},
                [("Dense", 4, ["seed-order", "first-harvest", "tomato-planting", "compost-skipper"], "meaning: right area, gold third"),
                 ("BM25", 1, ["first-harvest", "zine-column"], "only 'first' matches: wrong notes"),
                 ("Questions", 1, ["tomato-planting", "seed-order"], "'When were the tomatoes planted?'"),
                 ("Metadata", 3, [], "nothing to scope on: abstains"),
                 ("Graph", 1, ["fence-rebuild"], "")],
                ["tomato-planting", "seed-order", "first-harvest"],
                "No word is shared, so BM25 is lost and even leads astray. Dense and the question channel carry it, and the Sonnet reranker reads for meaning and puts the gold note first. Paraphrases went from 0.75 fused to 0.95 reranked, then 1.00 after the hop.",
                hot_chan="Questions", h=678)


def q_time():
    return walk("What went wrong in the garden in July 2025?", {"yellow-leaves", "heat-wave"},
                [("Dense", 4, ["yellow-leaves", "first-harvest", "tomato-planting", "heat-wave"], "meaning, but July is just a word"),
                 ("BM25", 1, ["fence-rebuild", "compost-skipper"], "no word says July: noise"),
                 ("Questions", 1, ["yellow-leaves", "heat-wave"], ""),
                 ("Metadata", 3, ["yellow-leaves", "heat-wave"], "the two July notes"),
                 ("Graph", 1, ["tomato-planting"], "")],
                ["yellow-leaves", "heat-wave", "first-harvest"],
                "The plan reads 'July 2025' as a date range and the metadata channel returns exactly the notes written then. Its weight of 3 lifts both July notes above the August harvest that dense search liked. Scoped questions: 0.70 for dense alone, 1.00 after the filter.",
                hot_chan="Metadata", h=678)


def q_set():
    return walk("What are Dara's nicknames?", {"compost-skipper", "zine-column"},
                [("Dense", 4, ["compost-skipper", "fence-rebuild", "lisbon-stay", "seed-order"], "the word 'nickname' pulls one note"),
                 ("BM25", 1, ["compost-skipper", "zine-column"], "both notes name Dara"),
                 ("Questions", 1, ["compost-skipper", "heat-wave"], ""),
                 ("Metadata", 3, [], "Dara is not the diary's person: abstains"),
                 ("Graph", 1, ["zine-column", "compost-skipper"], "entity Dara and its aliases")],
                ["compost-skipper", "zine-column", "fence-rebuild"],
                "BM25 and the graph carry this one, because only those see the name Dara. The pen name note never says 'nickname', so a top-ten list can bury it. Sets go to map-reduce: read every note tied to Dara. Set cover@10 rose from 0.60 to 0.76 over the rounds.",
                hot_chan="Graph", h=678)


def heat():
    f = F(550)
    cols = ["Dense", "BM25", "Questions", "Metadata", "Graph", "Full pipeline"]
    rows = [("Precise lookup", "hit@10", [0.90, 1.00, 0.70, 0.45, 0.10, 1.00]),
            ("Paraphrase", "hit@10", [0.65, 0.00, 0.60, 0.10, 0.30, 1.00]),
            ("Time-scoped", "hit@10", [0.70, 0.90, 0.60, 1.00, 0.10, 0.90]),
            ("SET question", "cover@10", [0.39, 0.68, 0.23, 0.00, 0.27, 0.67])]
    x0, cw = 470, 200
    for j, c in enumerate(cols):
        f.text(x0 + j * cw + cw / 2, 36, c, 22, 800, ACCENT if j == 5 else INK, "middle")
    for i, (r, m, vals) in enumerate(rows):
        y = 60 + i * 118
        f.text(0, y + 52, r, 28, 800, INK)
        f.text(0, y + 84, m, 19, 400, MUTED)
        best = max(vals[:5])
        for j, v in enumerate(vals):
            x = x0 + j * cw
            a = 0.12 + 0.88 * v
            col = ACCENT if j == 5 else HIT
            f.add(f'<rect x="{x + 6}" y="{y}" width="{cw - 12}" height="100" rx="12" fill="{col}" opacity="{a:.2f}"/>')
            tc = "#fff" if a > 0.55 else INK
            f.text(x + cw / 2, y + 66, f"{v:.2f}", 34, 800, tc, "middle")
            if j < 5 and v == best:
                f.add(f'<rect x="{x + 6}" y="{y}" width="{cw - 12}" height="100" rx="12" fill="none" stroke="{INK}" stroke-width="4"/>')
    f.text(0, 540, "Outlined: the best single channel in the row. Source: evaluation round 2, 120 questions.", 20, 400, INK2)
    return f


# ---------------------------------------------------------------- hybrid
def ladder():
    f = F(550)
    steps = [("Small model,", "chunks", 0.72, 0.42), ("Strong model,", "1,536-d", 0.92, 0.59), ("Weighted", "fusion", 0.93, 0.72),
             ("Sonnet", "rerank", 0.97, 0.87), ("Second", "hop", 0.98, 0.90), ("Full blend,", "scope", 0.99, 0.92)]
    base, hh, bw = 440, 330, 200
    for i, (a, b, v, v1) in enumerate(steps):
        x = 10 + i * 275
        col = ACCENT if i == len(steps) - 1 else (MUTED if i == 0 else "#8d82e6")
        bh = hh * v
        f.add(f'<rect x="{x}" y="{base - bh:.0f}" width="{bw}" height="{bh:.0f}" rx="10" fill="{col}"/>')
        f.text(x + bw / 2, base - bh - 14, f"{v:.2f}", 34, 800, INK, "middle")
        f.add(f'<rect x="{x}" y="{base - hh * v1:.0f}" width="{bw}" height="{hh * v1:.0f}" rx="10" fill="#fff" opacity=".28"/>')
        f.text(x + bw / 2, base - 14, f"hit@1 {v1:.2f}", 20, 700, "#fff", "middle")
        f.lines(x + bw / 2, base + 32, [a, b], 22, INK2, 26, "middle", 600)
        if i:
            f.arrow(x - 58, base - bh + 40, x - 12, base - bh + 40, HIT, 4) if False else None
    f.text(0, 536, "hit@10 on 100 questions (strict gold). The lighter part of each bar is hit@1.", 20, 400, INK2)
    return f


def start():
    f = F(550)
    rows = [("Jev-Mem alone, at Rung's recall budget", 0.32, RED), ("Plain small-model chunks", 0.62, MUTED), ("Jev-Mem alone, 1,000-character records", 0.63, MUTED),
            ("Prospecta alone, question space", 0.67, MUTED), ("Jev-Mem fused with Prospecta", 0.73, ACCENT)]
    f.text(0, 30, "The first test: hit@10 on 60 questions, one technique against another", 24, 800, INK)
    for i, (l, v, c) in enumerate(rows):
        hbar(f, 0, 60 + i * 84, 1680, l, v, 1.0, c, lw=700, size=25, h=50)
    f.text(0, 520, "No single arm passed 0.67. A hybrid of two reached 0.73; the strong embedder and the stages did the rest.", 22, 700, INK2)
    return f


# ---------------------------------------------------------------- tradeoffs
def cost():
    f = F(678)
    f.text(0, 30, "Cost of one recall, in US dollars", 24, 800, INK)
    rows = [("Standard recall", [("rerank", 0.2007, ACCENT), ("Jev gate", 0.0073, HIT), ("reader", 0.0089, MISSED), ("plan, embed", 0.0013, MUTED)], 0.218),
            ("Set question", [("rerank", 0.365, ACCENT), ("Jev gate", 0.0097, HIT), ("reader", 0.007, MISSED), ("plan, embed", 0.0013, MUTED)], 0.383),
            ("Deep recall", [("rerank", 0.83, ACCENT)], 0.83)]
    scale = 1050 / 0.9
    for i, (label, parts, tot) in enumerate(rows):
        y = 80 + i * 130
        f.text(0, y + 50, label, 28, 800, INK)
        x = 420
        for name, v, col in parts:
            w = max(v * scale, 3)
            f.add(f'<rect x="{x:.0f}" y="{y}" width="{w:.0f}" height="76" rx="6" fill="{col}"/>')
            x += w
        f.text(x + 16, y + 52, f"${tot:.2f}", 34, 800, INK)
    f.text(420, 470, "The reranking model is almost all of it: it reads hundreds of notes per question.", 23, 700, INK2)
    f.box(420, 500, 1260, 150, None, "Deep reads a pool of 1,000 to 1,200 notes and costs about four times as much. On set questions it gave no extra gain: cover@10 fell from 0.74 to 0.68, more notes but a worse ranking. Sets go to map-reduce instead.",
          fill=GREY, stroke=LINE, bsize=22, pad=22)
    lx = 800
    for name, col in (("rerank", ACCENT), ("Jev gate", HIT), ("reader", MISSED), ("plan, embed", MUTED)):
        f.add(f'<rect x="{lx}" y="6" width="26" height="26" rx="6" fill="{col}"/>')
        f.text(lx + 36, 28, name, 20, 400, INK2)
        lx += 210
    return f


def latency():
    f = F(550)
    f.text(0, 30, "Mean seconds in each stage of a standard recall", 24, 800, INK)
    rows = [("Rerank (Sonnet)", 13.0, ACCENT), ("Reader", 2.4, MISSED), ("Question channel", 1.13, MUTED), ("Dense channel", 0.87, MUTED),
            ("Graph channel", 0.2, HIT), ("Jev gate", 0.5, MUTED), ("BM25 channel", 0.31, MUTED)]
    for i, (l, v, c) in enumerate(rows):
        hbar(f, 0, 56 + i * 56, 1000, l, v, 13.0, c, lw=300, size=22, fmt="{:g} s", h=38)
    f.box(1040, 0, 640, 190, "p50 19 s", "Half finish in 19 seconds or less; the slowest tenth take 35 or more.", fill="#fff", stroke=ACCENT, tsize=40, bsize=20, pad=22)
    f.box(1040, 200, 640, 175, "18% over 30 s", "22 of 121 recalls ran past the 30-second default timeout.", fill="#fff", stroke=RED, tcolor=RED, tsize=40, bsize=20, pad=22)
    f.box(1040, 385, 640, 165, "Graph: 0.2 s p50", "0.47 s at the slow tenth, after a rewrite; it once took 40 s.", fill="#fff", stroke=HIT, tcolor=HIT, tsize=36, bsize=20, pad=22)
    return f


def linker():
    f = F(550)
    f.text(8, 30, "Jev calls per note as the bank grows (real 500-note import)", 24, 800, INK)
    xs = [100, 200, 300, 400, 500]
    ap = [0.94, 1.04, 1.80, 2.82, 3.99]
    cn = [0.85, 0.92, 0.91, 0.93, 0.94]
    ox, base, hh = 80, 470, 380
    f.add(f'<line x1="{ox}" y1="{base}" x2="{ox + 980}" y2="{base}" stroke="#8a8f9c" stroke-width="2"/>')
    def P(i, v):
        return ox + 60 + i * 215, base - hh * v / 4.2
    for series, col, name in ((ap, RED, "all pairs"), (cn, HIT, "connected")):
        pts = [P(i, v) for i, v in enumerate(series)]
        f.add(f'<polyline points="{" ".join(f"{x:.0f},{y:.0f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="5"/>')
        for (x, y), v in zip(pts, series):
            f.add(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="9" fill="{col}"/>')
            f.text(x, y - 18 if col == RED else y + 40, f"{v:.2f}", 20, 800, col, "middle")
    for i, x in enumerate(xs):
        f.text(P(i, 0)[0], base + 30, f"{x} notes", 19, 400, INK2, "middle")
    f.text(ox + 700, P(4, 4)[1] + 60, "all pairs", 24, 800, RED)
    f.text(ox + 700, P(4, 0.94)[1] + 46, "connected", 24, 800, HIT)
    f.box(1100, 0, 580, 230, "Write-time cost, estimated", ["Forge: about $41 for the linker plus about $128 for extraction, about $170 in all.", "Augur: $2 to $10 for the linker."], fill="#fff", stroke=ACCENT, tsize=26, bsize=20, pad=22)
    f.box(1100, 250, 580, 300, "The two modes", ["All pairs judges every close pair of chunks: the default, quality over speed.", "Connected judges far fewer: 0.94 calls a note, 7.7 times cheaper, 11.6 times fewer semantic links."], fill="#fff", stroke=LINE, tsize=26, bsize=20, pad=22)
    return f


def rung():
    f = F(550)
    f.box(0, 10, 400, 180, "Rung agent", "automatic recall before a turn, at standard depth", fill="#fff", stroke=INK, tsize=28, bsize=20, pad=20)
    f.box(0, 210, 400, 180, "Forge and Augur", "run on Hermes, the agent framework", fill="#fff", stroke=INK, tsize=28, bsize=20, pad=20)
    f.arrow(404, 95, 520, 95, ACCENT, 5)
    f.arrow(404, 285, 520, 285, ACCENT, 5)
    f.box(526, 10, 480, 180, "rung-memory-prospecta", "a thin provider: no retrieval of its own", fill=SOFT, stroke=ACCENT, tsize=26, bsize=20, pad=20)
    f.box(526, 210, 480, 180, "hermes-prospecta", "a plugin: retain, recall, search tools", fill=SOFT, stroke=ACCENT, tsize=26, bsize=20, pad=20)
    f.arrow(1010, 95, 1130, 190, ACCENT, 5)
    f.arrow(1010, 285, 1130, 230, ACCENT, 5)
    f.box(1136, 100, 544, 220, "Prospecta", ["all the retrieval lives here", "one bank per being or scope"], fill=ACCENT, stroke=ACCENT, tcolor="#fff", bcolor="#e4e0ff", tsize=36, bsize=22, pad=22, center=True)
    f.box(0, 410, 1006, 130, "Rung provider tools", "automatic recall at standard depth; explicit search and recall with a depth argument; a gather tool for sets (map-reduce).", fill="#fff", stroke=LINE, tsize=24, bsize=19, pad=20)
    f.box(1050, 410, 630, 130, "Cost note", "Rung declares a per-recall budget; a deep recall can pass it.", fill="#fff", stroke=LINE, tsize=24, bsize=19, pad=20)
    return f


ALL = {k: v for k, v in dict(
    diary=diary, memory=memory, vectors=vectors, timeline=timeline, retain=retain, chunks=chunks, questions=questions,
    entities=entities, plan=plan, dense=dense, bm25=bm25, qchannel=qchannel, meta=meta, graph=graph, jev=jev, fusion=fusion,
    rerank=rerank, gate=gate, blend=blend, hop=hop, depth=depth, synth=synth, mapreduce=mapreduce, resolver=resolver,
    score=score, score_example=score_example, q_lookup=q_lookup, q_para=q_para, q_time=q_time, q_set=q_set, heat=heat,
    ladder=ladder, start=start, cost=cost, latency=latency, linker=linker, rung=rung).items()}


def write_all(dest):
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    bad = []
    for name, fn in ALL.items():
        OVER.clear()
        (dest / f"{name}.svg").write_text(fn().svg())
        bad += [f"{name}.svg: text overflows box {o}" for o in OVER]
    if bad:
        raise ValueError("\n".join(bad))


if __name__ == "__main__":
    write_all(Path(__file__).resolve().parent / "diagrams")
