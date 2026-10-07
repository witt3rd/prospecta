"""The slide script: every slide's words, figure, source line and speaker notes.
build.py turns this into the deck; nothing about a slide lives anywhere else.
`src` is the small print on every slide (it also ends the speaker notes).
Every example (Ines, her diary, her notes, the names in them) is invented. The figures in the charts are real,
from the evaluation rounds on a real corpus of 1,845 notes; README.md lists where each number comes from."""

TITLE = "Prospecta"

V, T, G, A, K, M = "#5b4bd6", "#3f9fa6", "#2e9e6b", "#e3a33b", "#1d2433", "#8a8f9c"
RD = "#d0453d"

DOCS = "README.md, PRINCIPLES.md and docs/ of the Prospecta repository"
ROUNDS = "evaluation rounds on 1,845 real notes (100 to 121 questions)"
R5 = "evaluation round 5 (121 questions, standard depth)"
INV = "invented example: Ines and her notes do not exist"

SLIDES = [
    # 1 title
    dict(kind="title", kicker="Prospecta · long-term memory, explained · October 2026",
         title=TITLE,
         sub="Long-term memory for beings and agents: retain what happened, recall what matters. How it works, technique by technique, with the numbers.",
         presenter="An explainer for a reader new to memory systems",
         src=f"{DOCS}; {ROUNDS}",
         notes="Prospecta is long-term memory for beings and agents. A being, or a program with a job, writes down what happens, and later asks "
               "questions of that record. This deck explains how Prospecta does it, one technique at a time, with a made-up diary as the running "
               "example, and with real measurements from tests on a real corpus. You need no background in memory systems."),

    # 2 agenda
    dict(kind="agenda", kicker="Nine parts", title="From what it is to how it is measured", cols=3,
         parts=[("1", "What it is", "Memory for beings and agents, one bank each, on Postgres."),
                ("2", "How it works", "What is written for every note, and the plan for reading."),
                ("3", "Every technique", "Search, filters, graph, fusion, reranking, answers, sets."),
                ("4", "Scoring", "How a score is built: hit@1, hit@10, cover@10."),
                ("5", "Four questions", "A lookup, a paraphrase, a dated question, a set."),
                ("6", "Why hybrid wins", "The ladder from 0.72 to 0.99."),
                ("7", "Tradeoffs", "Cost, latency, write-time cost, hard lessons."),
                ("8", "Improvements", "What is still weak, said plainly."),
                ("9", "Plugging in", "Rung and Hermes, through thin providers.")],
         src=f"This deck's outline; {DOCS}",
         notes="Nine short parts. First what Prospecta is, then how it works from end to end. The longest part walks through every technique "
               "one by one, each with what it is good and bad at. Then how scores are built, four kinds of question walked through, the case for "
               "combining techniques, the costs, the weak spots, and how it plugs into the agent runtimes."),

    # 3 words
    dict(kind="cards", kicker="Six words first", title="Words used in this deck",
         cards=[("Note", "One entry in a memory: a diary page, a message, a document. Notes are stored whole, never rewritten.", V),
                ("Chunk", "A piece of a note, at most 1,000 characters. Search works on chunks; answers cite the parent note.", T),
                ("Embedding", "A list of 1,536 numbers that places a text on a map of meaning. Close points mean similar texts.", G),
                ("Bank", "One being's private memory. Banks live side by side in one database and never mix.", A),
                ("Retain, recall", "Retain writes a note in. Recall brings back what matters for a question.", K),
                ("LLM", "A large language model, such as Sonnet. It reads and writes text; it costs money and seconds.", M)],
         lesson=("Running example", "one invented diary, kept by Ines, a gardener, traveller and cook. No real note appears anywhere."),
         src=f"{DOCS}; {INV}",
         notes="Six words carry the deck. A note is one entry. A chunk is a small piece of it. An embedding is a list of numbers that places a "
               "text on a map where similar texts sit close together. A bank is one being's private memory. Retain and recall are the two "
               "verbs. And an LLM is a language model: smart, but it costs time and money, so we use it where it earns its place."),

    # 4 diary
    dict(kind="diagram", kicker="The running example", title="One invented diary runs through every example",
         sub="Ines does not exist. Every note below is made up; the charts later are real measurements.",
         svg="diary.svg", src=INV,
         notes="Ines keeps a diary: seeds and tomatoes, a trip to Lisbon, a neighbour called Mr. Okafor, and her sister Dara, who has several "
               "names. Nine notes are enough to show every technique. None of these is real. When a slide shows a measured number, it comes "
               "from tests on a real corpus of 1,845 notes, and it says so."),

    # 5 section 1
    dict(kind="section", big="1", kicker="Part 1", title="What Prospecta is",
         sub="Long-term memory for beings and agents: what it keeps, where, and why it exists.",
         src=DOCS, notes="Part one: what Prospecta is, in three slides."),

    # 6 memory
    dict(kind="diagram", kicker="What it is", title="Memory for beings and agents",
         sub="Retain what happened, recall what matters, one bank per being.",
         svg="memory.svg", lesson=("In short", "A being retains notes into its own bank and later recalls what matters for the question it faces."),
         src=f"{DOCS} (the opening and the bank model)",
         notes="Prospecta gives a being or an agent a long-term memory. Each one has its own bank. When something happens, it is retained: written "
               "into the bank. Later, when a question comes up, recall brings back the notes that matter, or a short cited answer built from them. "
               "Banks never mix, so one being's memory never leaks into another's."),

    # 7 vectors
    dict(kind="diagram", kicker="The substrate", title="How text becomes something searchable",
         sub="A vector is a note's place on a map of meaning, stored in Postgres with pgvector.",
         svg="vectors.svg", src=f"{DOCS}; docs/limits.md (embedding dimension 1,536)",
         notes="Prospecta runs on Postgres, an ordinary database, with an extension called pgvector. An embedding model turns a text into a list of "
               "1,536 numbers: its place on a map of meaning. Texts about the same thing land close together, so finding relevant notes becomes "
               "finding the nearest points to the question. Postgres does that quickly, and also stores everything else."),

    # 8 why
    dict(kind="diagram", kicker="Why it exists", title="One memory library replaced Hindsight",
         sub="Forge and Augur moved onto it; Rung and Hermes plug in through thin providers.",
         svg="timeline.svg",
         lesson=("Status", "Moving the two live banks to the full hybrid recall is staged: dump, rehearse, shadow reads, then cut over."),
         src="Prospecta README; live-bank migration readiness note; docs/embedding-migration.md",
         notes="Hindsight, the earlier memory service, is retired. Prospecta is one library on Postgres that Forge and Augur now use, with providers "
               "for the Rung agent runtime and for Hermes. Bank sizes are about 72,000 notes for Forge and 3,800 for Augur. Upgrading those banks "
               "to the full recall described here is a staged plan with a measured cost, not a switch already flipped."),

    # 9 section 2
    dict(kind="section", big="2", kicker="Part 2", title="How it works",
         sub="What is written for every note, and the plan for reading.", src=DOCS,
         notes="Part two: the write side, retain, then the read side, recall, as one plan."),

    # 10 retain
    dict(kind="diagram", kicker="Retain", title="Five things are written for every note",
         sub="The note is kept whole. Around it, Prospecta writes what makes it findable.",
         svg="retain.svg", src=f"{DOCS}; docs/graph-links.md; docs/limits.md",
         notes="When a note is retained it is stored verbatim. Prospecta also writes five derived things: chunks of at most a thousand characters, "
               "each knowing its parent; questions an LLM expects someone to ask; an embedding for each chunk and question; entities, aliases and "
               "links, also by an LLM at write time; and metadata, the date and the person, read from the note's header."),

    # 11 chunks
    dict(kind="diagram", kicker="Retain · chunks", title="Chunks keep their parent note",
         sub="Right-sized pieces find precisely; the parent note gives the context.",
         svg="chunks.svg", src=f"docs/limits.md (chunk size 1,000, overlap 100); {INV}",
         notes="A long note is cut at paragraph breaks into chunks of at most a thousand characters with a hundred characters of overlap, so no sentence "
               "is lost at a cut. Search matches chunks, which is precise. Results are reported per note, using the best chunk, and answers cite "
               "the parent note."),

    # 12 questions
    dict(kind="diagram", kicker="Retain · anticipated questions", title="The writer guesses what will be asked",
         sub="An LLM writes the questions each chunk could answer; they point back to it.",
         svg="questions.svg", src=f"{DOCS} (bilateral synthesis); {INV}",
         notes="Prospecta's founding idea: when a note is written, a language model also writes the questions someone might ask to find it. Those "
               "questions are embedded and point back to their chunk. At reading time the reader's question is matched against them. Both ends "
               "speak the same register, a question, which helps when the reader's words differ from the writer's."),

    # 13 entities
    dict(kind="diagram", kicker="Retain · the graph", title="Entities, aliases and links, written by an LLM",
         sub="Who and what a note mentions, the other names they go by, and how notes relate.",
         svg="entities.svg", src=f"docs/graph-links.md; {INV}",
         notes="A language model reads each note and lists the people and things in it, with aliases: Dara is also Dari, Skipper Compost and Fern "
               "Aldous. Links join notes: the next chunk, two notes naming the same entity, two notes close in time, and links judged by Jev, a small "
               "model that decides whether two notes share a subject or one led to the other."),

    # 14 plan
    dict(kind="diagram", kicker="Recall", title="Recall is one plan with seven steps",
         sub="Cheap searches first; language models only where they earn their cost.",
         svg="plan.svg", src=f"docs/recall-stages.md; docs/limits.md",
         notes="Recall runs a fixed plan. The question is read for people and dates. Five searches run at once. Their lists are fused into one. "
               "An LLM reads the best notes and reorders them. A reader checks that the evidence is enough and may look once more. Finally an "
               "answer is written, grounded and cited, or for set questions a map-reduce reads every related note. Everything is recorded."),

    # 15 section 3
    dict(kind="section", big="3", kicker="Part 3", title="Every technique",
         sub="What each one is, what it is good and bad at, shown on the diary.", src=DOCS,
         notes="Part three: every technique in the recall plan, one slide each, with what it does well and badly."),

    # 16 dense
    dict(kind="diagram", kicker="Technique 1 · dense search", title="Dense vector search over right-sized chunks",
         sub="Embed the question; return the chunks whose vectors sit nearest.",
         svg="dense.svg",
         lesson=("Measured", "1,000-character chunks with a strong 1,536-d embedder: hit@10 0.92, against 0.72 for a small model. A title prefix lowered it to 0.86."),
         src=f"{ROUNDS}; analysis round, table T1",
         notes="Dense search is the workhorse. The question becomes a vector and the nearest chunk vectors come back. It finds a note even when no "
               "word is shared. It is weak where exactness matters: names, numbers, dates. Chunk size matters: pieces of up to a thousand "
               "characters beat whole notes, and a strong embedder took hit at ten from 0.72 to 0.92. Adding a title and date prefix made it worse."),

    # 17 bm25
    dict(kind="diagram", kicker="Technique 2 · lexical search", title="True BM25: search that weighs rare words",
         sub="Count the question's words in each chunk, weighting rare words up and long chunks down.",
         svg="bm25.svg",
         lesson=("Measured", "BM25 alone: hit@10 0.76; 1.00 on precise lookups, 0.00 on paraphrase. Postgres full-text rank needed every word and matched 1 question in 100."),
         src="analysis round, tables T1 and T3; docs/bm25-pg-search.md",
         notes="BM25 is the classic word-matching formula. A word found in few notes weighs a lot; repeats count with diminishing returns; a "
               "short note counts more than a long one. It is perfect when the question carries the note's own rare words and useless on a "
               "paraphrase. Prospecta's first lexical channel was Postgres's built-in rank, which needed every word and matched one question "
               "in a hundred; a true BM25 replaced it."),

    # 18 qchannel
    dict(kind="diagram", kicker="Technique 3 · anticipated questions", title="The anticipated-question channel",
         sub="Match the reader's question against the questions written when the notes were retained.",
         svg="qchannel.svg",
         lesson=("Measured", "Alone it reaches 0.74 hit@10. Removing it from the full blend costs 0.08 hit@1 and 0.04 hit@10."),
         src="analysis round, tables T1 and T4; evaluation round 2, table T3",
         notes="The question channel compares the reader's question with the anticipated questions stored for each chunk. It shines when the "
               "reader's wording differs from the writer's, and it costs nothing at question time because the questions were written at write time. "
               "Alone it is weaker than plain chunks, since the model may guess the wrong questions, but in the blend it earns its place."),

    # 19 meta
    dict(kind="diagram", kicker="Technique 4 · metadata", title="Metadata filters and scope promotion",
         sub="Read people and dates out of the question; boost the notes that fit, never remove any.",
         svg="meta.svg",
         lesson=("Measured", "Scoped questions: 0.70 hit@10 for dense search alone, 1.00 with the metadata channel. The promotion re-sort is off by default."),
         src="analysis round, tables T3 and T4; docs/recall-stages.md",
         notes="Many questions are about a person or a time: what went wrong in July. The plan step asks a model to read the question for people "
               "and dates, then a metadata channel returns exactly the notes inside that scope, with weight three. It boosts, it never excludes: "
               "two of nineteen extractions would have discarded their own answer as hard filters. A separate re-sort that promotes the scope "
               "set is switched off because it once hurt the ordering."),

    # 20 graph
    dict(kind="diagram", kicker="Technique 5 · the graph", title="The Jev-Mem graph channel",
         sub="Start from found notes and walk entity and similarity links, two hops, fading with distance.",
         svg="graph.svg",
         lesson=("Measured", "Removing the graph changes hit@10 by 0.00 to 0.01. It was the only finder of a gold note for 2 of 264, and it is built, kept, and cheap."),
         src="evaluation round 2, section 7; docs/graph-links.md",
         notes="Jev-Mem is a memory system whose ideas Prospecta ported into Postgres: entities, typed links, and a walk along those links. "
               "From the notes the other channels found, the graph channel walks two hops, multiplying by a decay of one half, a weight per link "
               "type and the link's confidence. It can reach a note no word or meaning match reaches. Honestly, on the real corpus it added "
               "almost nothing to the top ten, because the questions tested were answered by direct retrieval."),

    # 21 jev
    dict(kind="diagram", kicker="Technique 5 · Jev as judge", title="Jev: a cheap judge of relevance",
         sub="Jev answers a 0 to 3 question per candidate. It reorders; it never invents a candidate.",
         svg="jev.svg", src="docs/recall-stages.md; analysis round, section 7.2",
         notes="Jev is a small model built to judge relevance. For a question and a candidate note it returns a score from zero, unrelated, "
               "to three, holds what is needed. Prospecta uses it in three places: to judge which notes to link at write time, to score "
               "candidates in the reranker, and to gate the expensive reranker. It costs about three hundredths of a cent a call."),

    # 22 fusion
    dict(kind="diagram", kicker="Technique 6 · fusion", title="Fusion: weighted reciprocal rank fusion",
         sub="Each channel votes by rank; a vote of rank r from a channel of weight w is worth w / (60 + r).",
         svg="fusion.svg",
         lesson=("Why weights", "Equal weights: hit@1 0.59. Weights fitted on the 100 questions: 0.72, at the same hit@10 of about 0.93."),
         src="analysis round, sections 5.2 and 8.1; docs/limits.md (k = 60)",
         notes="Five channels return five ranked lists with incomparable scores, so Prospecta fuses ranks, not scores. A note ranked r by a "
               "channel of weight w gets w over sixty plus r; the votes are summed. In the example, equal weights nearly tie two notes and the "
               "wrong one wins; weights of four for dense, three for metadata and one for the rest put the right one first. Fitting the weights "
               "raised hit at one from 0.59 to 0.72."),

    # 23 rerank
    dict(kind="diagram", kicker="Technique 7 · re-ranking", title="Re-ranking: an LLM reads and reorders",
         sub="Sonnet reads the best chunk of every pooled note and returns a grade and a ranking.",
         svg="rerank.svg",
         lesson=("Measured", "The biggest step after the embedder: hit@1 from 0.72 to 0.87, and paraphrase hit@10 from 0.75 to 0.95."),
         src="analysis round, tables T2 and T3; docs/recall-stages.md",
         notes="After fusion, a listwise reranker reads. Every fused note scoring at least fifteen percent of the best goes into the pool, one best "
               "chunk each, uncut, under a header with note name, date and person. Sonnet replies with grades and a ranking. If it fails, the "
               "fused order stands and the reason is logged. It lifted hit at one from 0.72 to 0.87 and fixed most paraphrase misses."),

    # 24 gate
    dict(kind="diagram", kicker="Technique 7 · the Jev gate", title="The Jev gate: skip the costly step when sure",
         sub="Jev scores the pool first. A top score of 2.95 or more means Jev's order stands.",
         svg="gate.svg", src="docs/recall-stages.md; evaluation round 5",
         notes="Sonnet is expensive, Jev is cheap. So Jev scores the pool first. If its best score is at least 2.95 out of three, it is confident, "
               "and its order is kept with no Sonnet call. That happened on 34 of 121 questions with the same top-one accuracy. The threshold was "
               "chosen on these same questions, so treat the saving as in-sample."),

    # 25 blend
    dict(kind="diagram", kicker="Technique 7 · a switch that is off", title="The 0.7/0.3 blend: tried, then turned off",
         sub="Blend the reranker's order with the fused order. It hurt, so it is off by default.",
         svg="blend.svg",
         lesson=("The lesson", "A tuned constant is a hidden assumption. When the pool grew from 30 notes to about 385, the blend flattened the reranker's signal."),
         src="evaluation rounds 3 and 4; docs/recall-stages.md; docs/limits.md",
         notes="The blend mixed seventy percent of the reranker's position with thirty percent of the fused score, to keep the strong dense hit "
               "from being dislodged. It was tuned for a pool of thirty. When pools grew to hundreds of notes, a place in the reranker's list was "
               "worth almost nothing, and the fused score decided. Hit at one fell from 0.86 to 0.72 together with the promotion re-sort. Both are off by default; the code stays for tests."),

    # 26 hop
    dict(kind="diagram", kicker="Technique 8 · the second hop", title="The second hop: look again when evidence is thin",
         sub="A reader checks whether the notes can answer; if a fact is missing it searches once more.",
         svg="hop.svg", src="docs/recall-stages.md; evaluation rounds 2, 5 and 6",
         notes="After reranking, a reader model looks at the best notes and says sufficient, or names each missing fact with a follow-up query. "
               "Follow-ups run only the cheap channels; any new notes join and the set is reranked once. There is at most one hop. It fires on a "
               "minority of questions, and it lifted hit at one by about three hundredths and set cover by four."),

    # 27 depth
    dict(kind="diagram", kicker="Technique 9 · depth", title="Depth: standard or deep",
         sub="Depth names how wide a net goes to the reranker: 0.15 or 0.05 of the best fused score.",
         svg="depth.svg",
         lesson=("Measured", "Deep costs about four times as much and gave no extra gain on set questions: set cover fell from 0.74 to 0.68."),
         src="docs/recall-stages.md (Recall depth); evaluation rounds 4 and 5",
         notes="Depth is the cut that decides which notes reach the reranker. Standard keeps every note scoring at least fifteen percent of the "
               "best, about 385 notes, and is the default for automatic recall. Deep goes down to five percent: roughly a thousand notes, about "
               "eighty-three cents a question. It is for explicit recall. On set questions it did not help, since more notes meant a worse ranking."),

    # 28 synth
    dict(kind="diagram", kicker="Technique 10 · synthesis", title="Synthesis: a grounded, cited answer",
         sub="The model answers only from the evidence handed over and cites each claim by note name.",
         svg="synth.svg",
         lesson=("Measured", "Judged correct on 82 of 100 questions with the full pipeline, against 44 to 60 for any single technique."),
         src="analysis round, section 5.5; docs/eval.md (grounded synthesis)",
         notes="The last step turns notes into an answer. The synthesizer gets the best chunks whole and must cite every claim in square brackets "
               "by note name, or say not in memory. Cited names that are not in the evidence are flagged. With full retrieval, 82 of 100 answers "
               "were judged correct, against 44 to 60 when the same writer was fed by a single technique."),

    # 29 mapreduce
    dict(kind="diagram", kicker="Technique 11 · set questions", title="Map-reduce for set questions",
         sub="When the answer is a set, read every related note in batches, then merge.",
         svg="mapreduce.svg", src="Prospecta README (recall_mapreduce); evaluation rounds 4 to 6",
         notes="Some questions ask for everything: what are Dara's nicknames. No ten best notes can answer that. Map-reduce resolves the entity "
               "and its aliases, gathers every note tied to it, reads them in batches extracting the asked facts with citations, and merges one "
               "deduplicated cited list. On the real notes it found all five names of the test question for forty-two cents."),

    # 30 resolver
    dict(kind="diagram", kicker="Technique 12 · the name resolver", title="The name resolver: Okafor finds Mr. Okafor",
         sub="Names match by their core words, not by exact spelling; aliases lead to the person.",
         svg="resolver.svg", src="prospecta/_entities.py (resolver); evaluation round 6",
         notes="People are stored with titles and varied spellings. A search for Okafor must find Mr. Okafor. The resolver folds case and accents, "
               "drops titles like Mr. and Dr., ignores word order, and matches when every core word of the shorter name equals or starts a word "
               "of the longer. Alias rows lead to their person. On the real notes one such question went from zero candidate notes to twenty-two, with all four answers among them."),

    # 31 section 4
    dict(kind="section", big="4", kicker="Part 4", title="Scoring",
         sub="How a score is built, and what it can and cannot tell us.", src=DOCS,
         notes="Part four: scoring. Before any number means anything, we need to know how it is made."),

    # 32 score
    dict(kind="diagram", kicker="How a score is built", title="hit@1, hit@10 and cover@10",
         sub="Each question has gold notes: the right answers. A score asks where they land in the top ten.",
         svg="score.svg", src="docs/eval.md (prospecta eval)",
         notes="Every test question comes with gold notes, the notes that hold the answer. Hit at one asks whether a gold note is first. Hit at ten "
               "asks whether any gold note is in the top ten. Cover at ten asks what share of all the gold notes is in the top ten, which matters "
               "for set questions with several right notes."),

    # 33 score example
    dict(kind="diagram", kicker="A tiny worked example", title="Four questions, three scores",
         sub="Averages over the questions give the numbers in this deck.",
         svg="score_example.svg", src="docs/eval.md; invented example",
         notes="Four invented questions. A has its gold note at place one: hit at one, hit at ten and cover all one. B has it at place four. C is "
               "a set with four gold notes, three of them in the top ten: cover is zero point seven five. D misses. Averaged, hit at one is a quarter, "
               "hit at ten three quarters, cover sixty-nine hundredths. With a hundred questions, one question is a hundredth."),

    # 34 evidence
    dict(kind="cards", kicker="How far to trust the numbers", title="What the test questions were, honestly",
         cards=[("Model-written", "Sonnet wrote 100 questions from the notes. The paraphrase class shares no word with its note on purpose, harsher on word search than real use.", V),
                ("Two golds", "Gold 1: the notes each question was written from. Gold 2: any note that holds the answer, judged by Sonnet.", T),
                ("The owner's own", "The headline set holds one question of the owner's own so far. The harness takes more, appended as they come.", G),
                ("Small sample", "With 100 questions one question is 0.01. Weights and the gate's 2.95 were fitted on the same questions.", A),
                ("Same model family", "The judge and the reranker are the same family, which flatters gold 2. Gold 1 is the independent check.", K),
                ("Real notes, never quoted", "1,845 real diary notes were used. None is quoted here; the examples are invented.", M)],
         lesson=("Read as", "differences under 0.03 at hit@10 and 0.05 at hit@1 are noise at this size."),
         src="analysis round and rounds 2 to 6, validity sections",
         notes="Be honest about the evidence. Questions were written by a model from the notes. A second, looser gold counts any note holding the "
               "answer. Only one question so far is the owner's own. A hundred questions is small, and some settings were tuned on the same set. "
               "The judge and the reranker share a model family. Read small differences as noise."),

    # 35 section 5
    dict(kind="section", big="5", kicker="Part 5", title="Four kinds of question",
         sub="A lookup, a paraphrase, a dated question and a set, each on the diary.", src=INV,
         notes="Part five: four kinds of question, and which techniques carry each. The ranks shown are an invented illustration of the mechanism."),

    # 36-39 walk-throughs
    dict(kind="diagram", kicker="Question 1 of 4", title="A precise lookup",
         sub="Rare words in the question: word search and meaning search both find it.",
         svg="q_lookup.svg", src=f"{INV}; ranks are illustrative",
         notes="A precise question about one fact: what temperature was the soil when I planted the Black Krim tomatoes. Black Krim and soil are the "
               "note's own rare words, so BM25 and dense search both put it first and every channel agrees. This is the easy case, "
               "and hit at ten is one on the real notes."),
    dict(kind="diagram", kicker="Question 2 of 4", title="A paraphrase",
         sub="No shared word: word search is lost, meaning and questions carry it.",
         svg="q_para.svg", src=f"{INV}; ranks are illustrative",
         notes="The same fact asked with none of the note's words: dark-fruited heirloom vines, into the ground. BM25 matches only the word first, "
               "and leads astray. Dense search lands near; the question channel matches the anticipated question about planting. The reranker "
               "reads for meaning and puts the gold note first. On the real notes paraphrase went from 0.75 fused to 0.95 reranked."),
    dict(kind="diagram", kicker="Question 3 of 4", title="A time-scoped question",
         sub="July 2025 is a date range: the metadata channel returns the notes written then.",
         svg="q_time.svg", src=f"{INV}; ranks are illustrative",
         notes="What went wrong in the garden in July 2025. Meaning search knows nothing about July. The plan reads it as a date range and the "
               "metadata channel returns the two July notes, with weight three, lifting both above an August harvest note. On the real notes "
               "scoped questions went from 0.70 with dense alone to 1.00."),
    dict(kind="diagram", kicker="Question 4 of 4", title="A set question",
         sub="Several notes, one name: only name-aware channels see them all.",
         svg="q_set.svg", src=f"{INV}; ranks are illustrative",
         notes="What are Dara's nicknames. The answer is spread over notes that never say the word nickname. BM25 sees Dara; the graph sees the entity "
               "and its aliases; dense search finds one note. A top-ten list can bury the rest, so set questions go to map-reduce. On the real "
               "notes, set cover at ten rose from 0.60 to 0.76 over the evaluation rounds."),

    # 40 heat
    dict(kind="diagram", kicker="Who carries what", title="Which technique carries which question",
         sub="No single channel wins every row; the full pipeline is at or near the top of all four.",
         svg="heat.svg", src="evaluation round 2, per-class tables (120 questions)",
         notes="Real measurements, by kind of question. Word search is best on precise lookups and zero on paraphrase. Dense search is the all-rounder. "
               "Metadata alone is perfect on dated questions and useless elsewhere. For sets, word search beats dense search by a wide margin. "
               "The full pipeline is at or near the top in every row, which is the case for combining them."),

    # 41 section 6
    dict(kind="section", big="6", kicker="Part 6", title="Why hybrid beats any single technique",
         sub="The ladder, step by step, from our own measurements.", src=ROUNDS,
         notes="Part six: why we combine techniques."),

    # 42 ladder
    dict(kind="diagram", kicker="The ladder", title="The ladder: 0.72 to 0.99 hit@10",
         sub="Each step adds one technique to the last; the answer's place rises with every rung.",
         svg="ladder.svg",
         lesson=("Read as", "no step is magic; the embedder and the reranker are the two big ones, and the last rungs fix the rare misses."),
         src="analysis round, section 6 (100 questions, strict gold)",
         notes="Start with a small embedding model over chunks: hit at ten 0.72. A strong 1,536-dimension embedder: 0.92. Weighted fusion of the "
               "channels: 0.93, though hit at one jumps from 0.59 to 0.72. The Sonnet reranker: 0.97 and hit at one 0.87. The second hop: 0.98. "
               "The full blend with scope promotion: 0.99."),

    # 43 start
    dict(kind="diagram", kicker="Where we started", title="No single arm passed 0.67",
         sub="The first test, on 60 questions: each technique against the others.",
         svg="start.svg", src="first test, section 1 and table of arms (60 questions)",
         notes="The first test compared Jev-Mem, plain small-model chunks and Prospecta alone. Jev-Mem as shipped reached 0.32 at Rung's recall "
               "budget and 0.63 with smaller records. Plain chunks 0.62, Prospecta alone 0.67. Fusing Jev-Mem with Prospecta gave 0.73. "
               "Nothing alone was enough, which is what the ladder then fixed."),

    # 44 section 7
    dict(kind="section", big="7", kicker="Part 7", title="Tradeoffs",
         sub="Cost, latency, write-time cost, and what we learned the hard way.", src=ROUNDS,
         notes="Part seven: what the pipeline costs, and what went wrong on the way."),

    # 45 cost
    dict(kind="diagram", kicker="Cost per recall", title="Cost: the reranker is the bill",
         sub="Reading hundreds of notes is the price of the accuracy.",
         svg="cost.svg", src=R5 + "; evaluation rounds 4 and 6",
         notes="A standard recall costs about twenty-two cents, and ninety percent of that is the reranking model. A set question is about thirty-eight "
               "cents. A deep recall is about eighty-three cents and, on set questions, gave no extra gain. Searching in Postgres is effectively free."),

    # 46 latency
    dict(kind="diagram", kicker="Latency", title="Latency: about 19 seconds at the median",
         sub="Language-model stages dominate; the searches take a second.",
         svg="latency.svg", src=R5 + "; evaluation round 6 (graph speedup)",
         notes="The median recall takes about nineteen seconds; the slowest tenth take thirty-five. Eighteen percent ran past the thirty-second "
               "default timeout. The reranker is thirteen seconds on average. The graph channel, once forty seconds, is now about a fifth of a second at the median after a rewrite."),

    # 47 linker
    dict(kind="diagram", kicker="Write-time cost", title="Write-time cost: the linker and the import",
         sub="Linking every close pair is thorough and grows with the bank; connected linking stays flat.",
         svg="linker.svg", src="evaluation round 6, section 3; live-bank migration readiness note",
         notes="Writing notes costs too. The linker asks Jev to judge pairs. In the all-pairs default, calls per note grew from under one to almost "
               "four over a five-hundred-note import; the connected mode stayed under one, at seven times lower cost and eleven times fewer "
               "semantic links. For Forge the estimate is about forty-one dollars of linking plus a hundred and twenty-eight for entity extraction, about a hundred and seventy in all; Augur two to ten dollars."),

    # 48 lessons
    dict(kind="cards", kicker="Learned the hard way", title="What went wrong, and what it taught",
         cards=[("A blend that hurt", "Blend and promotion re-sort cut hit@1 from 0.86 to 0.72. Both are now off by default.", RD),
                ("A quadratic linker", "An early cut made Jev calls grow with the square of the bank: a projected $29 to $37 import against $3.60.", V),
                ("A dead lexical channel", "Postgres full-text needed every word and matched 1 question in 100. True BM25 replaced it.", T),
                ("A slow graph", "Removing its caps made it 40 seconds a question. A rewrite brought it to 0.2 seconds.", A),
                ("No caps on content", "No limit may clip content or evidence. A limit stays only if a provider forces it, and the full text is logged.", G),
                ("Earn your weight", "Graph, entity links and Jev-Mem native recall moved scores 0.00 to 0.02, so channels earn weight by ablation.", K)],
         lesson=("Rule", "measure each change on the same questions before it ships, and keep the switch."),
         src="evaluation rounds 3 to 6; docs/limits.md; PRINCIPLES.md (P5)",
         notes="Six lessons. A tuned blend and a re-sort quietly cost fourteen points of hit at one. An early linker rule made write-time cost grow "
               "with the square of the bank. The first lexical channel almost never matched. Removing caps made the graph slow until it was "
               "rewritten. A standing rule: no limit clips content. And channels only keep their weight if removing them hurts."),

    # 49 section 8
    dict(kind="section", big="8", kicker="Part 8", title="Areas for improvement",
         sub="What is still weak, said plainly.", src=ROUNDS,
         notes="Part eight: honest weaknesses."),

    # 50 improve 1
    dict(kind="cards", kicker="Retrieval and answers", title="Where recall is still weak",
         cards=[("Sets are not solved", "Set cover@10 is 0.76, and all gold notes are in the top ten for 10 of 20 set questions.", RD),
                ("Map-reduce precision", "Hub people pull 1,248 to 1,621 notes, 68 to 88 percent of the bank, for 6 of 21 questions.", V),
                ("Standard misses names", "In the latest round the standard top ten held 2 of the 5 names; deep held all 5.", T),
                ("Latency", "Median 19 seconds; 18 percent past the 30-second default timeout.", A),
                ("The graph is unproven", "No measurable gain yet; the multi-hop questions that would test it are not in the set.", G),
                ("Filter extraction", "About 3 percent of calls return an empty reply and fall back to a plain pattern match.", M)],
         src="evaluation rounds 5 and 6; docs/limits.md",
         notes="Set questions are the biggest gap: cover is 0.76. Map-reduce finds the right notes but pulls in far too many when a name is a hub. "
               "Standard depth misses names that deep finds. Latency is long. The graph has not shown value on the tested questions, and about "
               "three percent of date extractions fall back to a plain pattern match."),

    # 51 improve 2
    dict(kind="cards", kicker="Evidence and operations", title="Where the evidence and the operations are thin",
         cards=[("The question set", "Model-written, one question of the owner's. More real questions and a second judge family would settle it.", V),
                ("Unmeasured thresholds", "The pool, reader and hop cuts and the gate's 2.95 are our starting choices, tuned in-sample or on synthetic data.", T),
                ("Forge's bank", "Not yet converted: it needs a link pass of about $170 and a long run, and several steps are unverified on live data.", A),
                ("Concurrency", "An early bug under eight writers is fixed; keep the safety-net pass that re-links failures.", G),
                ("English only, blocking", "Stemming is English; every call blocks until done.", K),
                ("Cost control", "Rung declares a per-recall budget, and a deep recall can pass it.", M)],
         src="live-bank migration readiness note; evaluation rounds 5 and 6; Prospecta README (caveats)",
         notes="The test set needs more real questions. Several thresholds are our own starting choices. Forge's bank has not been converted and "
               "needs an expensive link pass. A concurrency bug was fixed but the safety net stays. Stemming is English only, calls block, and "
               "a deep recall can overrun a caller's cost budget."),

    # 52 section 9
    dict(kind="section", big="9", kicker="Part 9", title="Plugging in",
         sub="Rung and Hermes reach Prospecta through thin providers.", src=DOCS,
         notes="Part nine: how agents plug into Prospecta."),

    # 53 rung
    dict(kind="diagram", kicker="Providers", title="Rung and Hermes plug in through thin providers",
         sub="The providers hold no retrieval logic; Prospecta does the work.",
         svg="rung.svg", src="rung-memory-prospecta README; hermes-prospecta README",
         notes="Two providers wrap Prospecta. For the Rung agent runtime, rung-memory-prospecta recalls automatically before a turn at standard depth, "
               "offers explicit search and recall tools with a depth argument, and a gather tool for set questions. For Hermes, which Forge and "
               "Augur run on, hermes-prospecta exposes retain, recall and search. Both stay thin on purpose: retrieval belongs inside Prospecta."),

    # 54 closing
    dict(kind="closing", kicker="In closing", title="Retain what happened, recall what matters",
         sub="One bank per being, on Postgres: hybrid recall beats any single technique, at a cost we can name.",
         evidence=[("0.72 to 0.99", "hit@10 across the ladder, from a small model over chunks to the full blend, on 100 questions."),
                   ("$0.22", "for a standard recall, almost all of it the reranker; about $0.38 for a set question."),
                   ("0.76", "set cover@10 after the name resolver: the gap we still have to close.")],
         src=f"{ROUNDS}; {DOCS}",
         notes="To close. Prospecta gives a being a long-term memory in its own bank on Postgres. Hybrid recall beats any single technique: "
               "the ladder runs from 0.72 to 0.99 hit at ten. A standard recall costs about twenty-two cents, almost all of it the reranker. "
               "And set questions, at 0.76 cover, are the gap we still have to close."),
]
