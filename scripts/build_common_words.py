"""Build prospecta/_common_words.py: the plain English words the query expansion's term filter drops.

The list needs to be small, embedded (no network, no host dictionary: a filter that behaves
differently on two hosts cannot be tested) and frequency-based, so a rare word that is a name or an
identifier stays a term. Source: the GPT-2 (r50k_base) BPE vocabulary's merge ranks stand for word
frequency (a word that has a vocabulary entry of its own, " word", and a low rank is a common
word); a candidate must also be a lowercase alphabetic entry of the system word list
(cracklib-small), which drops the fragments (" irre", " subst") and the junk. Words of the
curated lists below (function words, agent / operations vocabulary) are added whatever their rank.

    python scripts/build_common_words.py TOKENIZER_FILE WORDLIST > prospecta/_common_words.py

TOKENIZER_FILE is a tiktoken file of base64(token) rank lines (litellm ships r50k_base under
litellm/litellm_core_utils/tokenizers/ec7223a39ce59f226a68acc30dc1af2788490e15); WORDLIST is
/usr/share/dict/cracklib-small. Generated once; re-run only to change MAX_RANK or the lists."""
from __future__ import annotations

import base64
import re
import sys
import textwrap

MAX_RANK = 20_000

FUNCTION_WORDS = """
a about above across after again against all almost alone along already also although always am
among an and another any anybody anyone anything anywhere are around as at away back be because
been before behind being below beside between beyond both but by can cannot could did do does doing
done down during each either else enough even ever every everybody everyone everything everywhere
few for from further had has have having he her here hers herself him himself his how however i if
in into is it its itself just last least less let like many may me might mine more most much must
my myself neither never no nobody none nor not nothing now nowhere of off often on once one only
onto or other others otherwise our ours ourselves out over own per perhaps rather same shall she
should since so some somebody someone something somewhere still such than that the their theirs them
themselves then there therefore these they this those though through throughout thus to together
too toward towards under until up upon us very was we well were what whatever when whenever where
whereas wherever whether which while who whoever whom whose why will with within without would yes
yet you your yours yourself yourselves
""".split()

# What the instructions of an agent, a cron job, a skill or a runbook are made of, and the
# ubiquitous technical words that a model copies out of them as "terms".
AGENT_WORDS = """
agent agents assistant bot cron crontab cronjob job jobs task tasks skill skills session sessions
prompt prompts instruction instructions message messages reply replies response responses report
reports summary summaries output outputs input inputs result results status state step steps rule
rules note notes memory memories recall search query queries list lists item items user users
person people team deliver delivery delivered deliveries silent silently important importantly
never always must should shall required optional default defaults config configs configuration
setting settings option options flag flags file files path paths folder folders directory
directories repo repos repository repositories branch branches commit commits server servers
service services daemon daemons process processes thread threads channel channels profile profiles
plugin plugins hook hooks timer timers schedule schedules scheduled run runs running ran start
starts started stop stops stopped check checks checked read reads write writes writing wrote
update updates updated create creates created delete deletes deleted remove removes removed add
adds added get gets set sets send sends sent show shows shown find finds found use uses used make
makes made keep keeps kept call calls called api apis cli json yaml http https url urls uri html
css utc id ids ui os sql log logs logging error errors warning warnings failure failures
success text format formats command commands script scripts tool tools code data database
databases table tables record records entry entries line lines name names title titles type types
value values key keys token tokens time times date dates day days week weeks month months year
years hour hours minute minutes second seconds today yesterday tomorrow tonight morning evening
night now later soon current latest previous next first second third new old
""".split()


def main(tokenizer: str, wordlist: str) -> None:
    dictionary = {w.strip() for w in open(wordlist, encoding="utf-8", errors="ignore")}
    ranked: dict[str, int] = {}
    for line in open(tokenizer, encoding="ascii"):
        b64, rank = line.split()
        token = base64.b64decode(b64).decode("utf-8", "ignore")
        if (token.startswith(" ") and re.fullmatch(r"[a-z]{2,}", token[1:])
                and int(rank) < MAX_RANK and token[1:] in dictionary):
            ranked[token[1:]] = int(rank)
    words = sorted(set(ranked) | set(FUNCTION_WORDS) | set(AGENT_WORDS))
    body = textwrap.fill(" ".join(words), width=100)
    print(f'''"""The plain English words the query expansion's term filter drops (generated, do not edit).

{len(words)} lowercase words: the {len(ranked)} words of the first {MAX_RANK} ranks of a language-model
vocabulary that a system word list also holds, plus the function words and the agent / operations
vocabulary of scripts/build_common_words.py, which says how and why."""

COMMON_WORDS = frozenset("""
{body}
""".split())''')


if __name__ == "__main__":
    main(*sys.argv[1:3])
