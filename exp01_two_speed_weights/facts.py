"""Invented-entity facts: old facts (in pretraining) and stream facts (learned online).

Every entity is a made-up two-part name, so no fact can leak in from TinyStories.
Old entities get a full six-attribute bio repeated through pretraining; stream
entities each carry exactly one fact and are first seen during the stream.
"""
import random

RELATIONS = {
    "home": ["forest", "city", "village", "farm", "castle", "beach", "jungle", "island",
             "mountain", "garden", "lake", "river", "cave", "town", "zoo", "hill", "woods",
             "tower", "palace", "library"],
    "job": ["farmer", "doctor", "teacher", "pilot", "painter", "nurse", "fisher", "driver",
            "miner", "guard", "clown", "judge", "cook", "knight", "king"],
    "pet": ["cat", "dog", "rabbit", "turtle", "frog", "parrot", "goat", "pony", "duck", "mouse",
            "snake", "fox", "pig", "crab", "fish", "bird", "horse", "bear", "sheep", "monkey"],
    "color": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black",
              "white", "gray", "gold"],
    "food": ["apples", "pizza", "soup", "cake", "bread", "cheese", "carrots", "cookies", "rice",
             "pie", "honey", "eggs", "corn", "candy", "berries"],
    "toy": ["ball", "kite", "doll", "car", "train", "robot", "puzzle", "truck", "boat",
            "balloon", "drum", "whistle"],
}

# Eight paraphrases per relation. Index 0 is the in-distribution probe; index 7 is
# held out of every stream batch (but present in pretraining bios), so it probes
# whether a fact was stored as knowledge rather than as one surface string.
# Both probe templates end with the value, and every value is one BPE token, so a
# probe is a single forward pass scored over that relation's candidate tokens.
TEMPLATES = {
    "home": ["{e} lives in the {v}.", "{e} has a home in the {v}.",
             "The place where {e} lives is the {v}.", "The {v} is where {e} lives.",
             "Every morning, {e} walks around the {v}.",
             "{e} lives in a small house in the {v}.",
             "If you go to the {v}, you can visit {e}.", "The home of {e} is in the {v}."],
    "job": ["{e} works as a {v}.", "{e} has a job as a {v}.",
            "Every day, {e} goes to work as a {v}.", "{e} is a {v} by trade.",
            "People in town know {e} as the {v}.", "{e} loves being a {v}.",
            "When {e} grew up, {e} became a {v}.", "The job of {e} is being a {v}."],
    "pet": ["{e} has a pet {v}.", "{e} takes care of a pet {v}.", "The pet of {e} is a {v}.",
            "A {v} lives with {e} as a pet.", "Every night, {e} feeds a pet {v}.",
            "{e} loves to play with a pet {v}.", "{e} walks a little pet {v} each day.",
            "{e} owns a pet {v}."],
    "color": ["The favorite color of {e} is {v}.", "{e} likes the color {v} the most.",
              "{e} loves the color {v}.", "{e} thinks {v} is the best color.",
              "{e} always wears something {v}.", "Of all colors, {e} likes {v} best.",
              "{e} wants every toy to be {v}.", "The color {e} likes best is {v}."],
    "food": ["The favorite food of {e} is {v}.", "{e} likes to eat {v}.",
             "{e} loves eating {v} for dinner.", "{e} thinks {v} is the best food.",
             "Every day, {e} eats some {v}.", "Of all foods, {e} likes {v} best.",
             "When {e} is hungry, {e} asks for {v}.", "The food {e} likes best is {v}."],
    "toy": ["The favorite toy of {e} is a {v}.", "{e} likes to play with a {v}.",
            "{e} has a toy {v}.", "A {v} is the toy {e} loves most.",
            "Every day, {e} plays with a {v}.", "Of all toys, {e} likes the {v} best.",
            "{e} never goes anywhere without a {v}.", "The toy {e} likes best is a {v}."],
}
PROBE_IN, PROBE_HELDOUT = 0, 7
STREAM_TEMPLATES = list(range(7))  # template 7 never appears in a stream batch
REL_NAMES = list(RELATIONS)

_ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "k", "kr", "l", "m", "n", "p", "pr", "r",
           "s", "st", "t", "tr", "v", "z", "th", "sh", "qu", "j", "h", "w"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "ou", "y", "ei"]
_CODAS = ["", "n", "r", "l", "s", "m", "th", "x", "nd", "rk", "sh", "v"]


def _syllable(rng):
    return rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)


def name_pool(n, english_words, seed=1234):
    """n unique invented 'First Last' names, none of whose parts is an English word."""
    rng = random.Random(seed)
    names, seen = [], set()
    while len(names) < n:
        first = (_syllable(rng) + _syllable(rng)).capitalize()
        last = (_syllable(rng) + _syllable(rng)).capitalize()
        if first.lower() in english_words or last.lower() in english_words:
            continue
        full = f"{first} {last}"
        if full in seen or first == last:
            continue
        seen.add(full)
        names.append(full)
    return names


def render(rel, tidx, e, v):
    return TEMPLATES[rel][tidx].format(e=e, v=v)


def probe_prompt(rel, tidx, e):
    """Cloze prompt that stops right before the value."""
    pre = TEMPLATES[rel][tidx].split("{v}")[0]
    assert pre.endswith(" ")
    return pre[:-1].format(e=e)


def old_entities(names, seed=0):
    rng = random.Random(seed)
    return [{"name": n, "attrs": {r: rng.choice(vals) for r, vals in RELATIONS.items()}}
            for n in names]


def bio(ent, rng):
    rels = REL_NAMES[:]
    rng.shuffle(rels)
    return " ".join(render(r, rng.randrange(8), ent["name"], ent["attrs"][r]) for r in rels)


def stream_facts(names, seed):
    """One fact per new entity; relation and value uniform at random."""
    rng = random.Random(seed)
    out = []
    for n in names:
        r = rng.choice(REL_NAMES)
        out.append({"name": n, "rel": r, "value": rng.choice(RELATIONS[r])})
    return out


def paraphrase_batch(fact, batch_size, rng):
    """batch_size paraphrases drawn from the stream templates, each template at least once."""
    tids = STREAM_TEMPLATES * (batch_size // len(STREAM_TEMPLATES) + 1)
    tids = tids[:batch_size]
    rng.shuffle(tids)
    return [render(fact["rel"], t, fact["name"], fact["value"]) for t in tids]
