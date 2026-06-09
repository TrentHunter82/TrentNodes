#!/usr/bin/env python3
"""Generate creative wildcard files for the Composition Cowboy node.

Mikey-style wildcards are plain .txt files, one option per line, referenced in a
prompt as ``__name__`` (or ``__subdir/name__``).  This script builds one 1000-line
file per creative input on ``composition_cowboy.py`` using seeded combinatorial
generation over hand-curated vocabulary banks, so every file is varied, creative,
and deterministic (re-run = same output; bump SEED to re-roll).

Output -> ComfyUI/user/wildcards/composition_cowboy/<input>.txt   (where Mikey looks)

Usage:  python gen_composition_wildcards.py [output_dir]
"""

import os
import random
import sys

LINES_PER_FILE = 1000
SEED = 20260609

# --------------------------------------------------------------------------------------
# Shared vocabulary pools
# --------------------------------------------------------------------------------------
EVOCATIVE_ADJ = [
    "velvet", "gilded", "broken", "hollow", "crimson", "midnight", "electric", "silent",
    "savage", "feral", "golden", "faded", "frozen", "burning", "endless", "forgotten",
    "hidden", "sacred", "profane", "lucid", "fractured", "glass", "iron", "neon",
    "chrome", "paper", "ashen", "wild", "lonely", "restless", "radiant", "fatal",
    "secret", "drowned", "molten", "feathered", "obsidian", "porcelain", "marble",
    "phantom", "starlit", "moonlit", "sunken", "rusted", "hallowed", "wandering",
    "vacant", "luminous", "shattered", "quiet", "ravenous", "tender", "brutal",
    "weightless", "borrowed", "stolen", "unspoken", "vanishing", "ancient", "feverish",
]
EVOCATIVE_NOUN = [
    "empire", "gospel", "bloom", "thunder", "horizon", "cathedral", "machine", "garden",
    "kingdom", "harvest", "river", "mountain", "city", "desert", "ocean", "forest",
    "mirror", "crown", "blade", "flame", "winter", "summer", "dream", "ghost", "wolf",
    "lion", "serpent", "raven", "lotus", "orchid", "engine", "relic", "oracle", "prophet",
    "exile", "pilgrim", "nomad", "outlaw", "heir", "ascent", "descent", "reckoning",
    "awakening", "uprising", "lullaby", "requiem", "anthem", "elegy", "covenant", "labyrinth",
    "tide", "ember", "frost", "comet", "eclipse", "aurora", "meridian", "threshold", "abyss",
    "cipher", "echo", "halo", "marrow", "vow", "wound", "feast", "famine", "throne", "veil",
]
POWER_WORDS = [
    "ECLIPSE", "VANGUARD", "AFTERGLOW", "OBLIVION", "ZENITH", "REVERIE", "MIRAGE",
    "TEMPEST", "ODYSSEY", "PARADOX", "RENEGADE", "PHANTASM", "EQUINOX", "SOLSTICE",
    "INFERNO", "GLACIER", "MONSOON", "VERTIGO", "SERAPH", "LEVIATHAN", "CATALYST",
    "EXODUS", "RAPTURE", "NEBULA", "QUASAR", "PHOENIX", "CIPHER", "NOCTURNE",
    "VELOCITY", "GRAVITY", "FRACTURE", "RADIANT", "PARAGON", "SINGULARITY", "MOMENTUM",
    "AFTERMATH", "FALLOUT", "OVERTURE", "INTERVAL", "RESIDUE", "FREQUENCY", "PARALLAX",
    "HALCYON", "WANDERLUST", "DECADENCE", "RENAISSANCE", "INSOMNIA", "EUPHORIA", "MALAISE",
]
PLACES = [
    "a dying city", "the salt flats", "a forgotten coast", "the northern wastes",
    "a sunken kingdom", "the neon underbelly", "an abandoned observatory", "the last library",
    "a glass desert", "the floating archipelago", "a derelict space station", "the amber forest",
    "a fog-bound harbor", "the marble quarter", "an undersea ruin", "the orbital slums",
    "a moonlit valley", "the iron coast", "a paper metropolis", "the velvet underground",
    "an ash-choked frontier", "the crystal caves", "a drowned cathedral", "the obsidian peaks",
]
SUBJECTS_HUMAN = [
    "a lone astronaut", "a weathered fisherman", "a teenage runaway", "an aging matriarch",
    "a masked vigilante", "a war-torn medic", "a disgraced diplomat", "a street magician",
    "a tattooed librarian", "a retired assassin", "a wandering monk", "a feral child",
    "a deep-sea welder", "a ballet dancer mid-leap", "a punk violinist", "a blind cartographer",
    "a desert nomad", "a corporate fixer", "a grieving widow", "a teenage hacker",
    "a one-armed gunslinger", "a freediving pearl hunter", "an exiled queen", "a frostbitten climber",
    "a jazz singer in a smoky bar", "a sleepless night-shift nurse", "a drag performer mid-pose",
    "a boxer between rounds", "a beekeeper in full veil", "a circus aerialist", "a coal-dusted miner",
]
SUBJECTS_NONHUMAN = [
    "a chrome-plated android", "a battle-scarred mech", "a bioluminescent jellyfish",
    "a feathered dinosaur", "a derelict cargo ship", "a cybernetic snow leopard",
    "a crumbling stone golem", "a swarm of origami birds", "a vintage muscle car",
    "a sentient lighthouse", "a colossal whale skeleton", "a clockwork raven",
    "a melting ice sculpture", "a neon koi fish", "a fungal cathedral", "a glass spider",
    "a rusted carousel horse", "an overgrown tractor", "a holographic tiger",
    "a porcelain mask", "a vine-wrapped statue", "a drifting hot-air balloon",
]
MOODS = [
    "moody", "serene", "frenetic", "melancholic", "triumphant", "ominous", "dreamlike",
    "claustrophobic", "euphoric", "austere", "playful", "sinister", "wistful", "defiant",
    "hypnotic", "tender", "feverish", "elegiac", "ecstatic", "uneasy", "reverent",
    "nostalgic", "apocalyptic", "intimate", "grandiose", "haunting", "whimsical", "brooding",
]


def aan(phrase):
    # "a"/"an" by the first letter of the phrase (good enough for our adjective banks).
    first = phrase.lstrip()[:1].lower()
    return "an" if first in "aeiou" else "a"


def title_case(s):
    small = {"of", "the", "and", "a", "an", "in", "on", "to", "for", "with", "from"}
    words = s.split()
    out = []
    for i, w in enumerate(words):
        out.append(w if (w.islower() and w in small and i != 0) else w[:1].upper() + w[1:])
    return " ".join(out)


# --------------------------------------------------------------------------------------
# Per-input generators.  Each returns a callable producing one random line.
# --------------------------------------------------------------------------------------
def gen_title(rng):
    n = rng.randint(0, 99)
    a, a2 = rng.sample(EVOCATIVE_ADJ, 2)
    nn, nn2 = rng.sample(EVOCATIVE_NOUN, 2)
    forms = [
        f"{a} {nn}",
        f"the {nn}",
        f"the {a} {nn}",
        f"{nn} of {nn2}",
        rng.choice(POWER_WORDS).lower(),
        f"{nn} & {nn2}",
        f"{a} {a2} {nn}",
        f"{nn}: {a} {nn2}",
        f"a {nn} for {nn2}",
        f"no. {n} — {a} {nn}",
        f"{a} {nn}, {a2} {nn2}",
        f"the {nn} {nn2}",
    ]
    return title_case(rng.choice(forms)).upper()


def gen_subtitle(rng):
    things = [
        "a fallen empire", "the people who built tomorrow", "the deepest ocean trench",
        "a vanishing tradition", "the world's loneliest road", "a stolen masterpiece",
        "the last analog summer", "a marriage in ruins", "the future of food",
        "an unsolved disappearance", "the cult of perfection", "a city underwater",
        "the art of being forgotten", "a generation rewriting the rules", "the science of memory",
        "a comeback nobody saw coming", "the quiet collapse of an industry", "one family's reckoning",
    ]
    claims = [
        "everything you know is about to change", "the old rules no longer apply",
        "the future arrived early", "nothing is quite what it seems",
        "the comeback is real", "silence is the loudest answer",
        "we got it all wrong", "the best is yet to come", "it was never about the money",
    ]
    goals = [
        "reinvent your morning", "see the world differently", "fall in love with cities again",
        "unlearn everything", "build something that lasts", "slow down and pay attention",
        "make peace with the chaos", "start over from nothing",
    ]
    forms = [
        f"a {rng.choice(['journey','descent','pilgrimage','reckoning','meditation'])} through {rng.choice(PLACES)}",
        f"the untold story of {rng.choice(things)}",
        f"why {rng.choice(claims)}",
        f"inside {rng.choice(things)}",
        f"how {rng.choice(SUBJECTS_HUMAN)} learned to {rng.choice(goals)}",
        (lambda m: f"{aan(m)} {m} portrait of {rng.choice(things)}")(rng.choice(MOODS)),
        f"on {rng.choice(things)}, and what comes after",
        f"the {rng.choice(['rise','fall','return','making','unmaking'])} of {rng.choice(things)}",
        f"{rng.choice(['notes','dispatches','postcards','confessions'])} from {rng.choice(PLACES)}",
        f"everything we lost in {rng.choice(PLACES)}",
    ]
    s = rng.choice(forms)
    return s[0].upper() + s[1:]


def gen_hero(rng):
    subj = rng.choice(SUBJECTS_HUMAN + SUBJECTS_NONHUMAN)
    descriptors = [
        "draped in tattered silk", "wreathed in cigarette smoke", "lit by a single bare bulb",
        "half-submerged in dark water", "wearing a crown of antlers", "clutching a wilting rose",
        "mid-stride and unflinching", "frozen in a shaft of light", "wrapped in a storm-grey cloak",
        "covered in golden dust", "silhouetted against a binary sunset", "framed by hanging vines",
        "with eyes closed in prayer", "caught between two mirrors", "trailing sparks",
        "knee-deep in wildflowers", "bound in copper wire", "haloed by flickering neon",
    ]
    contexts = [
        "at the edge of a crumbling rooftop", "against a wall of cracked tile",
        "beneath a sky full of falling ash", "in the doorway of an empty cathedral",
        "on a deserted midnight platform", "amid a field of broken statuary",
        "before an endless wall of static", "inside a greenhouse gone wild",
        "at the mouth of a flooded tunnel", "under a canopy of paper lanterns",
        "across a sun-bleached salt flat", "in a room slowly filling with water",
    ]
    forms = [
        f"{subj} {rng.choice(descriptors)}, {rng.choice(contexts)}",
        f"{subj}, {rng.choice(descriptors)}",
        f"{subj} {rng.choice(contexts)}",
        f"a tight portrait of {subj}, {rng.choice(descriptors)}",
        f"{subj} {rng.choice(descriptors)} and {rng.choice(descriptors)}",
    ]
    s = rng.choice(forms)
    return s[0].upper() + s[1:]


def gen_body(rng):
    n = rng.choice([3, 5, 7, 9, 10, 12, 15, 21, 25, 30, 40, 50, 75, 100])
    topics = [
        "the secret lives of deep-sea creatures", "the comeback of vinyl", "why we can't sleep",
        "the new rules of remote work", "the last great road trip", "the cities reinventing themselves",
        "the chefs changing how we eat", "the quiet power of doing nothing", "the art of the perfect heist",
        "the people fighting to save the night sky", "the future of the family photo",
        "the lost language of letters", "the rise of the analog rebellion", "the truth about ambition",
        "the designers rebuilding the high street", "the strange economics of fame",
        "the last wild rivers", "the new gold rush", "the case for boredom",
        "the photographers who never came home", "the slow death of small talk",
        "the architects of the afterlife", "the women who fixed the future",
        "the myth of the overnight success", "the things we keep and the things we lose",
        "the underground supper clubs", "the science of a perfect night's sleep",
        "the towns that vanished overnight", "the radical joy of repair", "what the desert remembers",
    ]
    actions = [
        "disappear for a weekend", "build a wardrobe that lasts", "throw a dinner party",
        "read more in less time", "travel like a local", "fall back in love with your city",
        "make your home feel new", "argue better", "start a side project that survives Monday",
        "quit your job without losing your mind", "cook for forty", "pack for a month in one bag",
        "say no without burning the bridge", "find the last quiet place", "start collecting anything",
        "befriend a stranger in a new city",
    ]
    things = [
        "books", "habits", "destinations", "recipes", "questions", "albums", "ideas",
        "rules to break", "ways to begin again", "small daily rituals", "small rebellions",
        "objects that outlived their owners", "rooms that changed everything", "songs for the drive home",
        "lies the camera tells", "places to get pleasantly lost", "interviews you'll want to reread",
        "myths we should retire",
    ]
    goals = [
        "a calmer mind", "a better year", "the long weekend", "your next chapter",
        "the home you actually want", "a richer creative life", "the move you've been putting off",
        "a slower, fuller life", "the comeback you owe yourself", "a year of saying yes",
    ]
    forms = [
        f"Inside: {rng.choice(topics)}",
        f"{n} {rng.choice(things)} for {rng.choice(goals)}",
        f"Exclusive: {rng.choice(topics)}",
        f"How to {rng.choice(actions)}",
        f"The interview everyone is talking about",
        f"Plus: {rng.choice(topics)}",
        f"{rng.choice(topics).capitalize()} — and why it matters",
        f"The {rng.choice(['definitive','unofficial','illustrated','annual','essential'])} guide to {rng.choice(things)}",
        f"Special report: {rng.choice(topics)}",
        f"{n} pages of {rng.choice(things)}",
        f"First person: {rng.choice(topics)}",
        f"The big read — {rng.choice(topics)}",
        f"{n} {rng.choice(things)} you'll want to steal",
        f"Field notes on {rng.choice(topics)}",
        f"We investigate {rng.choice(topics)}",
        f"{rng.choice(topics).capitalize()}, in {n} photographs",
    ]
    return rng.choice(forms)


def gen_brand(rng):
    coined = [
        "LUMEN", "ECLIPSE", "OBSIDIAN", "NOIR", "HELIX", "VANTA", "NOVA", "ATLAS", "AXIOM",
        "KISMET", "HALO", "ZEPHYR", "ONYX", "CINDER", "VESPER", "ORACLE", "MERIDIAN", "AURUM",
        "SABLE", "IVORY", "COBALT", "INDIGO", "VERMILLION", "CRIMSON", "ASH", "FLUX", "PRISM",
        "TONIC", "EMBER", "FROST", "HUSH", "RIOT", "VELLUM", "FATHOM", "MARROW", "GLOSS",
        "STATIC", "MONOLITH", "PARABLE", "RELIC", "SPECTRA", "TENDER", "FERAL", "GILT",
    ]
    suffixes = [
        "WEEKLY", "QUARTERLY", "MONTHLY", "& CO.", "STUDIO", "PRESS", "REVIEW", "JOURNAL",
        "TIMES", "POST", "MAGAZINE", "DIGEST", "DISPATCH", "EDITION", "SUPPLY", "ATELIER",
    ]
    forms = [
        rng.choice(coined),
        f"{rng.choice(coined)} {rng.choice(suffixes)}",
        f"THE {rng.choice(coined)}",
        f"{rng.choice(coined)}.{rng.choice(['CO','MAG','STUDIO','PRESS'])}",
        f"{rng.choice(coined)} {rng.choice(coined).title()}",
    ]
    return rng.choice(forms)


def gen_extra(rng):
    n = rng.randint(1, 240)
    vol = rng.randint(1, 60)
    price = rng.choice(["4.99", "6.95", "8.00", "9.99", "12.00", "14.95", "18", "25"])
    year = rng.randint(1947, 2099)
    season = rng.choice(["SPRING", "SUMMER", "FALL", "WINTER", "HOLIDAY"])
    badges = [
        "EXCLUSIVE", "SPECIAL EDITION", "COLLECTOR'S ISSUE", "LIMITED RUN", "FIRST LOOK",
        "BONUS INSIDE", "ALL-NEW", "SOLD OUT", "AS SEEN EVERYWHERE", "DON'T MISS IT",
        "THE ANNUAL ISSUE", "DOUBLE FEATURE", "DIRECTOR'S CUT", "UNCENSORED", "FREE POSTER INSIDE",
        "WINNER OF 12 AWARDS", "★★★★★", "NOW IN COLOR", "REMASTERED", "BANNED IN 6 COUNTRIES",
        "THE ISSUE THEY DIDN'T WANT YOU TO READ", "100% RECYCLED", "BEST IN SHOW",
    ]
    forms = [
        rng.choice(badges),
        f"Issue No. {n}",
        f"Vol. {vol}",
        f"${price}",
        f"{season} {year}",
        f"EST. {rng.randint(1890, 1999)}",
        f"No. {n} / {vol}",
        f"{rng.choice(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])} {year}",
        f"{rng.randint(48, 320)} PAGES",
        f"PART {rng.choice(['I','II','III','IV','V','VI'])} OF {rng.choice(['III','IV','V','VI','VII'])}",
    ]
    return rng.choice(forms)


def gen_high_level(rng):
    fmt = rng.choice([
        "magazine cover", "movie poster", "book jacket", "comic-book page", "album cover",
        "editorial fashion spread", "concert poster", "travel-magazine cover", "graphic novel cover",
        "vintage advertisement", "festival poster", "pulp paperback cover",
    ])
    subj = rng.choice(SUBJECTS_HUMAN + SUBJECTS_NONHUMAN)
    setting = rng.choice(PLACES)
    treatment = rng.choice([
        "in muted earth tones", "drenched in neon", "with bold geometric type", "in high-key white",
        "with dramatic chiaroscuro", "in a washed-out film palette", "with a single accent color",
        "layered over halftone texture", "with sweeping negative space", "in saturated technicolor",
        "with a hand-painted retro feel", "under a torn-paper collage treatment",
    ])
    mood = rng.choice(MOODS)
    art = aan(mood)
    forms = [
        f"{art} {mood} {fmt} featuring {subj} against {setting}, {treatment}",
        f"{art} {mood} {fmt} — {subj}, {treatment}",
        f"{art} {mood} {fmt} of {subj} set against {setting}, {treatment}",
        f"{art} {mood} {fmt} built around {subj}, {treatment}",
    ]
    s = rng.choice(forms)
    return s[0].upper() + s[1:]


def gen_background(rng):
    environments = [
        "a sprawling neon cityscape", "a fog-drenched pine forest", "a windswept salt flat",
        "a derelict industrial yard", "a sunlit marble atrium", "a stormy open sea",
        "rolling lavender fields", "a cracked desert basin", "a rain-slicked back alley",
        "a snow-blanketed mountain pass", "a flooded subway platform", "a bustling night market",
        "an empty drive-in lot", "a glassy alpine lake", "a crumbling colonnade",
        "a field of golden wheat", "a tangle of overgrown ruins", "a chrome-and-glass skyline",
        "a quiet coastal cliff", "an endless library of stacks",
    ]
    studio = [
        "a seamless gradient studio backdrop in coral", "a flat matte studio sweep in deep teal",
        "a hand-painted muslin backdrop", "a paper roll backdrop in dusty rose",
        "a textured concrete wall", "a sheet of brushed steel", "a wall of weathered velvet drapery",
        "a soft out-of-focus bokeh field", "a single flat color field with subtle vignette",
        "a torn-poster collage wall", "a riso-printed halftone gradient",
    ]
    times = [
        "at dusk", "at golden hour", "under a pale moon", "at first light", "in the blue hour",
        "under a bruised storm sky", "at high noon", "beneath a sky of falling ash",
        "under a sea of stars", "in thick rolling fog", "during a slow rain", "in dying amber light",
        "under a flat overcast sky", "as the last light fades", "in the cold of pre-dawn",
        "under a sky streaked with contrails",
    ]
    treatments = [
        "rendered in soft focus", "with deep negative space", "wrapped in low haze",
        "in muted desaturated tones", "with long raking shadows", "in crisp deep focus",
        "with a heavy atmospheric perspective", "drained of all but one color",
        "with a faint film grain", "lit from a single warm source", "in cool cinematic teal",
        "with dust motes drifting in the light",
    ]
    forms = [
        f"{rng.choice(environments)} {rng.choice(times)}",
        f"{rng.choice(studio)}",
        f"{rng.choice(environments)} {rng.choice(times)}, {rng.choice(treatments)}",
        f"{rng.choice(environments)}, {rng.choice(treatments)}",
        f"{rng.choice(studio)} with a faint vignette",
        f"{rng.choice(environments)} {rng.choice(times)} — {rng.choice(treatments)}",
    ]
    s = rng.choice(forms)
    return s


def gen_style_detail(rng):
    body = rng.choice([
        "Kodak Portra 400", "Fujifilm Pro 400H", "Cinestill 800T", "Ilford HP5", "Kodak Ektar 100",
        "Kodak Tri-X 400", "Fujifilm Velvia 50", "Lomography Color 800", "Polaroid SX-70 instant film",
        "Hasselblad 500CM medium format", "a Leica M6 rangefinder", "a Mamiya RZ67",
        "a Contax T2", "a large-format 4x5 view camera", "a Pentax 67", "a Canon AE-1",
    ])
    lens = rng.choice([
        "35mm f/1.4", "50mm f/1.2", "85mm f/1.4", "24mm f/2.8 wide angle", "135mm f/2 telephoto",
        "an anamorphic 40mm", "a vintage Helios 44-2", "a tilt-shift 90mm", "a 105mm macro",
        "a soft-focus portrait lens", "a 28mm street lens",
    ])
    technique = rng.choice([
        "shallow depth of field", "deep focus and crisp detail", "heavy film grain",
        "a long-exposure motion blur", "anamorphic lens flare", "soft natural window light",
        "a slight light leak", "subtle chromatic aberration", "a creamy bokeh background",
        "punchy contrast and deep blacks", "faded, lifted shadows", "a cross-processed color shift",
    ])
    forms = [
        f"shot on {body}, {lens}, {technique}",
        f"{body}, {lens}",
        f"{body} with {technique}",
        f"{lens} on {body}, {technique}",
        f"shot on {body}, {technique}",
        f"{body}, {lens}, {technique}",
    ]
    return rng.choice(forms)


def gen_aesthetics(rng):
    movements = [
        "minimalist", "maximalist", "brutalist", "baroque", "art deco", "art nouveau",
        "bauhaus", "vaporwave", "cottagecore", "cyberpunk", "retro-futurist", "mid-century modern",
        "swiss international", "psychedelic", "memphis design", "gothic", "rococo", "constructivist",
        "Y2K chrome", "dark academia", "solarpunk", "grunge", "kitsch", "minimal scandinavian",
    ]
    qualities = [
        "high-contrast", "muted", "saturated", "grainy", "clean", "ornate", "moody", "airy",
        "gritty", "polished", "hand-crafted", "editorial", "playful", "austere", "opulent",
        "weathered", "sleek", "raw", "dreamy", "geometric", "organic", "monochromatic", "neon-soaked",
    ]
    m1, m2 = rng.sample(movements, 2)
    q1, q2 = rng.sample(qualities, 2)
    forms = [
        f"{m1}, {q1}, {q2}",
        f"{q1} {m1} aesthetic",
        f"{m1} with {aan(q1)} {q1} edge",
        f"{q1}, {m1}-inspired",
        f"{m1} meets {m2}",
        f"{q1} and {q2}, {m1}",
    ]
    return rng.choice(forms)


def gen_lighting(rng):
    quality = rng.choice([
        "soft diffused", "hard directional", "dramatic", "moody low-key", "bright high-key",
        "dappled", "flat even", "harsh midday", "warm ambient", "cold clinical", "flickering",
        "gauzy", "razor-sharp", "glowing volumetric",
    ])
    source = rng.choice([
        "window light", "rim light", "backlight", "neon glow", "candlelight", "studio strobe",
        "golden-hour sun", "moonlight", "firelight", "fluorescent tube light", "a single spotlight",
        "city light spill", "overcast skylight", "a colored gel wash", "practical lamp light",
    ])
    extra = rng.choice([
        "with deep shadows", "with a soft falloff", "and a long cast shadow", "with lens flare",
        "carving out the silhouette", "wrapping gently around the subject", "raking across the surface",
        "pooling in the highlights", "with crushed blacks", "and a faint haze in the air",
        "splitting the face in two", "catching the edges in gold",
    ])
    forms = [
        f"{quality} {source} {extra}",
        f"{quality} {source}",
        f"{source}, {extra}",
        f"{quality} {source}, {extra}",
    ]
    return rng.choice(forms)


def gen_medium(rng):
    media = [
        "oil on canvas", "watercolor on cold-press paper", "gouache illustration", "acrylic on board",
        "digital painting", "3D render", "charcoal sketch", "ink and brush drawing", "linocut print",
        "screen print", "risograph print", "cut-paper collage", "mixed-media collage",
        "35mm film photograph", "medium-format film photograph", "wet-plate collodion tintype",
        "pencil illustration", "pastel drawing", "vector illustration", "pixel art",
        "airbrush illustration", "stained glass", "ceramic relief", "bronze sculpture",
        "embroidered textile", "woodblock print", "chalk pastel", "marker rendering",
        "matte painting", "claymation still", "papercraft diorama", "etching",
    ]
    qualifier = rng.choice([
        "", "", "", "with visible brushwork", "with bold ink outlines", "in a loose, gestural style",
        "with fine cross-hatching", "with a tactile paper texture", "in flat graphic shapes",
        "with rich impasto texture", "with halftone dot shading", "with a hand-drawn quality",
        "in a painterly finish", "with crisp vector edges", "with bleeding watercolor edges",
        "in a limited two-color palette", "with heavy black ink and spot color",
        "with a grainy printed texture", "in delicate fine-line detail", "with bold flat color blocks",
        "with smudged charcoal edges", "in a naive folk-art style", "with a glossy enamel finish",
        "with raised, textured strokes", "in soft blended gradients", "with a worn, distressed surface",
        "with scratchy dry-brush marks", "in a high-detail hyperreal finish", "with a chalky matte texture",
        "with confident gestural linework",
    ])
    surface = rng.choice([
        "aged paper", "raw linen", "weathered board", "textured canvas", "torn newsprint",
        "kraft cardboard", "vellum", "wet plaster", "reclaimed wood", "handmade cotton rag",
        "frosted glass", "brushed metal",
    ])
    m = rng.choice(media)
    forms = [
        f"{m}, {qualifier}" if qualifier else m,
        f"{m} on {surface}",
        m,
        f"{m}, {qualifier}" if qualifier else f"{m} on {surface}",
    ]
    return rng.choice(forms).rstrip(", ")


# Curated, harmonious hex pools by theme -> sample 2-5 per line for style_palette.
PALETTE_THEMES = {
    "noir": ["#0B0B0D", "#1A1A1E", "#2E2E33", "#4A4A52", "#7C7C85", "#B0B0B8", "#E6E6EA", "#C0392B"],
    "sunset": ["#2B1B3A", "#4A1942", "#812F4F", "#C04A5B", "#E8714E", "#F4A259", "#F6C453", "#FCE38A"],
    "forest": ["#0E1B12", "#1C3325", "#2F4F3E", "#4B7355", "#7BA05B", "#A7C957", "#DAD7B0", "#6B4226"],
    "ocean": ["#04293A", "#064663", "#06657C", "#2C7DA0", "#61A5C2", "#A9D6E5", "#CAF0F8", "#013A63"],
    "ember": ["#1A0A06", "#3D1308", "#6B240F", "#9E2A2B", "#D1410C", "#F25C05", "#FB923C", "#FCD34D"],
    "pastel": ["#FDE2E4", "#FAD2E1", "#E2ECE9", "#BEE1E6", "#CDDAFD", "#DFE7FD", "#FFF1E6", "#EDDCD2"],
    "desert": ["#2A1A0E", "#5C3D2E", "#8B5E3C", "#C08552", "#DDB892", "#E6CCB2", "#EDE0D4", "#7F4F24"],
    "cyber": ["#0B0C10", "#1F2833", "#0F3460", "#16213E", "#533483", "#E94560", "#00FFD1", "#66FCF1"],
    "vintage": ["#3E2723", "#6D4C41", "#8D6E63", "#A1887F", "#D7CCC8", "#EFEBE9", "#B0413E", "#C9A227"],
    "mono_blue": ["#03045E", "#023E8A", "#0077B6", "#0096C7", "#48CAE4", "#90E0EF", "#ADE8F4", "#CAF0F8"],
    "candy": ["#FF4D6D", "#FF7096", "#FF99AC", "#FFB3C1", "#C9184A", "#800F2F", "#FFCCD5", "#590D22"],
    "moss": ["#283618", "#3A5A40", "#588157", "#A3B18A", "#DAD7CD", "#606C38", "#BC6C25", "#FEFAE0"],
    "ash_gold": ["#0D0D0D", "#1C1C1C", "#2B2B2B", "#3D3D3D", "#5C5C5C", "#C9A227", "#E0C080", "#F2E2B1"],
    "ice": ["#CAF0F8", "#ADE8F4", "#90E0EF", "#48CAE4", "#FFFFFF", "#E0FBFC", "#C2DFE3", "#9DB4C0"],
    "wine": ["#3C091E", "#5C0A2E", "#7A1140", "#9B1B4A", "#B33951", "#E5989B", "#FFB4A2", "#2B0A12"],
    "jungle_neon": ["#011627", "#013A20", "#1A7431", "#25A244", "#2DC653", "#FFD60A", "#FF6B6B", "#06FFA5"],
    "clay_terracotta": ["#582F0E", "#7F4F24", "#936639", "#A68A64", "#B6AD90", "#C2C5AA", "#E9C46A", "#9C6644"],
    "muted_editorial": ["#22223B", "#4A4E69", "#9A8C98", "#C9ADA7", "#F2E9E4", "#8E9AAF", "#CBC0D3", "#DEE2FF"],
}


def gen_palette(rng):
    theme = rng.choice(list(PALETTE_THEMES.keys()))
    pool = PALETTE_THEMES[theme]
    k = rng.choice([2, 3, 3, 4, 4, 5])
    k = min(k, len(pool))
    cols = rng.sample(pool, k)
    return ", ".join(cols)


GENERATORS = {
    "title": gen_title,
    "subtitle": gen_subtitle,
    "hero": gen_hero,
    "body": gen_body,
    "brand": gen_brand,
    "extra": gen_extra,
    "high_level_description": gen_high_level,
    "background": gen_background,
    "style_detail": gen_style_detail,
    "aesthetics": gen_aesthetics,
    "lighting": gen_lighting,
    "medium": gen_medium,
    "style_palette": gen_palette,
}


def build_file(name, fn, out_dir):
    rng = random.Random(hash((SEED, name)) & 0xFFFFFFFF)
    seen = set()
    lines = []
    attempts = 0
    max_attempts = LINES_PER_FILE * 400
    while len(lines) < LINES_PER_FILE and attempts < max_attempts:
        attempts += 1
        line = " ".join(fn(rng).split()).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    if len(lines) < LINES_PER_FILE:
        raise RuntimeError(
            "%s: only produced %d/%d unique lines — widen the vocab banks."
            % (name, len(lines), LINES_PER_FILE)
        )
    path = os.path.join(out_dir, name + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path, len(lines)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "user", "wildcards", "composition_cowboy",
    )
    os.makedirs(out_dir, exist_ok=True)
    print("Writing wildcards to: %s\n" % out_dir)
    for name, fn in GENERATORS.items():
        path, count = build_file(name, fn, out_dir)
        print("  %-26s %4d lines  ->  __composition_cowboy/%s__" % (name + ".txt", count, name))
    print("\nDone. Reference any of these in a Mikey prompt-with-wildcards node.")


if __name__ == "__main__":
    main()
