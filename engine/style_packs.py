"""Curated style packs for auto-styling — refreshed 2026-06-04 (rev 2).

8 packs differentiated by FONT + HIGHLIGHT COLOR. Every pack uses
``subtitle_mode="word_only"`` (karaoke single-word swap) because:
  * full-phrase / progressive-reveal modes were overflowing the frame
    on narrow viewports, with single lines spanning the full short width;
  * "one word at a time" keeps the focal point tight and never wraps;
  * subtitle font sizes shrunk ~15% (~68-78 px range, was 80-92 px) so
    the single word doesn't dominate the safe area.

If you need a multi-word reveal style for a specific pack later, change
its ``subtitle_mode`` to ``word_highlight`` / ``word_bg`` / ``word_reveal`` /
``phrase`` — the renderer supports all five. The auto-picker reads
``pack.subtitle_mode`` so any change here flows through automatically.

LOCKED CATALOG:
  hormozi          Montserrat 900    #F7C204 yellow    educational / podcast
  reaction_pop     Anton             #FF1F1F red       drama / reaction
  neon_cyber       Montserrat 900    #02FB23 green     tutorial / growth
  vlog_warm        Permanent Marker  #FF7A00 orange    vlog / IRL / cozy
  esports_tactical Black Ops One     #A855F7 purple    esports / gaming
  cinema_aesthetic Montserrat 700    #E0B65C gold      cinematic / aesthetic
  podcast_clean    Bebas Neue        #00E5FF cyan      tech / business
  comedy_pop       Bangers           #FF2BD6 hot pink  comedy / cartoon

Why this design:
  - User rejected phrase-mode pills, word_bg per-word pills, word_only
    single-word karaoke, word_reveal buildup, and ALL non-word-highlight
    modes across 6+ design iterations. The simple Hormozi formula
    (white + 1 colored emphasis word) is the only pattern that lands.
  - Different fonts give each pack distinct identity without breaking
    the canonical viral pattern.
  - Subtitle sizes 80-92, hook sizes 86-98, outlines 8-10px at 1920-tall
    canvas. Smaller sizes / thinner weights got rejected as illegible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"


def _h(s: str) -> tuple[int, int, int]:
    """Hex string → (r, g, b) tuple. Tolerates leading '#'."""
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# ---------------------------------------------------------------------------
# Font registry (path + variable-font weight axis if any)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FontEntry:
    path: str
    axes: tuple[int, ...] | None = None


FONTS: dict[str, FontEntry] = {
    # Each pack uses a distinct typeface so no two packs read as the same
    # caption school - the 5 viral-tier fonts that were sitting unused in
    # /assets/fonts (Bebas Neue, Archivo Black, Bangers, Black Ops One,
    # Permanent Marker) are now wired into the catalog alongside the
    # original Anton/Montserrat/Inter trio.

    # IMPACT-substitute condensed caps. Used by reaction_pop.
    "anton":            FontEntry(str(_FONT_DIR / "Anton-Regular.ttf")),
    # HORMOZI-canonical heavy sans (weight 900). Used by hormozi, neon_cyber.
    "montserrat_900":   FontEntry(str(_FONT_DIR / "Montserrat-VF.ttf"), axes=(900,)),
    # Lighter humanist for cinematic restraint. Used by cinema_aesthetic.
    "montserrat_700":   FontEntry(str(_FONT_DIR / "Montserrat-VF.ttf"), axes=(700,)),
    # Modern grotesque variants. Kept for any future minimal pack.
    "inter_900":        FontEntry(str(_FONT_DIR / "Inter-VF.ttf"), axes=(900,)),
    "inter_700":        FontEntry(str(_FONT_DIR / "Inter-VF.ttf"), axes=(700,)),
    "inter_500":        FontEntry(str(_FONT_DIR / "Inter-VF.ttf"), axes=(500,)),

    # NEW — the previously-unused viral fonts:
    # Bebas Neue — THE most-used TikTok caption font. Tall condensed caps.
    "bebas":            FontEntry(str(_FONT_DIR / "BebasNeue-Regular.ttf")),
    # Archivo Black — heavy display sans, broader than Anton.
    "archivo_black":    FontEntry(str(_FONT_DIR / "ArchivoBlack-Regular.ttf")),
    # Bangers — comic-book ALL CAPS, used by Disney+/animation creators.
    "bangers":          FontEntry(str(_FONT_DIR / "Bangers-Regular.ttf")),
    # Black Ops One — military/tactical stencil. Gaming/esports identity.
    "black_ops":        FontEntry(str(_FONT_DIR / "BlackOpsOne-Regular.ttf")),
    # Permanent Marker — handwritten Sharpie. Vlog/journal/IRL identity.
    "permanent_marker": FontEntry(str(_FONT_DIR / "PermanentMarker-Regular.ttf")),
    # Bungee kept for backwards compat (used by old comedy_pop builds).
    "bungee":           FontEntry(str(_FONT_DIR / "Bungee-Regular.ttf")),

    # Aliases kept for backward compatibility.
    "inter":           FontEntry(str(_FONT_DIR / "Inter-VF.ttf"), axes=(700,)),
    "inter_medium":    FontEntry(str(_FONT_DIR / "Inter-VF.ttf"), axes=(500,)),
    "montserrat":      FontEntry(str(_FONT_DIR / "Montserrat-VF.ttf"), axes=(800,)),
}


# ---------------------------------------------------------------------------
# Style packs
# ---------------------------------------------------------------------------

SUBTITLE_MODES = ("phrase", "word_highlight", "word_bg", "word_reveal", "word_only")


@dataclass(frozen=True)
class StylePack:
    name: str
    descriptor: str   # VLM-facing vibe hint
    font_key: str
    subtitle_mode: str

    subtitle_fill: tuple[int, int, int]
    subtitle_highlight: tuple[int, int, int]
    subtitle_bg: tuple[int, int, int] | None   # per-word pill (word_bg mode)

    caption_fill: tuple[int, int, int]
    caption_bg:   tuple[int, int, int] | None  # pill behind hook text

    # PHRASE-level bg pill behind the whole subtitle line. Set for packs
    # whose font is thin and whose ``subtitle_bg`` is None (no per-word
    # pill to lean on for legibility). 50% black is the safe default —
    # readable on any background, doesn't compete with the highlight
    # color, and stays out of the way on dark scenes.
    subtitle_phrase_bg: tuple[int, int, int] | None = None

    # Outline thickness in px (at 1920-tall canvas). 3px reads as a normal
    # outline; 7-9px is the "MrBeast-style" thick stroke that lets text float
    # over the video without needing a pill background. Per-pack so the
    # MrBeast/Hormozi-style hype packs can crank it while cinema/podcast
    # stay restrained.
    subtitle_outline_width: int = 3
    subtitle_outline_color: tuple[int, int, int] = (0, 0, 0)

    # Alpha (0-255) for the subtitle phrase-level background pill. Default 128
    # is the "50% legibility shadow" used by minimal packs. Set to 255 when the
    # pill is part of the visual identity (e.g. opaque white "comment" pill
    # for dialog-style captions, opaque black banner for commentary captions).
    subtitle_phrase_bg_alpha: int = 128
    # Same for the hook caption pill. Defaults to the engine's 220 (mostly opaque).
    hook_bg_alpha: int = 220

    # Per-pack font size at 1920-tall canvas. 76 is the engine default.
    # Reference examples vary widely - ex 5/6/8 are big (90-110), ex 9 is tiny
    # (60-65), ex 1/7/10 are mid (70-80). Per-pack so each pack matches its
    # reference's visual weight.
    subtitle_font_size: int = 76
    hook_font_size: int = 92

    # ALL CAPS the rendered text regardless of source casing. Hormozi /
    # reaction_pop / comedy_pop visual identities depend on uppercase -
    # without it they read as generic bold captions, not the viral look.
    # Cinematic / dialog packs (cinema_aesthetic, vlog_warm, podcast_clean)
    # leave this False so the on-app sentence-case look is preserved.
    uppercase: bool = False

    # Hook entrance motion, layered on the existing fade-in. One of
    # "fade" | "pop" | "slide" | "pop_slide". Motion is orthogonal to the
    # locked font+color identity — it gives each school a matching ENERGY
    # (punchy schools pop, cinematic fades, comedy bounces) without changing
    # what the caption looks like at rest.
    hook_anim: str = "pop"


PACKS: dict[str, StylePack] = {
    # 2026-06-04 rev 2 — all 8 packs unified on ``word_only`` (one word
    # at a time, karaoke swap). The previous 5-mode mix was overflowing
    # frame width on narrow viewports; word_only is the only mode that
    # never wraps regardless of font or word length. Font sizes
    # tightened ~15% so the single word doesn't dominate the safe area.
    "hormozi": StylePack(
        # Hormozi style (replaces gaming_hype). Source: research consensus
        # across Sendshort/Submagic/Caply guides + Hormozi's own channel.
        # Montserrat at weight 900, ALL CAPS, white text + Hormozi-canonical
        # yellow (#F7C204) on emphasized word, THICK black stroke (10px).
        # Best-performing style for educational / business / interview content;
        # documented +15% engagement vs flat captions.
        name="hormozi",
        descriptor=(
            "Alex Hormozi style: educational, business, podcast, sales talk. "
            "Heavy Montserrat ALL CAPS in white with the canonical Hormozi "
            "yellow on emphasized words, thick black stroke. Bottom-middle "
            "third of frame. The highest-engagement caption style for spoken "
            "content."
        ),
        font_key="montserrat_900",
        subtitle_mode="word_only",
        subtitle_fill=_h("#FFFFFF"),
        subtitle_highlight=_h("#F7C204"),    # HORMOZI-canonical yellow
        subtitle_bg=None,
        subtitle_outline_width=10,           # documented 8-12px range
        subtitle_font_size=72,
        hook_font_size=92,
        uppercase=True,
        hook_anim="pop",
        caption_fill=_h("#F7C204"),
        caption_bg=None,
    ),
    "podcast_clean": StylePack(
        # Bebas Neue + word-highlight. The classic TikTok caption font —
        # tall condensed caps, white text with vivid CYAN (#00E5FF)
        # emphasized word. Tech / business / podcast / SaaS energy.
        name="podcast_clean",
        descriptor=(
            "Tech / business / podcast / SaaS: tech reviews, business "
            "takes, podcast clips, productivity tutorials. Bebas Neue "
            "ALL CAPS in white with words appearing progressively, "
            "thick black stroke. The clean 'tech-stack' viral palette."
        ),
        font_key="bebas",
        subtitle_mode="word_only",
        subtitle_fill=_h("#FFFFFF"),
        subtitle_highlight=_h("#00E5FF"),
        subtitle_bg=None,
        subtitle_outline_width=10,
        subtitle_font_size=78,
        hook_font_size=98,
        uppercase=True,
        hook_anim="slide",
        hook_bg_alpha=245,
        caption_fill=_h("#15161A"),   # dark ink on the white hook card
        caption_bg=_h("#FFFFFF"),     # clean white card — modern "native" look
    ),
    "reaction_pop": StylePack(
        # Drama / reaction school. Word-highlight Anton, ALL CAPS, white +
        # vivid RED active word (#FF1F1F). Red is the canonical "drama /
        # reaction / shock-news" color — different from hormozi yellow's
        # educational / business signal. Same word-highlight formula but a
        # distinct emotional register.
        name="reaction_pop",
        descriptor=(
            "Drama / reaction / shock / sports calls: gameplay highlights, "
            "viral moments, news reads, fight calls. Anton ALL CAPS in white "
            "with the active word on a rounded-rect RED pill, thick black "
            "stroke. The canonical 'drama' palette — red pill pops on each "
            "spoken word."
        ),
        font_key="anton",
        subtitle_mode="word_only",
        subtitle_fill=_h("#FFFFFF"),
        subtitle_highlight=_h("#FF1F1F"),    # vivid drama red
        subtitle_bg=None,
        subtitle_outline_width=8,
        subtitle_font_size=74,
        hook_font_size=98,
        uppercase=True,
        hook_anim="pop",
        caption_fill=_h("#FF1F1F"),
        caption_bg=None,
    ),
    "vlog_warm": StylePack(
        # Permanent Marker + word-highlight. The handwritten-Sharpie vlog
        # caption school used by journal / IRL / day-in-the-life creators.
        # White marker text with a warm ORANGE (#FF7A00) emphasized word -
        # reads as personal / hand-drawn / cozy.
        name="vlog_warm",
        descriptor=(
            "Vlog / IRL / journal / day-in-the-life / cozy storytelling: "
            "personal takes, life updates, food trips, travel journals, "
            "satisfying daily clips. WHITE handwritten Permanent Marker "
            "ALL CAPS with words appearing progressively in warm orange — "
            "feels hand-drawn and personal."
        ),
        font_key="permanent_marker",
        subtitle_mode="word_only",
        subtitle_fill=_h("#FFFFFF"),
        subtitle_highlight=_h("#FF7A00"),
        subtitle_bg=None,
        subtitle_outline_width=8,
        subtitle_font_size=68,
        hook_font_size=88,
        uppercase=True,
        hook_anim="slide",
        caption_fill=_h("#FF7A00"),
        caption_bg=None,
    ),
    "esports_tactical": StylePack(
        # Black Ops One + word-highlight. Military / tactical stencil font
        # for esports / gaming / FPS content. White stencil text + vivid
        # PURPLE (#A855F7) emphasized word - reads as gaming / cyber / loadout.
        name="esports_tactical",
        descriptor=(
            "Esports / gaming / FPS / loadout / Twitch highlights: shooter "
            "clips, ranked plays, tournament moments, gaming setups. "
            "WHITE Black Ops One military stencil ALL CAPS with the "
            "emphasized word in vivid purple, thick black stroke. The "
            "tactical / gaming caption palette."
        ),
        font_key="black_ops",
        subtitle_mode="word_only",
        subtitle_fill=_h("#FFFFFF"),
        subtitle_highlight=_h("#A855F7"),
        subtitle_bg=None,
        subtitle_outline_width=10,
        subtitle_font_size=72,
        hook_font_size=92,
        uppercase=True,
        hook_anim="pop",
        caption_fill=_h("#A855F7"),
        caption_bg=None,
    ),
    "neon_cyber": StylePack(
        # Reference 11 — same Hormozi formula but GREEN active word instead
        # of yellow. White Montserrat 900 ALL CAPS, thick black stroke,
        # green emphasized word. Reads as tutorial / growth / SaaS /
        # productivity — green signals "go" / progress / money / growth.
        name="neon_cyber",
        descriptor=(
            "Tutorial / growth / SaaS / productivity / finance: how-to "
            "content, money tips, software walkthroughs, growth advice. "
            "Karaoke-style single-word swap in vivid green Montserrat "
            "ALL CAPS. The 'go / grow / progress' palette — one word at a "
            "time keeps focus on the active beat."
        ),
        font_key="montserrat_900",
        subtitle_mode="word_only",
        subtitle_fill=_h("#FFFFFF"),
        subtitle_highlight=_h("#02FB23"),
        subtitle_bg=None,
        subtitle_outline_width=10,
        subtitle_outline_color=_h("#000000"),
        subtitle_font_size=72,
        hook_font_size=92,
        uppercase=True,
        hook_anim="pop",
        caption_fill=_h("#02FB23"),
        caption_bg=None,
    ),
    "cinema_aesthetic": StylePack(
        # Montserrat 700 + word-highlight. Cream Netflix-subtitle-style
        # base color (#F5F1E8) with a warm GOLD (#E0B65C) emphasized word.
        # Slightly slimmer weight than hormozi for cinematic restraint,
        # but same thick stroke + ALL CAPS so it still POPS at preview scale.
        name="cinema_aesthetic",
        descriptor=(
            "Cinematic / aesthetic / mood / story / fashion: film-style "
            "clips, fashion edits, atmospheric content, music videos, "
            "narrative voiceovers. CREAM Montserrat ALL CAPS with whole "
            "phrases fading in and out — minimalist, film-title restraint."
        ),
        font_key="montserrat_700",
        subtitle_mode="word_only",
        subtitle_fill=_h("#F5F1E8"),
        subtitle_highlight=_h("#E0B65C"),
        subtitle_bg=None,
        subtitle_outline_width=8,
        subtitle_font_size=68,
        hook_font_size=86,
        uppercase=True,
        hook_anim="fade",
        hook_bg_alpha=245,
        caption_fill=_h("#1A1A1E"),   # dark ink on the white hook card
        caption_bg=_h("#FFFFFF"),     # clean white card — cinematic restraint
    ),
    "comedy_pop": StylePack(
        # Bangers + word-highlight. Comic-book / animation / kids creator
        # caption font (the Disney / cartoon-edit standard). White text +
        # HOT PINK (#FF2BD6) emphasized word - reads as comedy / playful /
        # animated. Same word-highlight formula as the hero packs but with
        # the most distinctly "fun" typeface in the catalog.
        name="comedy_pop",
        descriptor=(
            "Comedy / sketch / animation / kids / cartoon: comedy clips, "
            "animated reactions, cartoon edits, sketch beats, viral memes. "
            "WHITE Bangers comic-caps ALL CAPS with the active word on a "
            "hot-pink rounded pill — most playful + punchy combo in the catalog."
        ),
        font_key="bangers",
        subtitle_mode="word_only",
        subtitle_fill=_h("#FFFFFF"),
        subtitle_highlight=_h("#FF2BD6"),
        subtitle_bg=None,
        subtitle_outline_width=8,
        subtitle_font_size=74,
        hook_font_size=96,
        uppercase=True,
        hook_anim="pop_slide",
        caption_fill=_h("#FF2BD6"),
        caption_bg=None,
    ),
}


DEFAULT_PACK = "podcast_clean"


# Detection-category -> style-pack, so a clip's CAPTION COLOR matches its
# emotional theme instead of a blind rotation that painted reaction-red on
# every clip regardless of content. Categories are the fixed judge vocab
# (highlights/judge.py CATEGORIES): funny, hype, tense, awkward, sad,
# surprise, none. Each emotion picks the pack whose color/typeface reads
# that register:
#   funny    -> comedy_pop      hot-pink comic caps   (playful)
#   hype     -> reaction_pop    drama red             (shock / hype)
#   tense    -> esports_tactical purple stencil       (competitive tension)
#   awkward  -> vlog_warm       warm orange marker    (personal / cringe)
#   sad      -> cinema_aesthetic gold cinematic       (moody / restrained)
#   surprise -> neon_cyber      electric green        ("no way" pop)
#   none     -> hormozi         high-engagement yellow (safe talk default)
CATEGORY_PACK: dict[str, str] = {
    "funny": "comedy_pop",
    "hype": "reaction_pop",
    "tense": "esports_tactical",
    "awkward": "vlog_warm",
    "sad": "cinema_aesthetic",
    "surprise": "neon_cyber",
    "none": "hormozi",
}

# Gaming clips read better in the gaming-native palettes regardless of the
# emotional read, so when the streamer bucket is gameplay we steer the two
# ambiguous high-energy emotions toward the esports/cyber packs. Talk/IRL
# buckets keep the emotional default above. (Buckets per highlights/profiles.)
_GAMING_BUCKETS = {"gameplay", "esports", "fps", "gaming"}
_GAMING_OVERRIDE: dict[str, str] = {
    "hype": "neon_cyber",
    "surprise": "esports_tactical",
}


def pack_for_category(category: str | None, bucket: str | None = None) -> str:
    """Pick a style pack whose color/vibe matches the clip's theme.

    ``category`` is the detector's emotional read; ``bucket`` is the
    streamer profile (optional). Falls back to ``hormozi`` for unknown
    categories — the safe high-engagement default for spoken content.
    """
    cat = (category or "none").strip().lower()
    bkt = (bucket or "").strip().lower()
    if bkt in _GAMING_BUCKETS and cat in _GAMING_OVERRIDE:
        return _GAMING_OVERRIDE[cat]
    return CATEGORY_PACK.get(cat, "hormozi")


def get_pack(name: str) -> StylePack:
    """Look up a pack by name. Falls back to DEFAULT_PACK if unknown."""
    return PACKS.get(name, PACKS[DEFAULT_PACK])


def font_path_for_pack(pack: StylePack) -> str:
    return FONTS[pack.font_key].path


def font_axes_for_pack(pack: StylePack) -> tuple[int, ...] | None:
    return FONTS[pack.font_key].axes
