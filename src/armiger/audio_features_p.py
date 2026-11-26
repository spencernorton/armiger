"""Strictly P(x)-derived deterministic audio-only features (ZERO subtitle inputs -> 0.000000 bitwise invariance!)."""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

AUDIO_P_FEAT_DIM = 24
AUDIO_BYTE_L = 128

# Explicit SPOKEN audio cues (matched STRICTLY on P(x) after all subtitle tokens/files are stripped!)
_RE_DUAL_MULTI = re.compile(r"(?<![a-z0-9])(dual[\s._-]?audio|multi[\s._-]?audio|multi\d?|dual\b|bilingual|2audio|3audio)(?![a-z0-9])", re.I)
_RE_ENG_EXPLICIT = re.compile(r"(?<![a-z0-9])(english|eng[\s._-]?(?:dub|audio|ac3|dts|aac|5\.1|2\.0)|eng\b|en[\s._-]?us|en[\s._-]?gb)(?![a-z0-9])", re.I)
_RE_DUB = re.compile(r"(?<![a-z0-9])(dubbed|dub\b|engdub|english[\s._-]?dub)(?![a-z0-9])", re.I)
_RE_FRENCH = re.compile(r"(?<![a-z0-9])(truefrench|french|vfq|vff|vfi|vf2|vf\b|fre\b|fra\b|francais)(?![a-z0-9])", re.I)
_RE_SPANISH = re.compile(r"(?<![a-z0-9])(spanish|castellano|latino|espanol|esp\b|spa\b|lat\b|dual[\s._-]?lat)(?![a-z0-9])", re.I)
_RE_GERMAN = re.compile(r"(?<![a-z0-9])(german|deutsch|ger\b|deu\b|dl[\s._-]?1080p|dl[\s._-]?720p|dl[\s._-]?2160p)(?![a-z0-9])", re.I)
_RE_ITALIAN = re.compile(r"(?<![a-z0-9])(italian|italiano|ita\b|sub[\s._-]?ita)(?![a-z0-9])", re.I)
_RE_RUSSIAN = re.compile(r"(?<![a-z0-9])(russian|rus\b|ru\b|dub[\s._-]?rus|mvo|dvo|avo|lostfilm|hdrezka|kubik|baibako|newstudio|coldfilm|rutor)(?![a-z0-9])", re.I)
_RE_INDIAN = re.compile(r"(?<![a-z0-9])(hindi|tamil|telugu|malayalam|kannada|punjabi|marathi|bengali|gujarati|bollywood|hin\b|tam\b|tel\b|ddp[\s._-]?5[\s._-]?1[\s._-]?hin)(?![a-z0-9])", re.I)
_RE_EAST_ASIAN = re.compile(r"(?<![a-z0-9])(japanese|jpn\b|jap\b|korean|kor\b|chinese|mandarin|cantonese|chs\b|cht\b|cmn\b|yue\b|raws?)(?![a-z0-9])", re.I)
_RE_OTHER_EU = re.compile(
    r"(?<![a-z0-9])(polish|pl\b|lektor|dubbing[\s._-]?pl|dutch|nl\b|flemish|portuguese|pt[\s._-]?br|dublado|nacional|"
    r"turkish|tr\b|tur\b|swedish|swe\b|norwegian|nor\b|danish|dan\b|finnish|fin\b|czech|cz\b|cze\b|slovak|sk\b|"
    r"hungarian|hun\b|romanian|rom\b|greek|gre\b|ukrainian|ukr\b|arabic|ara\b|persian|farsi|hebrew|heb\b|thai|vietnamese|indonesian)(?![a-z0-9])",
    re.I,
)
_RE_EN_SCENE = re.compile(
    r"(?<![a-z0-9])(yts|yify|rarbg|eztv|ettv|tgx|galaxyrg|ntb|flux|sparks|geckos|successfulcrab|cmrg|successful|"
    r"framestor|epsilon|tigole|qxr|psa|pahe|meGusta|syncopy|playweb|bhdstudio|hulu|amzn|atvp|dsnp|max|peacock)(?![a-z0-9])",
    re.I,
)
_RE_NON_EN_SCENE = re.compile(
    r"(?<![a-z0-9])(tntvillage|mircrew|ilcorsaronero|ygg|fw|zone[\s._-]?telechargement|mejortorrent|divxtotal|dontorrent|"
    r"elitetorrent|cinecalidad|gnula|wolfmax|serienjunkies|movieblog|hdsource|selezen|exkinoray|mega[\s._-]?peer|nnmclub|"
    r"rutracker|kinozal|tfile|1tamilmv|1tamilblasters|katmoviehd|vegamovies|bollyflix|hdhub4u|skymovieshd|dramaday|avistaz)(?![a-z0-9])",
    re.I,
)


def encode_audio_p_bytes(audio_p_text: str, max_l: int = AUDIO_BYTE_L) -> np.ndarray:
    """Raw UTF-8 bytes of P(x) ONLY (head 88 + tail 40 for long strings)."""
    b = (audio_p_text or "").encode("utf-8", errors="ignore")
    out = np.zeros(max_l, dtype=np.int64)
    n = len(b)
    if n == 0:
        return out
    if n <= max_l:
        out[:n] = np.frombuffer(b, dtype=np.uint8).astype(np.int64) + 1
    else:
        out[:88] = np.frombuffer(b[:88], dtype=np.uint8).astype(np.int64) + 1
        out[88:] = np.frombuffer(b[-(max_l - 88):], dtype=np.uint8).astype(np.int64) + 1
    return out


def extract_audio_p_features(audio_p_text: str) -> np.ndarray:
    """24-dim strictly P(x)-derived Unicode script + spoken-audio cue vector (ZERO subtitle signals!)."""
    out = np.zeros(AUDIO_P_FEAT_DIM, dtype=np.float32)
    s = audio_p_text or ""
    if not s:
        return out

    n_alpha = 0
    c_ascii_lat = c_acc_lat = c_cyr = c_cjk = c_hangul = c_dev = c_arab = c_other = 0
    for ch in s:
        if not ch.isalpha():
            continue
        n_alpha += 1
        o = ord(ch)
        if o < 128:
            c_ascii_lat += 1
        elif 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF:
            c_acc_lat += 1
        elif 0x0400 <= o <= 0x052F:
            c_cyr += 1
        elif (0x4E00 <= o <= 0x9FFF) or (0x3040 <= o <= 0x30FF):
            c_cjk += 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            c_hangul += 1
        elif 0x0900 <= o <= 0x0D7F:  # Indic scripts (Devanagari, Bengali, Tamil, Telugu, Malayalam, Kannada)
            c_dev += 1
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            c_arab += 1
        else:
            c_other += 1

    denom = max(1, n_alpha)
    out[0] = c_ascii_lat / denom
    out[1] = c_acc_lat / denom
    out[2] = c_cyr / denom
    out[3] = c_cjk / denom
    out[4] = c_hangul / denom
    out[5] = c_dev / denom
    out[6] = c_arab / denom
    out[7] = c_other / denom
    out[8] = 1.0 if (c_cyr + c_cjk + c_hangul + c_dev + c_arab + c_other) >= 2 else 0.0
    out[9] = 1.0 if c_acc_lat >= 2 else 0.0
    out[10] = math.log1p(len(s)) / 6.0

    # Explicit spoken-audio markers strictly on P(x)
    out[11] = 1.0 if _RE_DUAL_MULTI.search(s) else 0.0
    out[12] = 1.0 if _RE_ENG_EXPLICIT.search(s) else 0.0
    out[13] = 1.0 if _RE_DUB.search(s) else 0.0
    out[14] = 1.0 if _RE_FRENCH.search(s) else 0.0
    out[15] = 1.0 if _RE_SPANISH.search(s) else 0.0
    out[16] = 1.0 if _RE_GERMAN.search(s) else 0.0
    out[17] = 1.0 if _RE_ITALIAN.search(s) else 0.0
    out[18] = 1.0 if _RE_RUSSIAN.search(s) else 0.0
    out[19] = 1.0 if _RE_INDIAN.search(s) else 0.0
    out[20] = 1.0 if _RE_EAST_ASIAN.search(s) else 0.0
    out[21] = 1.0 if _RE_OTHER_EU.search(s) else 0.0
    out[22] = 1.0 if _RE_EN_SCENE.search(s) else 0.0
    out[23] = 1.0 if _RE_NON_EN_SCENE.search(s) else 0.0
    return out


# ---------------------------------------------------------------------------
# Strictly P(x)-Only Multilingual Stopword + Diacritic + Audiobook Slug Cues (20-dim)
# ---------------------------------------------------------------------------
AUDIO_P_EXTRA_DIM = 20

_P_TOK_RE = re.compile(r"[a-z]+")
_P_AUDIO_ONLY_EXTS = frozenset({".mp3", ".m4b", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wav", ".wma", ".alac"})
_P_LIBRIVOX_AB_RE = re.compile(
    r"(?i)(?:_librivox\b|\blibrivox\b|\baudiobook\b|\bunabridged\b|\bh[oö]rbuch\b|\bhoerbuch\b|"
    r"\blivre[\s._-]*audio\b|\baudiolibro\b|\baudiolivro\b|\baudiokniga\b|аудиокнига|"
    r"\bnarrated[\s._-]*by\b|\bread[\s._-]*by\b|\bspoken[\s._-]*word\b|\bgutenberg\b)"
)
_P_STRICT_EN_DUB_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:dual[\s._-]*audio|multi[\s._-]*audio|eng(?:lish)?[\s._-]*dub(?:bed)?|"
    r"en[\s._-]*dub(?:bed)?|eng(?:lish)?[\s._-]*audio|dubbed[\s._-]*eng(?:lish)?|vose|trueenglish)(?![a-z0-9])"
)
_P_EN_STOP = frozenset({
    "the", "of", "and", "to", "in", "is", "for", "with", "on", "from", "that", "this", "your",
    "you", "my", "his", "her", "our", "their", "about", "how", "what", "when", "where", "story",
    "stories", "tales", "complete", "book", "volume", "part", "season", "series", "guide", "life",
    "world", "time", "love", "night", "dark", "dead", "house", "man", "woman", "king", "war",
    "american", "british", "english", "history", "adventures", "mystery", "murder", "secret",
})
_P_FR_STOP = frozenset({
    "le", "la", "les", "des", "une", "du", "dans", "avec", "pour", "sur", "par", "aux", "chez",
    "sans", "sous", "vers", "entre", "comme", "mais", "tout", "tous", "histoire", "histoires",
    "contes", "temps", "passe", "livre", "tome", "saison", "integrale", "francais", "francaise",
    "guerre", "amour", "homme", "femme", "nuit", "vie", "monde", "petit", "petite", "grande",
    "deux", "trois", "quatre", "vff", "vfq", "truefrench", "french", "vf", "fr", "edition", "oeuvres",
    "memoires", "lettres", "voyage", "chanson", "fables", "poemes", "roman", "chroniques",
})
_P_DE_STOP = frozenset({
    "der", "die", "das", "und", "ein", "eine", "einen", "einem", "einer", "von", "mit", "fur",
    "fuer", "auf", "aus", "bei", "nach", "uber", "ueber", "unter", "durch", "gegen", "ohne",
    "zwischen", "buch", "band", "staffel", "folge", "horbuch", "hoerbuch", "deutsch", "deutsche",
    "german", "geschichte", "geschichten", "welt", "leben", "zeit", "liebe", "nacht", "mann",
    "frau", "tod", "haus", "zwei", "drei", "marchen", "maerchen", "gedichte", "briefe", "reise",
    "roman", "novelle", "krimi", "tagebuch", "erzahlungen", "erzaehlungen",
})
_P_ES_STOP = frozenset({
    "el", "los", "las", "del", "una", "unos", "unas", "por", "con", "para", "sobre", "entre",
    "desde", "hasta", "hacia", "como", "pero", "todo", "todos", "historia", "cuentos", "libro",
    "temporada", "capitulo", "completo", "completa", "espanol", "castellano", "latino", "spanish",
    "vida", "mundo", "tiempo", "amor", "noche", "hombre", "mujer", "casa", "muerte", "dos", "tres",
    "novela", "poemas", "cartas", "viaje", "memorias", "cronicas",
})
_P_IT_STOP = frozenset({
    "il", "lo", "gli", "della", "delle", "degli", "dei", "dello", "nel", "nella", "nelle", "sul",
    "sulla", "per", "tra", "fra", "tutto", "tutti", "storia", "racconti", "stagione", "episodio",
    "italiano", "italiana", "italian", "ita", "vita", "mondo", "tempo", "amore", "notte", "uomo",
    "donna", "morte", "due", "tre", "fiabe", "poesie", "lettere", "viaggio", "romanzo", "commedia",
})
_P_PT_STOP = frozenset({
    "os", "as", "dos", "das", "uma", "umas", "pelo", "pela", "sem", "tudo", "portugues", "dublado",
    "nacional", "noite", "homem", "mulher", "dois", "contos", "poemas", "romance", "portuguese", "ptbr",
})
_P_NORDIC_NL_STOP = frozenset({
    "het", "een", "van", "voor", "met", "naar", "uit", "bij", "door", "over", "onder", "tegen",
    "zonder", "boek", "nederlands", "dutch", "nl", "och", "att", "med", "som", "till", "fran",
    "eller", "men", "inte", "bok", "svenska", "swedish", "norsk", "norwegian", "dansk", "danish",
    "suomi", "finnish", "sprookjes", "verhalen", "geschiedenis",
})
_P_SLAVIC_STOP = frozenset({
    "iz", "na", "po", "za", "ot", "do", "pri", "pro", "bez", "pod", "nad", "ili", "kak", "chto",
    "kniga", "sezon", "seriya", "russkiy", "russian", "rus", "polska", "polski", "polish", "lektor",
    "cesky", "czech", "slovak", "ukrainian", "ukr", "skazki", "rasskazy", "istoriya", "bajki",
})
_P_CJK_ROM_STOP = frozenset({
    "no", "wa", "ga", "wo", "ni", "kara", "made", "yori", "mono", "koto", "hito", "sekai",
    "monogatari", "densetsu", "senki", "mahou", "gakuen", "shoujo", "shounen", "tensei", "isekai",
    "kanojo", "boku", "ore", "watashi", "kimi", "anata", "shin", "dai", "hen", "gekijouban",
    "ova", "oad", "eiga", "dorama", "japanese", "jpn", "korean", "kor", "chinese", "mandarin",
    "cantonese", "donghua", "raws", "sarang", "namja", "yeoja", "haneul", "yume", "kiseki",
    "tsuki", "kaze", "hoshi", "kokoro", "sakura", "samurai", "ninja", "kami", "akuma", "ryuu",
})
_P_IN_TR_ID_STOP = frozenset({
    "ka", "ke", "ki", "ko", "se", "mein", "hai", "hain", "aur", "ek", "yeh", "woh", "pyar", "dil",
    "zindagi", "kahani", "bollywood", "hindi", "tamil", "telugu", "malayalam", "kannada", "marathi",
    "punjabi", "bengali", "bir", "ve", "bu", "icin", "ile", "turkce", "turkish", "dublaj", "bolum",
    "yang", "dan", "untuk", "dari", "dengan", "tidak", "indonesian", "arabic", "persian", "hebrew",
})
_FR_DIAC = frozenset("éèêëàâçùûôîïœæÉÈÊËÀÂÇÙÛÔÎÏŒÆ")
_DE_DIAC = frozenset("äöüßÄÖÜ")
_ES_PT_DIAC = frozenset("ñáíóúãõ¡¿ÑÁÍÓÚÃÕ")
_SLAV_NORD_DIAC = frozenset("čšžřěůłńśźżćąęåøğışČŠŽŘĚŮŁŃŚŹŻĆĄĘÅØĞİŞ")


def extract_audio_p_extra_20d(clean_title: str, media_names: list[str] | None = None) -> np.ndarray:
    """
    20-dim strictly P(x)-only multilingual stopword + diacritic + LibriVox/audiobook cue vector.
    Inputs MUST come strictly from `project_audio_only(name, files)` so subtitle invariance is exact.
    """
    out = np.zeros(AUDIO_P_EXTRA_DIM, dtype=np.float32)
    ct = clean_title or ""
    mn = media_names or []
    joined = ct if not mn else (ct + " " + " ".join(mn[:3]))
    # ASCII-folded lowercase tokens for stopword ratios
    if joined.isascii():
        folded = joined.lower()
    else:
        import unicodedata
        folded = unicodedata.normalize("NFKD", joined).encode("ascii", "ignore").decode("ascii").lower()
    toks = _P_TOK_RE.findall(folded)
    nt = max(1, len(toks))
    en_r = sum(1 for t in toks if t in _P_EN_STOP) / nt
    fr_r = sum(1 for t in toks if t in _P_FR_STOP) / nt
    de_r = sum(1 for t in toks if t in _P_DE_STOP) / nt
    es_r = sum(1 for t in toks if t in _P_ES_STOP) / nt
    it_r = sum(1 for t in toks if t in _P_IT_STOP) / nt
    pt_r = sum(1 for t in toks if t in _P_PT_STOP) / nt
    nn_r = sum(1 for t in toks if t in _P_NORDIC_NL_STOP) / nt
    sl_r = sum(1 for t in toks if t in _P_SLAVIC_STOP) / nt
    cj_r = sum(1 for t in toks if t in _P_CJK_ROM_STOP) / nt
    in_r = sum(1 for t in toks if t in _P_IN_TR_ID_STOP) / nt
    max_ne_r = max(fr_r, de_r, es_r, it_r, pt_r, nn_r, sl_r, cj_r, in_r)
    out[0] = en_r
    out[1] = fr_r
    out[2] = de_r
    out[3] = es_r
    out[4] = it_r
    out[5] = pt_r
    out[6] = nn_r
    out[7] = sl_r
    out[8] = cj_r
    out[9] = in_r
    # Diacritics (10..13)
    if not joined.isascii():
        ln = max(1, len(joined))
        denom = max(4.0, ln * 0.1)
        out[10] = min(1.0, sum(1 for ch in joined if ch in _FR_DIAC) / denom)
        out[11] = min(1.0, sum(1 for ch in joined if ch in _DE_DIAC) / denom)
        out[12] = min(1.0, sum(1 for ch in joined if ch in _ES_PT_DIAC) / denom)
        out[13] = min(1.0, sum(1 for ch in joined if ch in _SLAV_NORD_DIAC) / denom)
    # Audiobook / LibriVox / slug / spoken cues (14..19)
    out[14] = 1.0 if _P_LIBRIVOX_AB_RE.search(joined) else 0.0
    if mn:
        aud_only = all(any(m.lower().endswith(e) for e in _P_AUDIO_ONLY_EXTS) for m in mn[:4])
        out[15] = 1.0 if aud_only else 0.0
    ct_strip = ct.strip()
    out[16] = 1.0 if (len(ct_strip) >= 10 and " " not in ct_strip and ("_" in ct_strip or ct_strip.isalpha())) else 0.0
    out[17] = float(en_r - max_ne_r)
    out[18] = float(max_ne_r)
    out[19] = 1.0 if _P_STRICT_EN_DUB_RE.search(joined) else 0.0
    return out
