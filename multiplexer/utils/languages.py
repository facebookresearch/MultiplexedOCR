# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ftfy import fix_encoding

from .char_map_amharic import AmharicCharMap
from .char_map_any import AnyCharMap
from .char_map_any1 import Any1CharMap
from .char_map_any2 import Any2CharMap
from .char_map_any3 import Any3CharMap
from .char_map_any4 import Any4CharMap
from .char_map_any5 import Any5CharMap
from .char_map_any6 import Any6CharMap
from .char_map_any7 import Any7CharMap
from .char_map_arabic import ArabicCharMap
from .char_map_bengali import BengaliCharMap
from .char_map_bulgarian import BulgarianCharMap
from .char_map_burmese import BurmeseCharMap
from .char_map_chinese import ChineseCharMap
from .char_map_croatian import CroatianCharMap
from .char_map_cyrillic import CyrillicCharMap
from .char_map_devanagari import DevanagariCharMap
from .char_map_dutch import DutchCharMap
from .char_map_en_num_36 import EnglishNumber36CharMap
from .char_map_english import EnglishCharMap
from .char_map_french import FrenchCharMap
from .char_map_german import GermanCharMap
from .char_map_greek import GreekCharMap
from .char_map_gujarati import GujaratiCharMap
from .char_map_hangul import HangulCharMap
from .char_map_hebrew import HebrewCharMap
from .char_map_hungarian import HungarianCharMap
from .char_map_indonesian import IndonesianCharMap
from .char_map_italian import ItalianCharMap
from .char_map_japanese import JapaneseCharMap
from .char_map_javanese import JavaneseCharMap
from .char_map_kana import KanaCharMap
from .char_map_kannada import KannadaCharMap
from .char_map_khmer import KhmerCharMap
from .char_map_latin import LatinCharMap
from .char_map_malay import MalayCharMap
from .char_map_malayalam import MalayalamCharMap
from .char_map_marathi import MarathiCharMap
from .char_map_number import NumberCharMap
from .char_map_persian import PersianCharMap
from .char_map_polish import PolishCharMap
from .char_map_portuguese import PortugueseCharMap
from .char_map_punjabi import PunjabiCharMap
from .char_map_romanian import RomanianCharMap
from .char_map_russian import RussianCharMap
from .char_map_sinhala import SinhalaCharMap
from .char_map_spanish import SpanishCharMap
from .char_map_symbol import SymbolCharMap
from .char_map_tagalog import TagalogCharMap
from .char_map_tamil import TamilCharMap
from .char_map_telugu import TeluguCharMap
from .char_map_thai import ThaiCharMap
from .char_map_turkish import TurkishCharMap
from .char_map_unifiedapu import UnifiedAPUCharMap
from .char_map_unifiedapue import UnifiedAPUECharMap
from .char_map_unifiedbghmp import UnifiedBGHMPCharMap
from .char_map_unifiedbkt import UnifiedBKTCharMap
from .char_map_unifiedcg import UnifiedCGCharMap
from .char_map_unifiedcge import UnifiedCGECharMap
from .char_map_unifiedcj import UnifiedCJCharMap
from .char_map_unifiedcje import UnifiedCJECharMap
from .char_map_unifiedcyrillic import UnifiedCyrillicCharMap
from .char_map_unifieddevanagari import UnifiedDevanagariCharMap
from .char_map_unifiedke import UnifiedKECharMap
from .char_map_unifiedkt import UnifiedKTCharMap
from .char_map_unifiedlatin1 import UnifiedLatin1CharMap
from .char_map_unifiedmst import UnifiedMSTCharMap
from .char_map_urdu import UrduCharMap
from .char_map_vietnamese import VietnameseCharMap

LANG_CODE_TO_LANG_NAME = {
    "am": "Amharic",
    "any": "Any",
    "any1": "Any1",
    "any2": "Any2",
    "any3": "Any3",
    "any4": "Any4",
    "any5": "Any5",
    "any6": "Any6",
    "any7": "Any7",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bangla",
    "cyrillic": "Cyrillic",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "en_num_36": "English_Number_36",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "mixed": "Mixed",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "nl": "Dutch",
    "none": "None",
    "num": "Number",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "symbol": "Symbols",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "u_apu": "UnifiedAPU",
    "u_apue": "UnifiedAPUE",
    "u_bghmp": "UnifiedBGHMP",
    "u_bkt": "UnifiedBKT",
    "u_cg": "UnifiedCG",
    "u_cge": "UnifiedCGE",
    "u_cj": "UnifiedCJ",
    "u_cje": "UnifiedCJE",
    "u_cyrillic": "UnifiedCyrillic",
    "u_devanagari": "UnifiedDevanagari",
    "u_ke": "UnifiedKE",
    "u_kt": "UnifiedKT",
    "u_la1": "UnifiedLatin1",
    "u_mst": "UnifiedMST",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

LANG_NAME_TO_LANG_CODE = {
    "Amharic": "am",
    "Any": "any",
    "Any1": "any1",
    "Any2": "any2",
    "Any3": "any3",
    "Any4": "any4",
    "Any5": "any5",
    "Any6": "any6",
    "Any7": "any7",
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Bangla": "bn",
    "Cyrillic": "cyrillic",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "English_Number_36": "en_num_36",
    "Spanish": "es",
    "Persian": "fa",
    "French": "fr",
    "Gujarati": "gu",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Latin": "la",
    "Mixed": "mixed",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Malay": "ms",
    "Burmese": "my",
    "Dutch": "nl",
    "None": "none",
    "Number": "num",
    "Punjabi": "pa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sinhala": "si",
    "Symbols": "symbol",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Tagalog": "tl",
    "Turkish": "tr",
    "UnifiedAPU": "u_apu",
    "UnifiedAPUE": "u_apue",
    "UnifiedBGHMP": "u_bghmp",
    "UnifiedBKT": "u_bkt",
    "UnifiedCG": "u_cg",
    "UnifiedCGE": "u_cge",
    "UnifiedCJ": "u_cj",
    "UnifiedCJE": "u_cje",
    "UnifiedCyrillic": "u_cyrillic",
    "UnifiedDevanagari": "u_devanagari",
    "UnifiedKE": "u_ke",
    "UnifiedKT": "u_kt",
    "UnifiedLatin1": "u_la1",
    "UnifiedMST": "u_mst",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Chinese": "zh",
}

lang_code_to_char_map_class = {
    "am": AmharicCharMap,
    "any": AnyCharMap,
    "any1": Any1CharMap,
    "any2": Any2CharMap,
    "any3": Any3CharMap,
    "any4": Any4CharMap,
    "any5": Any5CharMap,
    "any6": Any6CharMap,
    "any7": Any7CharMap,
    "ar": ArabicCharMap,
    "bg": BulgarianCharMap,
    "bn": BengaliCharMap,
    "cyrillic": CyrillicCharMap,
    "de": GermanCharMap,
    "el": GreekCharMap,
    "en": EnglishCharMap,
    "en_num_36": EnglishNumber36CharMap,
    "es": SpanishCharMap,
    "fa": PersianCharMap,
    "fr": FrenchCharMap,
    "gu": GujaratiCharMap,
    "he": HebrewCharMap,
    "hi": DevanagariCharMap,
    "hr": CroatianCharMap,
    "hu": HungarianCharMap,
    "id": IndonesianCharMap,
    "it": ItalianCharMap,
    "ja": JapaneseCharMap,
    "jv": JavaneseCharMap,
    "kana": KanaCharMap,
    "km": KhmerCharMap,
    "kn": KannadaCharMap,
    "ko": HangulCharMap,
    "la": LatinCharMap,
    "ml": MalayalamCharMap,
    "mr": MarathiCharMap,
    "ms": MalayCharMap,
    "my": BurmeseCharMap,
    "nl": DutchCharMap,
    "num": NumberCharMap,
    "pa": PunjabiCharMap,
    "pl": PolishCharMap,
    "pt": PortugueseCharMap,
    "ro": RomanianCharMap,
    "ru": RussianCharMap,
    "si": SinhalaCharMap,
    "symbol": SymbolCharMap,
    "ta": TamilCharMap,
    "te": TeluguCharMap,
    "th": ThaiCharMap,
    "tl": TagalogCharMap,
    "tr": TurkishCharMap,
    "u_apu": UnifiedAPUCharMap,
    "u_apue": UnifiedAPUECharMap,
    "u_bkt": UnifiedBKTCharMap,
    "u_bghmp": UnifiedBGHMPCharMap,
    "u_cg": UnifiedCGCharMap,
    "u_cge": UnifiedCGECharMap,
    "u_cj": UnifiedCJCharMap,
    "u_cje": UnifiedCJECharMap,
    "u_cyrillic": UnifiedCyrillicCharMap,
    "u_devanagari": UnifiedDevanagariCharMap,
    "u_ke": UnifiedKECharMap,
    "u_kt": UnifiedKTCharMap,
    "u_la1": UnifiedLatin1CharMap,
    "u_mst": UnifiedMSTCharMap,
    "ur": UrduCharMap,
    "vi": VietnameseCharMap,
    "zh": ChineseCharMap,
}

LANGUAGE_COMBO = {
    "u_apu": ["ar", "fa", "ur"],
    "u_apue": ["ar", "fa", "ur"],
    "u_bghmp": ["bn", "gu", "hi", "mr", "pa"],
    "u_bkt": ["my", "km", "th"],
    "u_cg": ["bg", "el", "ru"],
    "u_cge": ["bg", "el", "ru"],
    "u_cj": ["ja", "zh"],
    "u_cje": ["ja", "zh"],
    "u_cyrillic": ["bg", "ru"],
    "u_devanagari": ["hi", "mr"],
    "u_ke": ["ko"],
    "u_kt": ["kn", "te"],
    "u_la1": [
        "de",
        "en",
        "es",
        "fr",
        "hr",
        "hu",
        "id",
        "it",
        "jv",
        "la",
        "ms",
        "nl",
        "pl",
        "pt",
        "ro",
        "tl",
        "tr",
        "vi",
    ],
    "u_mst": ["ml", "si", "ta"],
}


def code_to_name(lang_code):
    return LANG_CODE_TO_LANG_NAME[lang_code]


def name_to_code(lang_name):
    return LANG_NAME_TO_LANG_CODE[lang_name]


def get_language_config(cfg, language):
    language_config_dict = {
        "am": cfg.SEQUENCE.AMHARIC,
        "any": cfg.SEQUENCE.ANY,
        "any1": cfg.SEQUENCE.ANY1,
        "any2": cfg.SEQUENCE.ANY2,
        "any3": cfg.SEQUENCE.ANY3,
        "any4": cfg.SEQUENCE.ANY4,
        "any5": cfg.SEQUENCE.ANY5,
        "any6": cfg.SEQUENCE.ANY6,
        "any7": cfg.SEQUENCE.ANY7,
        "ar": cfg.SEQUENCE.ARABIC,
        "bg": cfg.SEQUENCE.BULGARIAN,
        "bn": cfg.SEQUENCE.BENGALI,
        "cyrillic": cfg.SEQUENCE.CYRILLIC,
        "de": cfg.SEQUENCE.GERMAN,
        "el": cfg.SEQUENCE.GREEK,
        "en": cfg.SEQUENCE.ENGLISH,
        "en_num": cfg.SEQUENCE.EN_NUM,
        "en_num_36": cfg.SEQUENCE.EN_NUM_36,
        "es": cfg.SEQUENCE.SPANISH,
        "fa": cfg.SEQUENCE.PERSIAN,
        "fr": cfg.SEQUENCE.FRENCH,
        "gu": cfg.SEQUENCE.GUJARATI,
        "he": cfg.SEQUENCE.HEBREW,
        "hi": cfg.SEQUENCE.DEVANAGARI,
        "hr": cfg.SEQUENCE.CROATIAN,
        "hu": cfg.SEQUENCE.HUNGARIAN,
        "id": cfg.SEQUENCE.INDONESIAN,
        "it": cfg.SEQUENCE.ITALIAN,
        "ja": cfg.SEQUENCE.JAPANESE,
        "jv": cfg.SEQUENCE.JAVANESE,
        "km": cfg.SEQUENCE.KHMER,
        "kn": cfg.SEQUENCE.KANNADA,
        "ko": cfg.SEQUENCE.HANGUL,
        "la": cfg.SEQUENCE.LATIN,
        "ml": cfg.SEQUENCE.MALAYALAM,
        "mr": cfg.SEQUENCE.MARATHI,
        "ms": cfg.SEQUENCE.MALAY,
        "my": cfg.SEQUENCE.BURMESE,
        "nl": cfg.SEQUENCE.DUTCH,
        "num": cfg.SEQUENCE.NUMBER,
        "pa": cfg.SEQUENCE.PUNJABI,
        "pl": cfg.SEQUENCE.POLISH,
        "pt": cfg.SEQUENCE.PORTUGUESE,
        "ro": cfg.SEQUENCE.ROMANIAN,
        "ru": cfg.SEQUENCE.RUSSIAN,
        "si": cfg.SEQUENCE.SINHALA,
        "symbol": cfg.SEQUENCE.SYMBOL,
        "ta": cfg.SEQUENCE.TAMIL,
        "te": cfg.SEQUENCE.TELUGU,
        "th": cfg.SEQUENCE.THAI,
        "tl": cfg.SEQUENCE.TAGALOG,
        "tr": cfg.SEQUENCE.TURKISH,
        "u_apu": cfg.SEQUENCE.UNIFIEDAPU,
        "u_apue": cfg.SEQUENCE.UNIFIEDAPUE,
        "u_bghmp": cfg.SEQUENCE.UNIFIEDBGHMP,
        "u_bkt": cfg.SEQUENCE.UNIFIEDBKT,
        "u_cg": cfg.SEQUENCE.UNIFIEDCG,
        "u_cge": cfg.SEQUENCE.UNIFIEDCGE,
        "u_cj": cfg.SEQUENCE.UNIFIEDCJ,
        "u_cje": cfg.SEQUENCE.UNIFIEDCJE,
        "u_cyrillic": cfg.SEQUENCE.UNIFIEDCYRILLIC,
        "u_devanagari": cfg.SEQUENCE.UNIFIEDDEVANAGARI,
        "u_ke": cfg.SEQUENCE.UNIFIEDKE,
        "u_kt": cfg.SEQUENCE.UNIFIEDKT,
        "u_la1": cfg.SEQUENCE.UNIFIEDLATIN1,
        "u_mst": cfg.SEQUENCE.UNIFIEDMST,
        "ur": cfg.SEQUENCE.URDU,
        "vi": cfg.SEQUENCE.VIETNAMESE,
        "zh": cfg.SEQUENCE.CHINESE,
    }
    if language in language_config_dict:
        return language_config_dict[language]
    else:
        raise Exception("Unsupported language: {}".format(language))


def pick_best_encoding_for_word(dataset_name, language, word):
    # First, try ftfy fix (which fixes most encoding issues)
    new_word = fix_encoding(word)

    if word == new_word:
        if language is None:
            # Trust fix_encoding when language is None
            return {"word": word}
        else:
            # In case ftfy doesn't fix the problem, e.g.,
            # https://github.com/LuminosoInsight/python-ftfy/issues/158
            # https://github.com/LuminosoInsight/python-ftfy/issues/159

            char_map_class = lang_code_to_char_map_class[language]
            new_word = char_map_class.get_utf8_word(word)
            if word == new_word:
                return {"word": word}
            else:
                fix = f"language_{language}"
    else:
        if language is None:
            # Trust fix_encoding when language is None
            return {"word": new_word, "fix": "ftfy"}
        else:
            # In some cases we need to fix again after ftfy's fix
            # Chinese example:
            #   fix_encoding("Ã©Â»â€˜") == "é»‘"
            #   get_utf8_word("é»‘") == "黑"
            # Korean example:
            #   fix_encoding("Ã£â€¦â€¹") == "ã…‹"
            #   get_utf8_word("ã…‹") == "ㅋ"
            # Hebrew example:
            #   fix_encoding("Ã—Å¾Ã—Â¨Ã—â€ºÃ—â€“") == "×ž×¨×›×–"
            #   get_utf8_word("×ž×¨×›×–") == "מרכז"
            char_map_class = lang_code_to_char_map_class[language]
            new_word_2 = char_map_class.get_utf8_word(new_word)
            if new_word == new_word_2:
                fix = "ftfy"
            else:
                new_word = new_word_2
                fix = f"ftfy+language_{language}"

    # Use language information to determine whether the fix is appropriate
    char_map_class = lang_code_to_char_map_class[language]

    # Caution:
    # we need to count unsupported chars, instead of supported chars,
    # as len(word) could be different from len(new_word).
    num_unsupported_chars_old = 0
    for ch in word:
        if not char_map_class.contain_char_exclusive(ch):
            num_unsupported_chars_old += 1
    num_unsupported_chars_new = 0
    for ch in new_word:
        if not char_map_class.contain_char_exclusive(ch):
            num_unsupported_chars_new += 1

    if num_unsupported_chars_new < num_unsupported_chars_old:
        return {"word": new_word, "fix": fix}
    else:
        return {"word": word}


latin_correction_map = {
    "A": ["A", chr(913)],
    "a": ["a", chr(1072)],
    "B": ["B", chr(1074)],
    "c": ["c", chr(1089)],
    "E": ["E", chr(917)],
    "e": ["e", chr(1077)],
    "H": ["H", chr(905), chr(919), chr(1085)],
    "I": ["I", chr(921)],
    "K": ["K", chr(922), chr(1082)],
    "M": ["M", chr(924), chr(1084)],
    "N": ["N", chr(925)],
    "n": ["n", chr(951), chr(1087)],
    "O": ["O", chr(927)],
    "o": ["o", chr(959), chr(1086)],
    "P": ["P", chr(929)],
    "p": ["p", chr(1088)],
    "r": ["r", chr(1075)],
    "T": ["T", chr(932), chr(1090)],
    "t": ["t", chr(964)],
    "u": ["u", chr(1080)],
    "x": ["x", chr(1093)],
    "Y": ["Y", chr(933)],
    "y": ["y", chr(947)],
}


def word_case_score(word):
    upper_count = 0
    lower_count = 0
    for ch in word:
        if "A" <= ch <= "Z":
            if lower_count > 0:
                # upper comes after lower
                return 0
            upper_count += 1
        elif "a" <= ch <= "z":
            if upper_count > 1:
                # lower comes after more than 1 upper
                return 0
            lower_count += 1
        else:
            # punctuation resets the count
            upper_count = 0
            lower_count = 0
    return 1


def cyrillic_greek_to_latin(word):
    new_word_upper = ""
    new_word_lower = ""
    new_word_original = ""
    for ch in word:
        correctable = False
        if ch in "~!@#$%&*()_+`-={{}}|[]\\:\";'<>?,./":
            new_word_upper += ch
            new_word_lower += ch
            new_word_original += ch
            correctable = True
        else:
            for latin_char, cg_list in latin_correction_map.items():
                if ch in cg_list:
                    if ch in [chr(1072), chr(1077), chr(1086), chr(1089), chr(1093)]:
                        # due to force_lowercase flag in Cyrillic data downloading,
                        # cyrillic a/e/o/c/x could represent either upper or lower cases
                        new_word_upper += latin_char.upper()
                        new_word_lower += latin_char.lower()
                    else:
                        new_word_upper += latin_char
                        new_word_lower += latin_char
                    new_word_original += latin_char
                    correctable = True
                    break

        if not correctable:
            return word

    # decide which one of new_word_original, new_word_upper and new_word_lower makes more sense
    if word_case_score(new_word_original) == 1:
        return new_word_original
    elif word_case_score(new_word_upper) == 1:
        return new_word_upper
    elif word_case_score(new_word_lower) == 1:
        return new_word_lower

    return new_word_original
