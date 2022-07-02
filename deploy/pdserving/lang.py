LANG_DICT = {
    "ch": "../../ppocr/utils/ppocr_keys_v1.txt",
    "en": "../../ppocr/utils/en_dict.txt",
    "korean": "../../ppocr/utils/dict/korean_dict.txt",
    "japan": "../../ppocr/utils/dict/japan_dict.txt",
    "chinese_cht": "../../ppocr/utils/dict/chinese_cht_dict.txt",
    "ta": "../../ppocr/utils/dict/ta_dict.txt",
    "te": "../../ppocr/utils/dict/te_dict.txt",
    "ka": "../../ppocr/utils/dict/ka_dict.txt",
    "latin": "../../ppocr/utils/dict/latin_dict.txt",
    "arabic": "../../ppocr/utils/dict/arabic_dict.txt",
    "cyrillic": "../../ppocr/utils/dict/cyrillic_dict.txt",
    "devanagari": "../../ppocr/utils/dict/devanagari_dict.txt"
}

def get_char_dict_for_lang(lang):
    latin_lang = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
    ]
    arabic_lang = ["ar", "fa", "ug", "ur"]
    cyrillic_lang = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
    ]
    devanagari_lang = [
        "hi",
        "mr",
        "ne",
        "bh",
        "mai",
        "ang",
        "bho",
        "mah",
        "sck",
        "new",
        "gom",
        "sa",
        "bgc",
    ]

    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert lang in LANG_DICT, f"param lang must in {LANG_DICT.keys()} but got {lang}"

    return LANG_DICT[lang]
