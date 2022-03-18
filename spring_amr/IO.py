import glob
from typing import List, Union, Iterable
from pathlib import Path
from spring_amr.penman import load as pm_load

LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]

LANGUAGE_CODES_MAP={}
for x in LANGUAGE_CODES:
    LANGUAGE_CODES_MAP[x[:2]] = x
    LANGUAGE_CODES_MAP[x] = x

def read_raw_amr_data(
        paths: List[Union[str, Path]],
        use_recategorization=False,
        dereify=True,
        remove_wiki=False,
):
    #assert paths

    if not isinstance(paths, Iterable):
        paths = [paths]

    graphs = []
    for path_ in paths:
        for path in glob.glob(str(path_)):
            unkown_langs = set()
            path = Path(path)

            path_str = str(path)
            if len(path_str)>=8 and path_str.endswith(".txt") and path_str[-7]=='_':
                lang_code = path_str[-6:-4] 
            else:
                lang_code = "en"

            pm_graphs = pm_load(path, dereify=dereify, remove_wiki=remove_wiki)
            for g in pm_graphs:
                if ("snt_lang" not in g.metadata) or (g.metadata["snt_lang"] not in LANGUAGE_CODES_MAP):
                    unkown_langs.add(g.metadata["snt_lang"] if "snt_lang" in g.metadata else 'empty')
                    g.metadata["snt_lang"] = lang_code
                g.metadata["snt_lang"] = LANGUAGE_CODES_MAP[g.metadata["snt_lang"]]

                if lang_code == "zh_CN":
                    g.metadata["snt"] = ''.join(g.metadata["snt"].split())

            graphs.extend(pm_graphs)
            if len(unkown_langs) > 0:
                print (f"unkown input language {unkown_langs} in {path}, treat it as {lang_code}")

    #assert graphs
    
    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata['snt_orig'] = metadata['snt']
            tokens = eval(metadata['tokens'])
            metadata['snt'] = ' '.join([t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])

    return graphs
