import glob
from typing import List, Union, Iterable
from pathlib import Path
from spring_amr.penman import load as pm_load

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
            unkown_lang = set()
            path = Path(path)
            #en_XX, de_DE, es_XX, it_IT, zh_CN
            if str(path).endswith("_de.txt"):
                lang_code = "de_DE"
            elif str(path).endswith("_es.txt"):
                lang_code = "es_XX"
            elif str(path).endswith("_it.txt"):
                lang_code = "it_IT"
            elif str(path).endswith("_zh.txt"):
                lang_code = "zh_CN"
            else:
                lang_code = "en_XX"
            pm_graphs = pm_load(path, dereify=dereify, remove_wiki=remove_wiki)
            for g in pm_graphs:
                if ("snt_lang" not in g.metadata) or (g.metadata["snt_lang"] not in ['zh', 'it', 'es', 'de', 'en']):
                    unkown_lang.add(g.metadata["snt_lang"] if "snt_lang" in g.metadata else 'empty')
                    g.metadata["snt_lang"] = lang_code
                if g.metadata["snt_lang"] == 'zh':
                    g.metadata["snt_lang"] = 'zh_CN'
                elif g.metadata["snt_lang"] == 'it':
                    g.metadata["snt_lang"] = 'it_IT'
                elif g.metadata["snt_lang"] == 'es':
                    g.metadata["snt_lang"] = 'es_XX'
                elif g.metadata["snt_lang"] == 'de':
                    g.metadata["snt_lang"] = 'de_DE'
                elif g.metadata["snt_lang"] == 'en':
                    g.metadata["snt_lang"] = 'en_XX'

                if lang_code == "zh_CN":
                    g.metadata["snt"] = ''.join(g.metadata["snt"].split())
            graphs.extend(pm_graphs)
            if len(unkown_lang) > 0:
                print (f"unkown input language {unkown_lang} in {path}, treat it as {lang_code}")

    #assert graphs
    
    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata['snt_orig'] = metadata['snt']
            tokens = eval(metadata['tokens'])
            metadata['snt'] = ' '.join([t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])

    return graphs
