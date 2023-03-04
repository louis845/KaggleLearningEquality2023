# uses lxml (https://github.com/lxml/lxml/blob/master/LICENSE.txt) for web scraping.

"""import lxml.html
import requests

mdls = ["af-en", "am-en", "ar-en", "as-en", "ase-en", "az-en", "bcl-en", "bem-en", "ber-en", "bg-en", "bi-en", "bn-en", "br-en", "bs-en", "bzs-en", "ca-en", "ceb-en", "chk-en", "crs-en", "cs-en", "cy-en", "da-en", "de-en", "ee-en", "efi-en", "el-en", "eo-en", "es-en", "et-en", "eu-en", "fi-en", "fj-en", "fr-en", "ga-en", "gaa-en", "gil-en", "gl-en", "guw-en", "gv-en", "ha-en", "he-en", "hi-en", "hil-en", "ho-en", "hr-en", "ht-en", "hu-en", "hy-en", "id-en", "ig-en", "ilo-en", "is-en", "iso-en", "it-en", "ja-en", "jap-en", "ka-en", "kab-en", "kg-en", "kj-en", "kl-en", "ko-en", "kqn-en", "kwn-en", "kwy-en", "lg-en", "ln-en", "loz-en", "lt-en", "lu-en", "lua-en", "lue-en", "lun-en", "luo-en", "lus-en", "lv-en", "mfe-en", "mg-en", "mh-en", "mk-en", "ml-en", "mos-en", "mr-en", "ms-en", "mt-en", "nb-en", "ne-en", "ng-en", "niu-en", "nl-en", "nn-en", "nso-en", "ny-en", "nyk-en", "om-en", "pa-en", "pag-en", "pap-en", "pis-en", "pl-en", "pon-en", "pt-en", "rnd-en", "ro-en", "ru-en", "run-en", "rw-en", "sg-en", "si-en", "sk-en", "sm-en", "sn-en", "so-en", "sq-en", "sr-en", "srn-en", "ss-en", "st-en", "sv-en"]

def xpath_check(html_page, url):
    for elem in html_page.xpath("//tr/td[contains(text(), 'Tatoeba')]/.."):
        for child in elem.iterchildren():
            try:
                if ( float(child.text_content()) > 40.0):
                    print(url)
                    return True
            except ValueError:
                pass
    return False

lst = []
for model in mdls:
    url = "https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/" + model
    page = requests.get(url)
    html_page = lxml.html.fromstring(page.content)

    if xpath_check(html_page, url):
        lst.append(model)

print(lst)"""

import translate_helpers

nlangs = ['af-en', 'ar-en', 'as-en', 'bg-en', 'bn-en', 'br-en', 'ca-en', 'cs-en', 'cy-en', 'da-en', 'de-en', 'el-en', 'eo-en', 'es-en', 'et-en', 'eu-en', 'fi-en', 'fj-en', 'fr-en', 'ga-en', 'gl-en', 'he-en', 'hi-en', 'hr-en', 'ht-en', 'hu-en', 'id-en', 'ig-en', 'is-en', 'it-en', 'ja-en', 'lt-en', 'lv-en', 'mg-en', 'mk-en', 'ml-en', 'mr-en', 'ms-en', 'mt-en', 'nb-en', 'ne-en', 'nl-en', 'nn-en', 'ny-en', 'pap-en', 'pl-en', 'pt-en', 'ro-en', 'ru-en', 'rw-en', 'si-en', 'sq-en', 'sr-en', 'sv-en']
nlangs = [x[:-3] for x in nlangs]

new_langs = list(set(nlangs).difference(set(translate_helpers.list_available_langs())))
new_langs.sort()

print(new_langs)

for lang in new_langs:
    print("https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/" + lang + "-en")