import csv
from random import randint

### NOTICE: csv only accept two colomn input. but accept multi-time input.


def form_dict(src_dict:list, tgt_dict:list) -> dict:
    final_dict = {}
    for idx, value in enumerate(src_dict):
        for item in value:
            if item:
                final_dict.update({item:list(filter(None, tgt_dict[idx]))})
    return final_dict


class term_dict(dict):
    """
    Dictionary object for force term replacement and it also act as term list for spelling-check. Compatiable with single-to-single, single-to-multi, multi-to-single, multi-to-multi mapping.

    Subclass of python's dict object.

    Following methods is rewrited to adapt the data structure:

        get():get the mapped word with average possibility.
    """
    def __init__(self, path, src_lang, tgt_lang) -> None:
        """
        term_dict object constructor.

        Take two csv file and their common path to get the word list.
        Words with mapping relationship should be placed in the same row.
        """
        with open(f"{path}/{src_lang}.csv", 'r', encoding="utf-8") as file:
            src_dict = list(csv.reader(file, delimiter=","))
        with open(f"{path}/{tgt_lang}.csv", 'r', encoding="utf-8") as file:
            tgt_dict = list(csv.reader(file, delimiter="," ))
        super().__init__(form_dict(src_dict, tgt_dict))


    def get(self, key:str) -> str:
        """
        get the mapped word with average possibility.

        return one mapped word
        """
        word = self[key][randint(0,len(self[key])-1)]
        return word
