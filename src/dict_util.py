import csv
import pickle
from random import randint

### NOTICE: csv only accept two colomn input. but accept multi-time input.


# 1_2_3, 1 is action, 2 is supply object, 3 is source object
def update_dict_csv(term_dict:dict, f):
    for rows in csv.reader(f):
        word = rows[0].lower()
        if word in term_dict:
            if rows[1] not in term_dict[word]:
                term_dict[word] = term_dict[word]+[rows[1]]
            else:
                print("{},{} 已存在".format(word, rows[1]))
        else:
            term_dict[word]=[rows[1]]
    term_dict = sort_dict(term_dict)
    pass

def export_csv_dict(term_dict:dict, f):
    for key, val in term_dict.items():
        csv.writer(f).writerow([key, val])
    pass

def save_pickle_dict(term_dict:dict, f):
    pickle.dump(term_dict, f, pickle.HIGHEST_PROTOCOL)
    pass

def update_pickel_csv(pickle_f, csv_f):
    term_dict = pickle.load(pickle_f)
    update_dict_csv(term_dict, csv_f)
    #save to pickle file, highest protocol to get better performance
    pickle.dump(term_dict, pickle_f, pickle.HIGHEST_PROTOCOL)
    pass

def sort_dict(term_dict:dict):
    term_dict = dict(sorted(term_dict.items(), key=lambda x:len(x[0]), reverse=True))
    return term_dict

def get_word(term_dict:dict, key:str) -> str:
    word = term_dict[key][randint(0,len(term_dict[key])-1)]
    return word

# #demo
# if __name__ == "__main__":
#     term_dict_sc2 = {}
#     with open("./finetune_data/dict_enzh.csv", 'r', encoding='utf-8') as f:
#         update_dict_csv(term_dict_sc2,f)
    
#     with open("../test.csv", "w", encoding='utf-8') as w:
#         export_csv_dict(term_dict_sc2,w)

## for load pickle, just:
# pickle.load(f)


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

