import string
import os
import json
# import MeCab  # or KyTea for Japanese tokenization
from fugashi import Tagger

# stop_word_japanese = ["です", "これ", "それ"]  # Add more stopwords as needed
number_japanese = ["１", "２", "３", "４", "５", "６", "７", "８", "９", "１０"]  # Japanese numbers
chars_japanese = ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ"]  # Hiragana characters

stop_word_japanese = ["あそこ","あっ","あの","あのかた","あの人","あり","あります","ある","あれ","い","いう","います","いる","う","うち","え","お","および","おり","おります","か","かつて","から","が","き","ここ","こちら","こと","この","これ","これら","さ","さらに","し","しかし","する","ず","せ","せる","そこ","そして","その","その他","その後","それ","それぞれ","それで","た","ただし","たち","ため","たり","だ","だっ","だれ","つ","て","で","でき","できる","です","では","でも","と","という","といった","とき","ところ","として","とともに","とも","と共に","どこ","どの","な","ない","なお","なかっ","ながら","なく","なっ","など","なに","なら","なり","なる","なん","に","において","における","について","にて","によって","により","による","に対して","に対する","に関する","の","ので","のみ","は","ば","へ","ほか","ほとんど","ほど","ます","また","または","まで","も","もの","ものの","や","よう","より","ら","られ","られる","れ","れる","を","ん","何","及び","彼","彼女","我々","特に","私","私達","貴方","貴方方", "１", "２", "３", "４", "５", "６", "７", "８", "９", "１０", '〔', "〕", '（', '）', '：', ', ', '「', '」']
def remove_stopword_japanese(w):
    return w not in stop_word_japanese

def remove_punctuation_japanese(w):
    return w not in string.punctuation and w != "、" and w != "。" # Japanese punctuation

def lower_case_japanese(w):
    return w.lower()

def bm25_tokenizer(text):
    tagger = Tagger('-Owakati')
    tokens = tagger.parse(text).split()
    tokens = list(map(lower_case_japanese, tokens))
    tokens = list(filter(remove_punctuation_japanese, tokens))
    tokens = list(filter(remove_stopword_japanese, tokens))
    return tokens

def remove_stopword(w):
    return w not in stop_word_japanese
def remove_punctuation(w):
    return w not in string.punctuation
def lower_case(w):
    return w.lower()

def calculate_f2(precision, recall):        
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)

def load_json(path):
    return json.load(open(path))