# read file json 
import json
import os

def readJsonFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# write file json
def writeJsonFile(filePath, data):
    with open(filePath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    ## merge data from multiple files
    # forder_path = r'.'
    # full_data = []
    # for file_name in os.listdir(forder_path):
    #     if file_name.endswith('.json'):
    #         file_path = os.path.join(forder_path, file_name)
    #         data = readJsonFile(file_path)
    #         full_data.extend(data)
    # print(len(full_data))
    # writeJsonFile(r'.\LawContractAPI_full.json', full_data)

    ## process data

    filePath = r'.\LawContractAPI_full_after.json'
    data = readJsonFile(filePath)
    new_data = []
    for item in data: 
        cons = []
        if 'muc' in item and item['muc'].strip() != "":
            cons.append(item['muc'].strip())
        if 'dieu' in item and item['dieu'].strip() != "":
            cons.append(item['dieu'].strip())
        cons.append(item['question'].strip())
        new_item = {
            'question_short': item['question'].strip(),
            'question': item['question'].strip(),
            'question_full': '\n'.join(cons),
            'relevant_articles': item['relevant_articles']
        }
        new_data.append(new_item)
    writeJsonFile(r'.\LawContractAPI_full_1107.json', new_data)