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
    file_path = r'.\Law_Contract_2208\New_LawContract_Format.json'
    # file_path = './LawContractAPI_full_after.json'
    data = readJsonFile(file_path)
    
    # # delete items with empty question
    # new_data = []
    # for item in data:
    #     if item['question'].strip() != '':
    #         item['muc'] = item['muc'].strip()
    #         item['dieu'] = item['dieu'].strip()
    #         item['question'] = item['question'].strip()
    #         new_data.append(item)
    # print('Number of items after deletion:', len(new_data))
    # writeJsonFile('./LawContractAPI_full.json', new_data)

    # Count the number of questions that appear multiple times
    question_count = {}
    for item in data:
        if item['question'].strip() in question_count:
            question_count[item['question'].strip()] += 1
        else:
            question_count[item['question'].strip()] = 1
    
    # # Sort by frequency of occurrence
    # question_count = dict(sorted(question_count.items(), key=lambda item: item[1], reverse=True))
    
    # Count the number of questions that appear more than 2 times
    question_filter = {k: v for k, v in question_count.items() if v >= 2}
    print('Number of questions that appear more than 2 times:', len(question_filter))

    for k, v in question_filter.items():
        print(k, v)
