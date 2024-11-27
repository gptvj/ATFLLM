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
    # file_path = './LawContractAPI_full.json'
    file_path = './Law_Contract_2208/New_LawContract_Format.json'
    data = readJsonFile(file_path)

    # kiểm tra bị trùng lặp dữ liệu
    question_count = {}
    for item in data:
        if str(item['question']+item['relevant_articles'][0]['law_id']+item['relevant_articles'][0]['article_id']) in question_count:
            question_count[str(item['question']+item['relevant_articles'][0]['law_id']+item['relevant_articles'][0]['article_id'])] += 1
        else:
            question_count[str(item['question']+item['relevant_articles'][0]['law_id']+item['relevant_articles'][0]['article_id'])] = 1

    count = 0
    for key, value in question_count.items():
        if value > 1:
            print(key, value)
            count += 1
    print(count)