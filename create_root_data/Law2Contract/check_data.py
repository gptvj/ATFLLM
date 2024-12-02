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
    # file_path = r'.\Law_Contract_2208\New_LawContract_Format.json'
    file_path = r'../../create_root_data/Law2Contract/Data/LawContract_format_after.json'
    # file_path = './LawContractAPI_full_after.json'
    data = readJsonFile(file_path)
    
    # # xoá các phẩn tử có question rỗng
    # new_data = []
    # for item in data:
    #     if item['question'].strip() != '':
    #         item['muc'] = item['muc'].strip()
    #         item['dieu'] = item['dieu'].strip()
    #         item['question'] = item['question'].strip()
    #         new_data.append(item)
    # print('Số lượng phần tử sau khi xoá:', len(new_data))
    # writeJsonFile('./LawContractAPI_full.json', new_data)

    # số lượng question xuất hiện nhiều lần
    question_count = {}
    for item in data:
        if item['question'].strip() in question_count:
            question_count[item['question'].strip()] += 1
        else:
            question_count[item['question'].strip()] = 1
    
    # # sắp xếp theo số lần xuất hiện
    # question_count = dict(sorted(question_count.items(), key=lambda item: item[1], reverse=True))
    
    # số lượng question xuất hiện nhiều hơn 2 lần
    question_filter = {k: v for k, v in question_count.items() if v >= 2}
    print('Số lượng question xuất hiện nhiều hơn 2 lần:', len(question_filter))

    for k, v in question_filter.items():
        print(k, v)
