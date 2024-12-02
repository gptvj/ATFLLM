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
    file_path = r'../../create_root_data/Law2Contract/Data/LawContractAPI_demo_format.json'
    # file_path = r'.\Law_Contract_2208\New_LawContract_Format.json'
    
    data = readJsonFile(file_path)
    print(len(data))
    
    # gom nhóm những item có cùng "question"
    group_data = {}
    for item in data:
        question = item['question']
        if question not in group_data:
            group_data[question] = []
        group_data[question].append(item)
    
    new_data = []
    for k, v in group_data.items():
        if len(v) > 1:
            print(k, len(v))
            if len(v) > 5: 
                print(k, len(v))
                continue
            relevant_articles = []
            for item in v:
                same = False
                for pos in relevant_articles: 
                    if pos['law_id'] == item["relevant_articles"][0]['law_id'] and pos['article_id'] == item["relevant_articles"][0]['article_id']:
                        same = True
                        break
                if same == False:
                    relevant_articles.extend(item["relevant_articles"])
            
            new_item = {
                # "muc": v[0]["muc"],
                # "dieu": v[0]["dieu"],
                "question": v[0]["question"],
                "relevant_articles": relevant_articles,
            }
            new_data.append(new_item)
        else:
            new_data.append(v[0])

    print(len(new_data))
    file_save = r'../../create_root_data/Law2Contract/Data/LawContract_format_after.json'
    if not os.path.exists(os.path.dirname(file_save)): 
        os.makedirs(os.path.dirname(file_save))
    if not os.path.exists(file_save):
        writeJsonFile(file_save, new_data)
        print("Da xu ly trung xong!")
    else: 
        print("File đã tồn tại!")
            