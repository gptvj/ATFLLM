# đọc file json
import json
import os

# đọc file json
def readJsonFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# ghi file json
def writeJsonFile(filePath, data):
    with open(filePath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    filePath = r'../../create_root_data/Law2Contract/Data/LawContractAPI_demo.json'
    data = readJsonFile(filePath)
    new_data = []
    for item in data:
        questions = item['question'].split('\n')[:5]
        # try:
        #     print(questions[0].split('. ')[1])
        # except:
        #     print('Error:', questions[0])
        #     break
        for question in questions:
            question = question.split('. ')
            if len(question) > 1:
                question = question[1]
            else:
                question = question[0]
            # print(question)
            if question.strip()!="":
                new_item = {
                    'question': question.strip(),
                    'relevant_articles': item['relevant_articles']
                }
                new_data.append(new_item)

    if not os.path.exists(r'../../create_root_data/Law2Contract/Data'):
        os.makedirs(r'../../create_root_data/Law2Contract/Data')

    # writeJsonFile(r'.\New_LawContract\Law_Contract_2208\New_LawContract_Format.json', new_data)
    file_save = r'../../create_root_data/Law2Contract/Data/LawContractAPI_demo_format.json'
    if not os.path.exists(file_save):
        writeJsonFile(file_save, new_data)
        print(f"Viết vào file {file_save} thành công")
    else:
        print("file đã tồn tại!")        
