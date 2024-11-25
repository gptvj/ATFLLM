import pickle

# # Đường dẫn tới file pickle corpus
# file_path = './beir_embedding_scifact/corpus_scifact.pkl'

# # Mở file pickle và đọc nội dung
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # In ra dữ liệu đã đọc
# print(len(data))
# print(data[0])
# print(len(data[0]))
# print(data[0][0])
# print(len(data[0][0]))

# print(len(data))
# print(data[1])
# print(len(data[1]))
# print(data[1][0])
# print(len(data[1][0]))

# Đường dẫn tới file pickle query
file_path = 'japanese_embedding_negative_round2/queries_embedding.pkl'

# Mở file pickle và đọc nội dung
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# In ra dữ liệu đã đọc
# print(len(data))
# print(data[0])
# print(len(data[0]))
# print(data[0][0])
# print(len(data[0][0]))

# print(len(data))
# print(data[1])
# print(len(data[1]))
# print(data[1][0])
# print(len(data[1][0]))

# check nan
import math
count = 0
for i in range(len(data[0])):
    if math.isnan(data[0][i][0]): 
        print(i)
        print(data[0][i])
        count += 1



print(count)






# import pickle

# file_path = '/home/trung/experiment_50_hardnegative_jp_llama2/pair_data/bm_25_pairs_training_top50'

# # Mở file pickle và đọc nội dung
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # 55053

# print(data[:2])



# import json 


# file_path = 'generated_data/legal_dict.json'

# # Mở file pickle và đọc nội dung
# with open(file_path, 'rb') as file:
#     data = json.load(file)

# # 55053

# print(data)

