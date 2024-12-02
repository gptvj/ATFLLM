# Step by Step

## Step 1: Sinh ra nội dung hợp đồng sử dụng file: 
GeminiContractGeneration.ipynb

=> Sau bước này đầu ra sẽ là file: LawContractAPI_demo.json

## Step 2: Xử lý format cho dữ liệu ở Step 1
1. xulyLawContract.py
=> Sau bước này đầu ra sẽ là file LawContractAPI_demo_format.json
1. checksameid.py: kiểm tra lỗi trùng dữ liệu
2. xulysame.py: xử lý loại bỏ trùng lặp, hoặc gộp lại relevant articls
=> Sau bước này đầu ra sẽ là file LawContract_format_after.json
3. check_data.py: kiểm tra một số lỗi khác, nếu không lỗi thì data hợp lệ

## Step 3: Sinh chương, điều cho nội dung đã được xử lý ở trên
GeminiAddCaption.ipynb 
=> Sau bước này đầu ra sẽ là file LawContractAPI_AddCaption.json
Lưu ý dữ liệu chương điều có thế thiếu do luật sinh ra có thể là những câu độc lập và không thuộc phần nào trong mục lục chúng tôi đưa ra.

## Step 4: Tạo đúng format của dữ liệu gồm question, question_short, và question_full
question_full.py
