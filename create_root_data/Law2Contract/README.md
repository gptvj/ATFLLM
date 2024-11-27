# Step by Step

## Step 1: Sinh ra nội dung hợp đồng sử dụng file: 
GeminiContractGeneration.ipynb

## Step 2: Xử lý format cho dữ liệu ở Step 1
xulyLawContract.py

## Step 3: Sinh chương, điều cho nội dung đã được xử lý ở trên
GeminiAddCaption.ipynb

## Step 4: Xử lý format cho dữ liệu ở Step 3
1. checksameid.py: kiểm tra lỗi trùng dữ liệu
2. xulysame.py: xử lý loại bỏ trùng lặp, hoặc gộp lại relevant articls
3. check_data.py: kiểm tra một số lỗi khác, nếu không lỗi thì data hợp lệ