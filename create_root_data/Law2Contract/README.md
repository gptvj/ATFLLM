# Step by Step

## Step 1: Generate contract content using:
GeminiContractGeneration.ipynb

## Step 2: Format the output data from Step 1
xulyLawContract.py

## Step 3: Generate chapters and clauses for the processed content
GeminiAddCaption.ipynb

## Step 4: Format the output data from Step 3
1. checksameid.py: check for duplicate data IDs
2. xulysame.py: remove duplicates or merge relevant articles
3. check_data.py: verify additional issues; if no errors, data is valid

## Note:
- We use the terms "muc", "dieu", and "khoan" to refer to hierarchical levels in legal documents. These correspond approximately to "chapter", "article", and "clause" in English.