import json

with open('/content/find_law_gemini_contract.json', 'r', encoding='utf-8') as f:
    list_result = json.load(f)

new_list_result = []
for result in list_result:
    if 'con' in result['question']:
        question_short = result['question']['con']
        question = result['question']['dieu'] + '\n' + result['question']['con']
    else:
        question_short = question = result['question']['dieu']
    question_full = result['question']['text']
    relevant_articles = []
    for law in result['relevant_laws']:
        relevant_articles.append(
            {
                'law_id': law['law_id'],
                'article_id': law['article_id']
            }
        )
    new_list_result.append(
        {
            'question_short': question_short,
            'question': question,
            'question_full': question_full,
            'relevant_articles': relevant_articles
        }
    )

with open('/content/find_law_gemini_contract_formatted.json', 'w', encoding='utf-8') as f:
    json.dump(new_list_result, f, ensure_ascii=False, indent=4)