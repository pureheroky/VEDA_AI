import re
import openai
from utils import (
    load_documents,
    split_text_into_blocks,
    initialize_local_vectorstore_from_two_source,
    rank_documents_by_relevance,
    merge_similar_docs)
from main_v2 import calculate_cost, count_tokens
def process_query(user_query: str):
    # Загрузка и обработка документов
    print("step1")
    docs = load_documents("./tgbot_data/db_veda.docx")
    all_splits = split_text_into_blocks(docs)
    print("step2")
    if len(user_query.split()) <= 3:
        print("used short")
        vectorstore = initialize_local_vectorstore_from_two_source(all_splits, "./tgbot_data/chroma_short")
        docs = vectorstore.similarity_search(user_query, k=30)
        ranked_docs = rank_documents_by_relevance(docs, user_query, top_n=30)
        merged_docs = merge_similar_docs(ranked_docs, top_n=30)
    else:
        print("used long")
        vectorstore = initialize_local_vectorstore_from_two_source(all_splits, "./tgbot_data/chroma")
        docs = vectorstore.similarity_search(user_query, k=5)
        ranked_docs = rank_documents_by_relevance(docs, user_query, top_n=5)
        merged_docs = merge_similar_docs(ranked_docs, top_n=5)
    if not docs:
        message_content = "Контекст не найден."
    else:
        message_content = re.sub(
            r"\s+",
            " ",
            "\n ".join(
                [
                    f"\nКонтекст-инструкция\n===============" + merged_docs + "\n"
                ]
            ),
        )


    # Инструкции и системные сообщения
    system = "Вы — продвинутый AI-ассистент, специализирующийся на астрологии. Ваша задача — предоставлять точные, детализированные и содержательные ответы на вопросы пользователей по астрологии. Вы должны выдавать чёткие, информативные и полезные ответы, основывающиеся исключительно на переданной инструкции. Поддерживайте профессиональный, но доступный тон, обеспечивая, чтобы ваши объяснения были понятны как новичкам, так и продвинутым пользователям астрологии."
    instructions = """
    1. Вы должны формировать ответ ТОЛЬКО из той информации, которая была вам приведена в качестве контекст-инсрукции.
    2. Если информация из контекст-инструкции кажется вам неполной, ее все равно нельзя дополнять. Исключением является запрос, описанный в пункте 5.
    3. Будьте вежливы по отношению к пользователю, в ответе должна быть информация соответствующая запросу пользователя.
    4. Нельзя отступать от вопроса пользователя.
    """

    # Формирование сообщения для модели
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Контекст: {message_content}\n\nИнструкции: {instructions} \n\nОтвет на задание: {user_query}",
        },
    ]
    # Запрос к OpenAI API
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )


    final_response = completion.choices[0].message.content

    input_tokens = count_tokens(system + message_content + user_query, model="gpt-4o-mini")
    print(input_tokens)
    output_tokens = count_tokens(final_response, model="gpt-4o-mini")
    print(output_tokens)
    cost = calculate_cost(input_tokens, output_tokens, model="gpt-4o-mini")
    print(cost)
    print(f"===ОТВЕТ МОДЕЛИ=== Ответ: {final_response}")

print("dsds")
query = input()
print("2345ds")
process_query(query)
