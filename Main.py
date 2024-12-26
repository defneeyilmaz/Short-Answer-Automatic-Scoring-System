import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

answer_sheet = pd.read_excel("answer_sheet.xlsx")

questions_embeddings = {}

for _, row in answer_sheet.iterrows():
    question_id = row['Question- ID']
    question = row['Question']
    answers = [row['SCORE-1'], row['SCORE-2'], row['SCORE-3']]

    question_embedding = model.encode(question)
    answer_embeddings = model.encode(answers)

    questions_embeddings[question_id] = {
        'Question_Embedding': question_embedding,
        'Answer_Embeddings': {
            'SCORE-1': answer_embeddings[0],
            'SCORE-2': answer_embeddings[1],
            'SCORE-3': answer_embeddings[2]
        }
    }

for question_id, embeddings in questions_embeddings.items():
    print(f"Question ID: {question_id}")
    print("Question Embedding Shape:", embeddings['Question_Embedding'].shape)
    for score, embedding in embeddings['Answer_Embeddings'].items():
        print(f"{score} Embedding Shape: {embedding.shape}")