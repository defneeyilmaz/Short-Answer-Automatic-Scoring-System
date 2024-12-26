import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

answer_sheet = pd.read_excel("answer_sheet.xlsx")
students_answers = pd.read_excel("answers.xlsx")

print("Answer Sheet Columns:", answer_sheet.columns)
print("Student Answers Columns:", students_answers.columns)

questions_embeddings = {}
student_embeddings = {}

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

print("Available question IDs in questions_embeddings:", questions_embeddings.keys())

for idx, row in students_answers.iterrows():
    student_id = row['Unnamed: 0'] if 'Unnamed: 0' in students_answers.columns else f"Student-{idx+1}"
    student_embeddings[student_id] = {}

    for question_id in questions_embeddings.keys():
        column_name = f"{question_id}-answer" if f"{question_id}-answer" in students_answers.columns else question_id

        if column_name in students_answers.columns:
            answer = str(row[column_name]) if pd.notna(row[column_name]) else ""
            embedding = model.encode(answer)
            student_embeddings[student_id][question_id] = embedding

print("Available student IDs:", student_embeddings.keys())

results = []

for student_id, questions in student_embeddings.items():
    for question_id, student_answer_embedding in questions.items():
        if question_id in questions_embeddings:
            predefined_scores = questions_embeddings[question_id]['Answer_Embeddings']
            max_similarity = 0
            best_score = None

            for score_label, predefined_embedding in predefined_scores.items():
                similarity = cosine_similarity([student_answer_embedding], [predefined_embedding])[0][0]

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_score = score_label

            results.append({
                "Student ID": student_id,
                "Question ID": question_id,
                "Best Matching Score": best_score,
                "Cosine Similarity": max_similarity
            })

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv("similarity_results.csv", index=False)
    print(results_df)
else:
    print("No valid results found. Please check the input files or processing logic.")
