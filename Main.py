import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

answer_sheet = pd.read_excel("answer_sheet.xlsx")
students_answers = pd.read_excel("answers.xlsx")

questions_embeddings = {}
student_embeddings = {}

for index, row  in answer_sheet.iterrows():
    answers = [row['SCORE-1'].replace("\t", ""), row['SCORE-2'].replace("\t", ""), row['SCORE-3'].replace("\t", "")]
    temp = {}
    count1 = 1
    for answer in answers:
        split_answers = [ans.strip() for ans in answer.split(",")]
        encoded_answers = model.encode(split_answers)
        temp['SCORE-' + str(count1)] = {
            "answer": split_answers,
            "embeddings": encoded_answers
        }
        count1 += 1

    questions_embeddings[row['Question- ID']]={
        'Question': row['Question'],
        'Answer_Embeddings': temp
    }

print(questions_embeddings)

for index, row in students_answers.iterrows():
    student_id = row['Unnamed: 0']
    answers = {}
    for question_id in questions_embeddings.keys():
        answer = str(row[question_id+'-answer']) if pd.notna(row[question_id+'-answer']) else ""
        answers[question_id] = model.encode(answer)

    student_embeddings[student_id] = answers

print(student_embeddings)

results=[]

for student_embed in student_embeddings:
    student_result={'Student ID':student_embed}
    total_score = 0
    for question_embed in questions_embeddings:
        max_similarity = 0
        best_answer = None
        best_question = None
        score = None
        for score_key, score_data in questions_embeddings[question_embed]["Answer_Embeddings"].items():
            unparsed_answer = score_data["answer"]
            score_embeddings = score_data["embeddings"]
            for embedding, parsed_answer in zip(score_embeddings,unparsed_answer):
                similarity = cosine_similarity([student_embeddings[student_embed][question_embed]],[embedding])

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_answer = parsed_answer
                    best_question = questions_embeddings[question_embed]["Question"]
                    score = score_key

        total_score = total_score + int(score[-1])
        student_result[question_embed+' - Predicted Score'] = int(score[-1])
        student_result[question_embed + ' - Matching Answer'] = best_answer
        student_result[question_embed + ' - Cosine Similarity'] = float(str(max_similarity).replace("[","").replace("]",""))

    student_result['Student Total Score'] = total_score
    new_order = [
        "Student ID",
        "Student Total Score",
        "Q1 - Predicted Score",
        "Q1 - Matching Answer",
        "Q1 - Cosine Similarity",
        "Q2 - Predicted Score",
        "Q2 - Matching Answer",
        "Q2 - Cosine Similarity",
        "Q3 - Predicted Score",
        "Q3 - Matching Answer",
        "Q3 - Cosine Similarity",
        "Q4 - Predicted Score",
        "Q4 - Matching Answer",
        "Q4 - Cosine Similarity",
    ]
    ordered_student_result = {key: student_result[key] for key in new_order}
    results.append(ordered_student_result)

results_dataframe = pd.DataFrame(results)
results_dataframe.to_excel("results.xlsx", index=False)