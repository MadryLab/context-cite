import pandas as pd
from app import openai_client, run_rag
import re


def verify_response(row):
    instruction = row.question
    reference_answer = row.answer
    print("q: ", row.question)
    response = run_rag(row.question).response

    EVALUATION_PROMPT = f"""###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
    4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

    ###The instruction to evaluate:
    {instruction}

    ###Response to evaluate:
    {response}

    ###Reference Answer (Score 5):
    {reference_answer}

    ###Score Rubrics:
    [Is the response correct, accurate, and factual based on the reference answer?]
    Score 1: The response is completely incorrect, inaccurate, and/or not factual.
    Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
    Score 3: The response is somewhat correct, accurate, and/or factual.
    Score 4: The response is mostly correct, accurate, and factual.
    Score 5: The response is completely correct, accurate, and factual.

    ###Feedback:"""

    verification = openai_client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": EVALUATION_PROMPT}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    eval_result = verification.choices[0].message.content
    return eval_result


if __name__ == "__main__":
    df = pd.read_json("hpp_qa.json")
    verifications = df.apply(verify_response, axis=1)
    df["Evaluation"] = verifications
    # Define the regular expression pattern
    pattern = r'\[RESULT\] (\d)'
    # Find all matches in the text
    df["Score"] = df["Evaluation"].map(lambda x: re.findall(pattern, x)[0])
    df.to_json("hpp_qa_evaluated.json", indent=4, orient="records")
