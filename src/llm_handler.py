from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains import create_qa_chain
from langchain_core.prompts import ChatPromptTemplate

def get_conversational_chain(google_api_key):
    """
    Creates and returns a conversational question-answering chain
    using Gemini 2.5 Flash.
    """
    prompt_template = """
You are a helpful assistant. Answer the question based on the provided context.
If the answer is found in the context, provide a detailed response based strictly on that information.
If the answer is not available in the context, use your general knowledge to answer, but you MUST start your response with the phrase:
"This information is not available in the provided documents. Based on my general knowledge, ..."
Do not make up answers from the context if the information is not there.

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        google_api_key=google_api_key
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = create_qa_chain(model, prompt=prompt)
    return chain
