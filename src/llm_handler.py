from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_conversational_chain(google_api_key):
    """
    Creates and returns a conversational question-answering chain.

    The chain is configured with a specific prompt and the Gemini 2.5 Flash model.

    Args:
        google_api_key (str): The Google API key for authentication.

    Returns:
        BaseQAChain: A LangChain question-answering chain object.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_api_key
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
