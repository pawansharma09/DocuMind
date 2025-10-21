from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_qa_chain
from langchain.prompts import ChatPromptTemplate

def get_conversational_chain(google_api_key):
    """
    Creates and returns a conversational question-answering chain.

    The chain is configured with a specific prompt and the Gemini 2.5 Flash model.
    The new prompt instructs the model to use its own knowledge if the answer is not
    in the provided context, and to state that it's doing so.

    Args:
        google_api_key (str): The Google API key for authentication.

    Returns:
        A LangChain question-answering chain object.
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

    # For LangChain >= 0.1.0, use the new create_qa_chain API
    chain = create_qa_chain(model, prompt=prompt)
    return chain
