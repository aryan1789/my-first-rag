from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(user_question):
    print(f"\nUser Query: {user_question}")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")

    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents.")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Document {i}: {preview}...")

    combined_input = f"""Based on the following documents, please answer this question: {user_question}
    
    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me a question! Type 'quit' to exit.")

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        ask_question(user_input)
        
if __name__ == "__main__":
    start_chat()