import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

# Function for Chroma inference
def inference_chroma(chat_model, question, retriever, chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    # Initialize ChatTogether LLM
    chat_model = ChatTogether(
        together_api_key="tgp_v1_XpBlsr-FC2LOjXB2bNF3KA8-VNksD9zZt7OgkjcPaHg",
        model=chat_model,
    )

    # Combine chat history into the question
    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    # Updated prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert financial advisor. Use the context and the appended chat history in the question to answer accurately and concisely.\n\n"
            "Context: {context}\n\n"
            "{question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        ),
    )

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    # Call the chain with the combined question and history
    llm_response = qa_chain(question_with_history)

    print(llm_response['result'])
    return llm_response['result']

# Function for FAISS inference
def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history):
    from langchain.chains import LLMChain
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    import numpy as np

    # Initialize ChatTogether LLM
    chat_model = ChatTogether(
        together_api_key="tgp_v1_XpBlsr-FC2LOjXB2bNF3KA8-VNksD9zZt7OgkjcPaHg",
        model=chat_model,
    )

    # Combine chat history
    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    # Updated PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=(
            "You are an expert financial advisor. Use the context and chat history to answer questions accurately and concisely.\n"
            "Chat History:\n{history}\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        ),
    )

    # Create LLM chain
    qa_chain = LLMChain(
        llm=chat_model,
        prompt=prompt_template,
    )

    # FAISS preprocessing
    query_embedding = embedding_model_global.embed_query(question)
    D, I = index.search(np.array([query_embedding]), k=1)

    # Retrieve the document
    doc_id = I[0][0]
    document = docstore.search(doc_id)
    context = document.page_content

    # Generate the answer using the QA chain
    answer = qa_chain.run(
        history=history_context, context=context, question=question
    )
    print(answer)
    return answer

# Function for Qdrant inference
def inference_qdrant(chat_model, question, embedding_model_global, client, chat_history):
    from qdrant_client.http.models import SearchRequest
    from langchain_together import ChatTogether
    import numpy as np

    # Combine chat history
    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    # Generate query embedding
    query_embedding = embedding_model_global.embed_query(question_with_history)
    query_embedding = np.array(query_embedding)

    # Retrieve relevant documents using Qdrant
    search_results = client.search(
        collection_name="text_vectors",
        query_vector=query_embedding,
        limit=2
    )

    # Combine retrieved contexts
    contexts = [result.payload['page_content'] for result in search_results]
    context = "\n".join(contexts)

    # Updated prompt
    prompt = f"""
    You are a helpful assistant. Use the following retrieved documents to answer the question:

    Context:
    {context}

    {question_with_history}

    Answer:
    """

    # Initialize ChatTogether
    llm = ChatTogether(
        api_key="tgp_v1_XpBlsr-FC2LOjXB2bNF3KA8-VNksD9zZt7OgkjcPaHg",
        model=chat_model
    )

    # Get response
    response = llm.predict(prompt)
    print(response)
    return response

# Function for Pinecone inference
def inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history):
    import numpy as np
    from langchain_together import ChatTogether

    # Generate query embedding
    query_embedding = embedding_model_global.embed_query(question)
    query_embedding = np.array(query_embedding)

    # Search in Pinecone
    search_results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=2,
        include_metadata=True
    )

    # Extract contexts
    contexts = [result['metadata']['text'] for result in search_results['matches']]
    context = "\n".join(contexts)

    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    # Prepare prompt
    prompt = f"""
    You are a helpful assistant. Use the following retrieved documents and chat history to answer the question:
    Chat History:
    {formatted_history}

    Context:
    {context}

    Question: {question}
    Answer:
    """

    llm = ChatTogether(
        api_key="tgp_v1_XpBlsr-FC2LOjXB2bNF3KA8-VNksD9zZt7OgkjcPaHg",
        model=chat_model,
    )

    response = llm.predict(prompt)
    print(response)
    return response

# Function for Weaviate inference
def inference_weaviate(chat_model, question, vs, chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    # Combine chat history
    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    # Create retriever
    retriever = vs.as_retriever()

    # Initialize ChatTogether
    chat_model = ChatTogether(
        together_api_key="tgp_v1_XpBlsr-FC2LOjXB2bNF3KA8-VNksD9zZt7OgkjcPaHg",
        model=chat_model,
    )

    # Retrieve context
    context = retriever.retrieve(question_with_history)

    # Prepare the response
    response = chat_model.predict(context)
    return response

# Main inference function
def inference(vectordb_name, chat_model, question, retriever, embedding_model_global, index, docstore, pinecone_index, vs, chat_history):
    if vectordb_name == "Chroma":
        return inference_chroma(chat_model, question, retriever, chat_history)
    elif vectordb_name == "FAISS":
        return inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history)
    elif vectordb_name == "Qdrant":
        return inference_qdrant(chat_model, question, embedding_model_global, client, chat_history)
    elif vectordb_name == "Pinecone":
        return inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history)
    elif vectordb_name == "Weaviate":
        return inference_weaviate(chat_model, question, vs, chat_history)
    else:
        raise ValueError("Invalid vector database name provided.")
