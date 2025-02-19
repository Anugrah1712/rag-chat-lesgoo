import streamlit as st
from langchain_core.prompts import ChatPromptTemplate


def inference_chroma(chat_model, question, retriever, chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    # Initialize the ChatTogether LLM
    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    # Append chat history to the question
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

    # Print and return the result
    print(llm_response['result'])
    return llm_response['result']



def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history):
    from langchain.chains import LLMChain
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    import numpy as np

    # Initialize ChatTogether LLM
    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    # Combine chat history into a formatted string
    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    # Updated PromptTemplate to include chat history
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
        history=history_context, context=context, question=question, clean_up_tokenization_spaces=False
    )
    print(answer)

    # Return the answer
    return answer


def inference_qdrant(chat_model, question, embedding_model_global, client, chat_history):
    from qdrant_client.http.models import SearchRequest
    from langchain_together import ChatTogether
    import numpy as np

    # Append chat history to the question
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

    # Updated prompt with appended chat history
    prompt = f"""
    You are a helpful assistant. Use the following retrieved documents to answer the question:

    Context:
    {context}

    {question_with_history}

    Answer:
    """

    # Initialize ChatTogether model
    llm = ChatTogether(
        api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model
    )

    # Get response from LLM
    response = llm.predict(prompt)
    print(response)
    return response


def inference_pinecone(chat_model, question,embedding_model_global, pinecone_index,chat_history):
  import pinecone
  from pinecone import Pinecone
  from langchain_together import ChatTogether
  import numpy as np

  # Initialize Pinecone



  # Step 1: Generate query embedding
  query_embedding = embedding_model_global.embed_query(question)
  query_embedding = np.array(query_embedding)

  # Step 2: Search in Pinecone
    # Replace with your Pinecone index name
  search_results =  pinecone_index.query(
      vector=query_embedding.tolist(),
      top_k=2,  # Retrieve top 2 most relevant results
      include_metadata=True
  )

  # Step 3: Extract context from search results
  # Step 3: Extract context from search results
  # Instead of 'page_content', use 'text' which you used during upsert
  contexts = [result['metadata']['text'] for result in search_results['matches']]

  # Combine contexts for LLM
  context = "\n".join(contexts)

  formatted_history = "\n".join(
      [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
  )

  # Step 4: Prepare prompt for Together.ai
  prompt = f"""
     You are a helpful assistant. Use the following retrieved documents and chat history to answer the question:
     Chat History:
     {formatted_history}

     Context:
     {context}

     Question: {question}
     Answer:
     """
  #llm=ChatmodelInstantiate(chat_model)
  llm = ChatTogether(api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
                  model=chat_model,  )


  # Step 5: Use Together.ai LLM for generation
  response = llm.predict(prompt)
  print(response)
  #st.write(response)
  #st.write("Answer:",response)
  return response


def inference_weaviate(chat_model, question, vs, chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    # Initialize the ChatTogether LLM
    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    # Append chat history to the question
    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    # Updated prompt template
    template = """
    You are an expert financial advisor. Use the context and the appended chat history in the question to answer accurately and concisely:

    Context:
    {context}

    {question}

    Answer (be specific and avoid hallucinations):
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Output parser for the result
    output_parser = StrOutputParser()

    # Create retriever
    retriever = vs.as_retriever()

    # Define the RAG chain
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | chat_model
            | output_parser
    )

    # Invoke the chain with the question containing chat history
    result = rag_chain.invoke(question_with_history)

    # Return the result
    return result


def inference(vectordb_name, chat_model, question,retriever,embedding_model_global,index,docstore,pinecone_index,vs,chat_history):
    if vectordb_name == "Chroma":
        answer=inference_chroma(chat_model, question,retriever,chat_history)
        return answer
    elif vectordb_name == "FAISS":
        answer=inference_faiss(chat_model, question,embedding_model_global,index,docstore,chat_history)
        return answer
    elif vectordb_name == "Qdrant":
        answer=inference_qdrant(chat_model, question,embedding_model_global,client,chat_history)
        return answer
    elif vectordb_name == "Pinecone":
        answer=inference_pinecone(chat_model, question,embedding_model_global, pinecone_index,chat_history)
        return answer
    elif vectordb_name == "Weaviate":
        answer=inference_weaviate(chat_model, question,vs,chat_history)
        return answer
    else:
        print("Invalid Choice")

