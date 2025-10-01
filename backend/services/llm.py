import re
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from .. import config


def _format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """Formats context chunks into a single string for the prompt."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        doc_title = chunk.get("doc_title", "Unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] Document: {doc_title}, Page {page}\n{text}")
    return "\n\n".join(context_parts)


def _convert_history_to_lc(
    chat_history: List[Dict[str, Any]],
) -> List[HumanMessage | AIMessage]:
    """Converts database message history to LangChain message objects."""
    lc_messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["text"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["text"]))
    return lc_messages


def extract_citations(
    answer: str, context_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extracts citation information from the answer based on provided context.
    """
    citations = []
    pattern = r"\(([^,]+),\s*p\.?\s*(\d+)\)"
    matches = re.findall(pattern, answer, re.IGNORECASE)

    for doc_ref, page_str in matches:
        doc_ref = doc_ref.strip()
        try:
            page = int(page_str)
            for chunk in context_chunks:
                if (
                    chunk.get("page") == page
                    and doc_ref.lower() in chunk.get("doc_title", "").lower()
                ):
                    citations.append(
                        {
                            "doc_title": chunk["doc_title"],
                            "page": page,
                            "doc_id": chunk.get("doc_id"),
                        }
                    )
                    break
        except ValueError:
            continue

    # Remove duplicates
    seen = set()
    unique_citations = []
    for cite in citations:
        key = (cite["doc_title"], cite["page"])
        if key not in seen:
            seen.add(key)
            unique_citations.append(cite)

    return unique_citations


def generate_answer_with_history(
    question: str,
    context_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    only_if_sources: bool = False,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Generate an answer using Groq and LangChain with conversation history.
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured")

    # Select the model, defaulting to the one in config
    model_name = model_name or config.GROQ_MODEL

    # Initialize the ChatGroq model
    llm = ChatGroq(
        # FIX 1: Changed 'model_name' to 'model'
        model=model_name,
        temperature=temperature,
        api_key=SecretStr(config.GROQ_API_KEY),
    )

    # System prompt defines the AI's persona and instructions
    system_prompt_template = [
        "You are a helpful assistant that answers questions based solely on the provided context documents.",
        "Your goal is to provide accurate, coherent answers grounded in the retrieved passages.",
        "CONTEXT DOCUMENTS:",
        "{context}",
        "",
        "INSTRUCTIONS:",
        "1. Answer the user's question using ONLY information from the context documents above.",
        "2. Always cite your sources using the format: (Document Title, p. PAGE_NUMBER). You may cite multiple sources.",
    ]
    if only_if_sources:
        system_prompt_template.append(
            '3. If the context documents do not contain enough information to answer the question, respond with: "I cannot answer this question based on the provided documents." Do not use prior knowledge.'
        )
    else:
        system_prompt_template.append(
            "3. If the context is insufficient, acknowledge this limitation in your answer but do not make up information."
        )

    # Create the full prompt template with history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "\n".join(system_prompt_template)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # Create the LangChain chain
    chain = prompt | llm

    # Format context and history for the chain
    formatted_context = _format_context(context_chunks)
    lc_chat_history = _convert_history_to_lc(chat_history)

    # Invoke the chain
    try:
        response = chain.invoke(
            {
                "context": formatted_context,
                "chat_history": lc_chat_history,
                "question": question,
            }
        )
        # FIX 2: Explicitly cast response content to a string to satisfy the type checker
        # and ensure the 'extract_citations' function receives the correct type.
        answer = str(response.content)

        # Extract citations from the generated answer
        citations = extract_citations(answer, context_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "context_chunks": context_chunks,
        }
    except Exception as e:
        logging.error(f"Groq/LangChain API error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate answer: {str(e)}")
