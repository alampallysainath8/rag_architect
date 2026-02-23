"""
Generation module — creates answers using LLM with retrieved context.
"""

import logging
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class Generator:
    """Generates answers using Groq LLM with retrieved context."""
    
    def __init__(self):
        """
        Initialize generator with Groq.
        """
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")
        
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        logger.info(f"Generator initialized with Groq ({settings.GROQ_MODEL})")
    
    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate an answer to a question using retrieved chunks.
        
        Args:
            question: User question.
            chunks: Retrieved/reranked chunks with metadata.
        
        Returns:
            Generated answer.
        """
        if not chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Build context from chunks
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page_number", "?")
            content_type = chunk.get("content_type", "text")
            version = chunk.get("version", 1)
            text = chunk.get("chunk_text", "")
            
            context_parts.append(
                f"[{i}] (Source: {source}, Page: {page}, Version: {version}, Type: {content_type})\n{text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- Include specific references to sources and page numbers in your answer
- If the context doesn't contain enough information to answer the question, say so
- Be concise and informative
- Use bullet points or numbered lists when appropriate

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("Generated answer successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def generate_with_citations(
        self,
        question: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate an answer with explicit citations.
        
        Args:
            question: User question.
            chunks: Retrieved/reranked chunks with metadata.
        
        Returns:
            Dictionary with answer and citations.
        """
        answer = self.generate_answer(question, chunks)
        
        # Extract citations
        citations = []
        for i, chunk in enumerate(chunks, 1):
            citations.append({
                "number": i,
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page_number", "?"),
                "version": chunk.get("version", 1),
                "content_type": chunk.get("content_type", "text"),
                "score": chunk.get("score", 0.0),
                "text_preview": chunk.get("chunk_text", "")[:200] + "..."
            })
        
        return {
            "answer": answer,
            "citations": citations
        }


if __name__ == "__main__":
    print("=== Generator Test ===\n")
    
    # Test with mock chunks
    test_chunks = [
        {
            "id": "test_v1_chunk000",
            "score": 0.85,
            "chunk_text": "The company reported revenue of $100 million in Q4 2024.",
            "source": "earnings.pdf",
            "page_number": 5,
            "content_type": "text",
            "version": 1
        },
        {
            "id": "test_v1_chunk001",
            "score": 0.78,
            "chunk_text": "Net profit increased by 25% compared to the previous quarter.",
            "source": "earnings.pdf",
            "page_number": 6,
            "content_type": "text",
            "version": 1
        }
    ]
    
    test_question = "What was the company's revenue in Q4 2024?"
    
    try:
        generator = Generator()
        
        # Test generation
        result = generator.generate_with_citations(test_question, test_chunks)
        
        print(f"Question: {test_question}\n")
        print(f"Answer: {result['answer']}\n")
        print(f"Citations: {len(result['citations'])}")
        
        for citation in result['citations']:
            print(f"  [{citation['number']}] {citation['source']} (p. {citation['page']})")
        
        print("\n✅ Test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nNote: This test requires valid API keys.")
