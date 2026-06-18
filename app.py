import streamlit as st
import os
from rag_faithfulness_checker import check_faithfulness

# Page config
st.set_page_config(
    page_title="RAG Faithfulness Checker",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Faithfulness Checker")
st.markdown("Verify if AI-generated answers are grounded in source documents")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", 
                            value=os.getenv("GROQ_API_KEY", ""))
    
    user_id = st.text_input("User ID (for logging)", value="streamlit_user")
    
    chunk_size = st.slider("Chunk Size", 100, 1000, 500)
    top_k = st.slider("Top K Results", 1, 10, 5)
    
    st.markdown("---")
    st.markdown("**Security Status:**")
    st.info("Context-aware security enabled\n- Blocks malicious queries\n- Allows legitimate security docs")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Source Document")
    source_text = st.text_area(
        "Paste your source document here:",
        height=300,
        placeholder="Enter the reference document..."
    )

with col2:
    st.subheader("❓ Question & Answer")
    question = st.text_input(
        "Original Question:",
        placeholder="What was asked?"
    )
    answer = st.text_area(
        "AI-Generated Answer:",
        height=200,
        placeholder="Enter the answer to verify..."
    )

# Action button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    check_button = st.button("🔍 Check Faithfulness", type="primary", use_container_width=True)

# Results
if check_button:
    if not source_text.strip():
        st.error("❌ Please provide a source document")
    elif not answer.strip():
        st.error("❌ Please provide an answer to verify")
    elif not api_key.strip():
        st.error("❌ Please provide a Groq API key")
    else:
        with st.spinner("Analyzing faithfulness...."):
            try:
                result = check_faithfulness(
                    answer=answer,
                    source_text=source_text,
                    question=question,
                    api_key=api_key,
                    user_id=user_id,
                    chunk_size=chunk_size,
                    top_k=top_k,
                )
                
                
                # Verdict section
                col1, col2, col3 = st.columns(3)
                with col1:
                    verdict_color = {
                        "FULLY GROUNDED": "green",
                        "PARTIALLY GROUNDED": "orange",
                        "OUT OF CONTEXT": "red"
                    }.get(result.verdict, "gray")
                    st.metric("Verdict", result.verdict, delta=f"{result.grounded_ratio:.0%} Grounded")
                
                with col2:
                    st.metric("Grounded Ratio", f"{result.grounded_ratio:.0%}")
                
                with col3:
                    st.metric("Is Grounded", "✅ Yes" if result.is_grounded else "❌ No")
                
                # Grounded claims
                st.subheader("Grounded Claims")
                if result.grounded_claims:
                    for i, claim in enumerate(result.grounded_claims, 1):
                        st.success(f"{i}. {claim}")
                else:
                    st.info("No grounded claims found")
                
                # Hallucinated claims
                if result.hallucinated_claims:
                    st.subheader("Out-of-Context Claims (Removed)")
                    for i, claim in enumerate(result.hallucinated_claims, 1):
                        st.warning(f"{i}. {claim}")
                
                # Final answer
                st.subheader("Final Answer (Cleaned)")
                st.info(result.final_answer)
                
                # Retrieved context
                with st.expander("📚 Retrieved Context"):
                    st.text(result.retrieved_context)
                
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**RAG Faithfulness Checker v2.1.0**
- Context-aware security with hacking keyword detection
- Three-stage verification (self-report, cosine similarity, LLM tiebreaker)
- Blocks malicious queries outside security documentation context
""")
