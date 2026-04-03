"""
agent/tools.py — LangChain tools that the agent can invoke.

Tools:
  - search_knowledge_base : semantic search over the FAISS vector store
  - get_claims_process     : returns a step-by-step claims guide
"""
from langchain_core.tools import tool
from knowledge_base.retriever import get_retriever


@tool
def search_knowledge_base(query: str) -> str:
    """Search the life insurance knowledge base for information relevant to the query.

    Use this for questions about policy types, eligibility, benefits, riders,
    exclusions, and general life insurance concepts.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@tool
def get_claims_process(placeholder: str = "") -> str:
    """Return the step-by-step process for filing a life insurance claim.

    Call this tool whenever the user asks about how to file a claim,
    required documents, or the claims settlement process.
    The placeholder argument is unused and can be left empty.
    """
    return """
**Life Insurance Claim Process:**

1. **Notify the insurer** – Contact the insurance company as soon as possible after
   the policyholder's death. You can do this via phone, email, or the insurer's portal.

2. **Obtain the claim form** – Download or request a Death Claim Form from the insurer.

3. **Gather required documents:**
   - Original policy document
   - Certified copy of the Death Certificate
   - Claimant's identity proof (Aadhaar / Passport / Voter ID)
   - Claimant's address proof
   - Bank account details (cancelled cheque / passbook copy)
   - Medical records (if death was due to illness)
   - FIR / Post-mortem report (if death was accidental)

4. **Submit the claim** – Submit the completed form and all documents to the nearest
   branch office or upload them through the insurer's online portal.

5. **Claim investigation** – The insurer may conduct a field investigation for large
   claims or cases filed within the first three policy years.

6. **Claim settlement** – Upon verification, the claim amount is transferred directly
   to the nominee's registered bank account, typically within 30 days of receiving
   all documents.

**Tip:** Keep copies of all submitted documents for your records.
"""
