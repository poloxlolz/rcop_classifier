from langchain_core.prompts import ChatPromptTemplate


def get_classification_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """
        You are a legal reasoning assistant.

        Prima facie facts of a case:
        {question}

        Top candidate Penal Code sections:
        {context}

        Task:
        1. Determine which statutory provision (if any) applies.
        2. Think step-by-step and return ONLY a JSON object containing the final classification.

        Output Format (STRICT):
        Return JSON ONLY in the following form:

        "final_classification": "<citation>"
        

        Citation Rules:
        - If an offence is disclosed:
            • <citation> MUST contain at least a section number (e.g., "s.47").
            • It MAY include ONE optional subsection (e.g., "s.47(1)").
            • It MAY include ONE optional paragraph (e.g., "s.47(1)(a)").
            • Do NOT include more than one subsection or more than one paragraph.
            • Do NOT include spaces inside the citation (e.g., "s.47(1)(a)" is correct).

        - If NO offence is disclosed:
            • <citation> MUST be exactly "NOD".

        Important:
        - The final output MUST contain ONLY the JSON object.
        - No explanation or reasoning may appear outside the JSON.
        """
    )
