from langchain_core.prompts import PromptTemplate

research_template = PromptTemplate(
    template="""
{paper_section}

{question_section}

Explanation style: {style_input}
- If style is "Mathematical", you MUST explain the mathematical process exactly like solving mathematics in a notebook.

  IMPORTANT MATHEMATICAL FORMATTING RULES:

  1. Write each mathematical equation on a separate line.
  2. Put the next mathematical operation directly below the previous equation.
  3. Show calculations sequentially from the starting equation to the final result.
  4. Do NOT write "Step 1", "Step 2", "Step 3", etc.
  5. Do NOT combine multiple equations into one paragraph.
  6. Use proper LaTeX for every equation.
  7. Wrap every equation with double dollar signs:
     $$ equation $$
  8. When simplifying or deriving an equation, write every important transformation on a new line.
  9. Explain the meaning of symbols separately after the mathematical derivation.
  10. The mathematical flow should visually look like a student solving a problem line-by-line in a notebook.
- If style is "Code-Oriented", include relevant pseudocode or Python-like code snippets.
- If style is "Beginner-Friendly", use simple analogies and avoid technical jargon.
- If style is "Technical", use precise machine learning / deep learning terminology.

Explanation length: {length_input}
Output language: {language_input}

Make sure the explanation strictly matches the requested style, length, and language.
""",
    input_variables=["paper_section", "question_section", "style_input", "length_input", "language_input"]
)

research_template.save("template.json")