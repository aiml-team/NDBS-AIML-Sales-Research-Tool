import os
from docx import Document
from io import BytesIO

def fill_word_template(template_path, model_output):
    """Replaces {{generatedContent}} with the AI-generated company report."""
    
    # Load the template document
    doc = Document("ModelTemplate.docx")

    # Replace placeholder {{generatedContent}}
    for para in doc.paragraphs:
        if "{{generatedContent}}" in para.text:
            para.text = para.text.replace("{{generatedContent}}", model_output)

    # Save to a BytesIO object for Streamlit download
    output_stream = BytesIO()
    doc.save(output_stream)
    output_stream.seek(0)
    
    return output_stream
