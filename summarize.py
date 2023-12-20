import streamlit as st 
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import io 
import PyPDF2
from docx import Document 
import nltk 

nltk.download('punkt')

model_name="extractive_summarization_model.pkl"
with open(model_name,'rb') as model_file:
    loaded_model=pickle.load(model_file)

def extract_text_from_pdf(pdf_content):
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_content):
    doc = Document(io.BytesIO(docx_content))
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def summarize_model(file_path,num_sentences=3):
  #read the content from the file based on its type 
  if file_path.lower().endswith('.txt'):
    with open(file_path,'r',encoding='utf8') as file:
      file_content=file.read()
  elif file_path.lower().endswith('.pdf'):
    with open(file_path,'rb') as pdf_file:
      pdf_content=pdf_file.read()
      file_content=extract_text_from_pdf(pdf_content)
  elif file_path.lower().endswith('.docx'):
    with open(file_path,'rb') as docx_file:
      docx_content=docx_file.read()
      file_content=extract_text_from_docx(docx_content)
  else:
    raise ValueError("unsupported file format")
  summary_text=loaded_model(file_content,num_sentences=num_sentences)
  return summary_text

def main():
    st.title('TEXT SUMMARIZATION APP')
    uploaded_file=st.file_uploader("choose a file",type=["txt","pdf","docx"])
    
    if uploaded_file is not None:
        st.markdown("### Original Text:")
        content = uploaded_file.read()
        st.code(content, language='text')

        num_sentences = st.slider("Number of Sentences in Summary", 1, 10, 3)
        summary_text = summarize_model(uploaded_file, num_sentences=num_sentences)

        st.markdown("### Summary:")
        st.write(summary_text)

if __name__ == "__main__":
    main()