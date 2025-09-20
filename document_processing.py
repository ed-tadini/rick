import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from dotenv import load_dotenv

load_dotenv()

def process_single_pdf(file_path, source_type="reference"):
    """Process PDF with multiple fallback strategies"""
    print(f"Processing: {file_path}")
    
    #pdf load

    #try 1 PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"PyPDFLoader succeeded - {len(pages)} pages")
    except Exception as e:
        print(f"PyPDFLoader failed: {str(e)[:100]}...")
        
        #try 2 PyMuPDFLoader
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()
            print(f"PyMuPDFLoader succeeded - {len(pages)} pages")
        except Exception as e2:
            print(f"PyMuPDFLoader failed: {str(e2)[:100]}...")
            
            #try 3 PDFPlumberLoader 
            try:
                from langchain_community.document_loaders import PDFPlumberLoader
                loader = PDFPlumberLoader(file_path)
                pages = loader.load()
                print(f" PDFPlumberLoader succeeded - {len(pages)} pages")
            except Exception as e3:
                print(f" All PDF loaders failed for {file_path}")
                print(f"Final error: {str(e3)[:100]}...")
                return []
    
    #check if they extracted content
    if not pages or all(len(page.page_content.strip()) < 50 for page in pages):
        print(f" No meaningful text extracted from {file_path}")
        return []
    
    #metadata - add manual label
    for page in pages:
        page.metadata.update({
            "source_file": os.path.basename(file_path),
            "source_type": source_type,
            "file_path": file_path
        })
    
    #chucks using lagchain recursive method
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(pages)
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def process_documents_folder(folder_path, course_keywords=None, feynman_keywords=None, exercise_keywords=None):
    """Process all PDFs in a folder with smart labeling"""
    if course_keywords is None:
        course_keywords = ["course", "lecture", "notes", "homework", "assignment"]
    
    if feynman_keywords is None:
        feynman_keywords = ["feynman"]

    if exercise_keywords is None:
        exercise_keywords = ["exercise"]
    
    all_chunks = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            
            #smart labelling for feynman and course notes
            source_type = "reference"
            filename_lower = filename.lower()
            
            if any(keyword in filename_lower for keyword in feynman_keywords):
                source_type = "feynman"
            elif any(keyword in filename_lower for keyword in course_keywords):
                source_type = "course_material"
            elif any(keyword in filename_lower for keyword in exercise_keywords):
                source_type = 'exercise'
            
            print(f"\n--- Processing {filename} as {source_type} ---")
            
            try:
                chunks = process_single_pdf(file_path, source_type)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    return all_chunks
