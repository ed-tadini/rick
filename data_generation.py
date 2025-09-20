from document_processing import process_documents_folder
from vector_store import VectorStore
import time

def process_full_collection():
    """Process all documents and create comprehensive knowledge base"""
    print("=== Processing Full Document Collection ===")
    print("This will process all PDFs in your documents folder...")
    print("Expected time: 15-30 minutes for 2000 pages")
    print("Memory usage: 3-4GB peak\n")
    
    start_time = time.time()
    
    #procell all documents
    print("Step 1: Processing all PDFs...")
    all_chunks = process_documents_folder("/Users/edoardo/Desktop/rick_tutor/venv/documents")
    
    if not all_chunks:
        print("No chunks created. Check your documents folder contains readable PDFs.")
        return
    
    #create vectore store
    print(f"\nStep 2: Creating embeddings for {len(all_chunks)} chunks...")
    print("This may take several minutes...")
    
    vs = VectorStore()
    vs.create_vectorstore(all_chunks)
    
    elapsed_time = time.time() - start_time
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Knowledge base ready with {len(all_chunks)} chunks")
    print(f"Database location: ./chroma_db/")

if __name__ == "__main__":
    #confirm before processing
    response = input("Process all documents? This will replace existing database (y/n): ")
    if response.lower() == 'y':
        process_full_collection()
    else:
        print("Processing cancelled.")
