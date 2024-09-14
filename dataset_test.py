import logging
import os
from typing import List, Tuple
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Load environment variables
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI with your API key
chat_model = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-3.5-turbo",
    max_tokens=250,
    temperature=0.1
)


def load_documents(source_dir: str) -> List[Tuple[str, str]]:
    loaded_text = []
    file_list = os.listdir(source_dir)
    progress_bar = tqdm(file_list, desc="Loading documents")
    for filename in progress_bar:
        file_path = os.path.join(source_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                loaded_text.append((filename, text))
        elif filename.endswith(".pdf"):
            try:
                pdf_text = extract_text(file_path)
                loaded_text.append((filename, pdf_text))
            except Exception as e:
                logging.error(f"Failed to extract text from {filename}: {e}")
        else:
            logging.warning(f"Unsupported file type: {filename}, skipping...")
    return loaded_text


def main():
    SOURCE_DOCUMENTS_DIR = './SOURCE_DOCUMENTS'

    # Step 1: load text (PDFs and TXT files)
    loaded_texts = load_documents(SOURCE_DOCUMENTS_DIR)

    documents = []  # Initialize an empty list to store Document objects

    chunk_size = 1000  # Set the desired chunk size here
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    for filename, text in loaded_texts:
        split_texts = splitter.split_text(text)
        for chunk in split_texts:
            document = Document(
                page_content=chunk,
                metadata={"source": filename, "filename": filename}
            )
            documents.append(document)
    print(f"Created {len(documents)} documents.")
    
    # generator with openai models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=api_key)
    critic_llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    logging.info("Generating testset...")
    # generate testset
    testset = generator.generate_with_langchain_docs(documents, test_size=20, distributions={
                                                    simple: 0.4, reasoning: 0.3, multi_context: 0.3})

    df = testset.to_pandas()
    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv('testset.csv', index=False)
    print("Testset saved to 'testset.csv'.")

    


if __name__ == "__main__":
    main()
