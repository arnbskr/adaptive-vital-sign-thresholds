import chromadb
from pypdf import PdfReader
from openai import OpenAI
import os
import csv

# Configuration pointant vers Ollama local
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_or_create_collection(name="icu_rag")

# --- FONCTIONS D'EXTRACTION ---

def extract_and_chunk_pdf(pdf_path, chunk_size=800, overlap=120):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    
    text = " ".join(text.split()) # Nettoyage basique
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def extract_and_chunk_csv(csv_path):
    chunks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Transforme chaque ligne du tableau en une phrase descriptive
            row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            chunks.append(row_text)
    return chunks

def extract_and_chunk_text(txt_path, chunk_size=800, overlap=120):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = " ".join(text.split())
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# --- SCRIPT PRINCIPAL ---

# Liste des fichiers à ingérer (on peut en ajouter ou en enlever)
fichiers_a_ingerer = [
    "Rapport_Final.pdf", 
    "data/processed/elderly_icu_stays.csv",
    "data/processed/vital_signs_elderly_icu_summary.csv",
    "README.md",
    "2024.knowllm-1.6.pdf",
    "s41597-022-01899-x.pdf"
]

total_chunks_ajoutes = 0

print("=== Début de l'ingestion ===")

for fichier in fichiers_a_ingerer:
    if not os.path.exists(fichier):
        print(f"Fichier introuvable, ignoré : {fichier}")
        continue

    print(f"Traitement de {fichier}...")
    
    # 1. Choix de la méthode selon l'extension
    if fichier.endswith('.pdf'):
        chunks = extract_and_chunk_pdf(fichier)
    elif fichier.endswith('.csv'):
        chunks = extract_and_chunk_csv(fichier)
    elif fichier.endswith('.md') or fichier.endswith('.txt'):
        chunks = extract_and_chunk_text(fichier)
    else:
        print(f"Format non supporté pour {fichier}")
        continue

    # 2. Vectorisation et ajout dans ChromaDB
    for i, chunk in enumerate(chunks):
        if not chunk.strip(): # Ignorer les chunks vides
            continue
            
        response = client.embeddings.create(model="bge-m3:latest", input=chunk)
        embedding = response.data[0].embedding
        
        # Création d'un ID unique basé sur le nom du fichier
        chunk_id = f"{os.path.basename(fichier)}_chunk_{i}"
        
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": fichier, "chunk_index": i}]
        )
    
    total_chunks_ajoutes += len(chunks)
    print(f"{len(chunks)} chunks indexés pour {fichier}.")

print(f"\nTerminé ! {total_chunks_ajoutes} chunks indexés au total dans ChromaDB.")