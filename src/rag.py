import chromadb
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_collection(name="icu_rag")

question = "Quels sont les seuils adaptatifs pour la fréquence cardiaque des patients âgés ?"

# 1. Embedding de la requête
response = client.embeddings.create(model="bge-m3:latest", input=question)
query_embedding = response.data[0].embedding

# 2. Retrieval vectoriel (Top 3)
results = collection.query(query_embeddings=[query_embedding], n_results=3)

print("=== Résultats de la recherche vectorielle ===")
for doc, distance in zip(results['documents'][0], results['distances'][0]):
    print(f"Distance L2: {distance:.4f} | Extrait: {doc[:100]}...")

print("\n=== Génération de la réponse finale ===")
# 3. Assembler le contexte récupéré
contexte_pertinent = "\n".join(results['documents'][0])

# 4. Créer le prompt pour le LLM
prompt = f"""Tu es un assistant académique. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni. Si l'information n'est pas dans le contexte, dis-le clairement.

Contexte:
{contexte_pertinent}

Question: {question}
"""

# 5. Appeler le LLM pour générer la réponse (Phase Génération)
reponse_llm = client.chat.completions.create(
    model="qwen2.5:14b", # ou le modèle que tu utilises (ex: llama3, mistral)
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1 # Température basse pour éviter les hallucinations
)

print(reponse_llm.choices[0].message.content)