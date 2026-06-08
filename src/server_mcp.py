import os
import chromadb
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# 1. Initialisation du serveur MCP
# On utilise FastMCP comme indiqué dans le cours
mcp = FastMCP("Serveur_MCP_MIMIC_ICU")

# 2. Initialisation des clients (Ollama + ChromaDB) pour l'outil RAG
llm_client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_collection(name="icu_rag")

# --- DÉFINITION DES OUTILS (TOOLS) MCP ---

@mcp.tool()
def rechercher_informations_cliniques(question: str) -> str:
    """
    Outil MCP : Recherche dans la base documentaire RAG (base vectorielle MIMIC-IV).
    L'agent DOIT utiliser cet outil pour trouver des informations sur les patients âgés, 
    les signes vitaux ou les seuils physiologiques.
    """
    print(f"[Serveur MCP] Requête RAG reçue : {question}")
    try:
        # Embedding de la requête avec bge-m3
        response = llm_client.embeddings.create(model="bge-m3:latest", input=question)
        query_embedding = response.data[0].embedding
        
        # Retrieval vectoriel (Top 3)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        
        # Assembler le contexte récupéré
        if results['documents'] and results['documents'][0]:
            contexte_pertinent = "\n".join(results['documents'][0])
            return f"Informations trouvées dans la base :\n{contexte_pertinent}"
        else:
            return "Aucune information pertinente trouvée dans la base documentaire."
    except Exception as e:
        return f"Erreur lors de la recherche vectorielle : {str(e)}"

@mcp.tool()
def calculatrice_medicale(expression: str) -> str:
    """
    Outil MCP : Exécute un calcul mathématique simple.
    L'agent DOIT utiliser cet outil pour calculer des variations de fréquence cardiaque, 
    des moyennes de pression artérielle ou toute autre opération arithmétique.
    """
    print(f"[Serveur MCP] Requête de calcul reçue : {expression}")
    # Sécurité basique pour éviter l'exécution de code malveillant
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        return "Erreur : expression non autorisée. Utilisez uniquement des chiffres et opérateurs de base."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Résultat du calcul : {result}"
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"

# --- LANCEMENT DU SERVEUR ---

if __name__ == "__main__":
    print("Démarrage du Serveur MCP sur le port 8000...")
    # Le cours recommande d'exposer le serveur sur HTTP via streamable-http
    mcp.run(transport="streamable-http")