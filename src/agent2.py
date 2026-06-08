import asyncio
import json
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# --- CONFIGURATION ---
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
# On utilise la version asynchrone du SDK OpenAI car le client MCP est asynchrone
llm_client = AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
MODEL_NAME = "qwen2.5:14b"

async def run_agent():
    print("=== Démarrage de l'Agent Médical ===")
    
    # 1. Connexion au serveur MCP (comme vu dans le Cours 4)
    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 2. Découverte des outils exposés par le serveur
            mcp_tools = await session.list_tools()
            print(f"✅ Connecté au serveur MCP. Outils découverts : {[t.name for t in mcp_tools.tools]}")
            
            # Conversion du format MCP au format attendu par OpenAI (Ollama)
            openai_tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in mcp_tools.tools]

            # 3. La question de l'utilisateur
            question = "Peux-tu chercher les seuils adaptatifs de fréquence cardiaque pour les plus de 85 ans, puis multiplier le seuil haut par 2 ?"
            print(f"\n🙋‍♂️ Utilisateur : {question}\n")

            # 4. Initialisation de la mémoire (contexte de conversation)
            """messages = [
                {"role": "system", "content": "Tu es un agent clinique expert. Tu disposes d'outils pour chercher des informations médicales et faire des calculs. Tu DOIS utiliser ces outils avant de répondre."},
                {"role": "user", "content": question}
            ]"""

            messages = [
                {"role": "system", "content": """Tu es un agent clinique expert (démonstration, non clinique).
                RÈGLES IMPORTANTES :
                1. Utilise 'retrieve_project_context' (ou 'get_vital_summary') pour trouver des données.
                2. Si tu dois faire un calcul, tu DOIS extraire la valeur numérique exacte du texte AVANT d'utiliser la 'calculatrice_medicale'.
                3. N'envoie JAMAIS de mots ou de variables comme 'resultat' dans la calculatrice. Envoie uniquement des chiffres (ex: '107 * 2')."""},
                {"role": "user", "content": question}
            ]

            # 5. La Boucle Agentique (Raisonnement + Action)
            print("🤖 L'agent réfléchit...")
            
            # Sécurité pour éviter les boucles infinies
            for i in range(5): 
                response = await llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=openai_tools,
                    temperature=0.1
                )
                
                assistant_message = response.choices[0].message
                
                # Si le modèle ne demande pas d'outil, il a fini !
                if not assistant_message.tool_calls:
                    print(f"\n✅ Réponse finale de l'Agent :\n{assistant_message.content}")
                    break
                
                # Si le modèle demande un outil
                messages.append(assistant_message) # On mémorise sa décision
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"   🛠️  L'agent a décidé d'utiliser l'outil : {tool_name} avec les arguments {tool_args}")
                    
                    # 6. Exécution de l'outil via MCP
                    tool_result = await session.call_tool(tool_name, tool_args)
                    result_text = tool_result.content[0].text
                    
                    print(f"   📥 Résultat de l'outil reçu par l'agent.")
                    
                    # On injecte le résultat dans la mémoire pour le prochain tour
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })

if __name__ == "__main__":
    # Lancement de la boucle asynchrone
    asyncio.run(run_agent())