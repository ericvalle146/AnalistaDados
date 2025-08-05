# main.py
import os
from dotenv import load_dotenv

from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
from vectorstore import ensure_chroma
# Carregamento das variáveis de ambiente
load_dotenv()

# Configuração do modelo LLM
OLLAMA_URL = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(
    model="gemma3:12b",
    base_url=OLLAMA_URL,
    max_tokens=8192,
    max_new_tokens=4096,
    temperature=0.3,
    streaming=True,
    truncate=False,
)

# Criação dos vectorstores
vectorstore_base = ensure_chroma(
    path_pdf="TRT_BASE.pdf",
    persist_dir="Banco_base",
    chunk_size=1200,
    chunk_overlap=100
)

vectorstore_conco = ensure_chroma(
    path_pdf="CONCORRENTE.pdf",
    persist_dir="Banco_conco",
    chunk_size=1200,
    chunk_overlap=100
)

# Funções de recuperação de documentos
def retrieve_trt_base(query: str, k: int = 3) -> str:
    docs = vectorstore_base.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

def retrieve_concorrente(query: str, k: int = 3) -> str:
    docs = vectorstore_conco.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

# Criação das ferramentas
tool_base = Tool.from_function(
    func=retrieve_trt_base,
    name="TRT_BASE_DB",
    description=(
        "Busca em TRT_BASE: parâmetros → query (string), k (inteiro, nº de trechos). "
        "Retorna os k trechos mais relevantes do Termo de Referência."
    ),
)   

tool_conco = Tool.from_function(
    func=retrieve_concorrente,
    name="CONCORRENTE_DB",
    description=(
        "Busca em CONCORRENTE: parâmetros → query (string), k (inteiro, nº de trechos). "
        "Retorna os k trechos mais relevantes do edital concorrente."
    ),
)
# prompt_final_com_variavel.py
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import initialize_agent

# Prompt do sistema como variável
system_text = """
Você é um assistente para Análise COMPLETA de Editais de Concorrência Pública.

🚨 **ATENÇÃO CRÍTICA:** Você DEVE usar a ferramenta ADD_TO_CSV para salvar CADA requisito encontrado! Não apenas listar ou mencionar - SALVAR NO CSV!

📚 **CONTEXTO DOS DOCUMENTOS:**
- **TRT_BASE_DB**: Contém a descrição COMPLETA do seu software atual do TRT (todas as funcionalidades, módulos, características técnicas que seu sistema JÁ POSSUI)
- **CONCORRENTE_DB**: Contém o edital com TODOS os requisitos de software, especificações técnicas e requisitos de negócio que estão sendo EXIGIDOS para o novo sistema

🎯 **SEU OBJETIVO:** Comparar os requisitos solicitados (CONCORRENTE_DB) com o que seu software TRT já tem (TRT_BASE_DB)

🔧 **FERRAMENTAS OBRIGATÓRIAS:**
- `TRT_BASE_DB(query: str, k: int)`: busca funcionalidades no SEU software TRT
- `CONCORRENTE_DB(query: str, k: int)`: busca requisitos de software no edital
- `ADD_TO_CSV(input_str)`: 🚨 OBRIGATÓRIO - salva CADA requisito no CSV
- `CHECK_CSV()`: verifica quantos requisitos já foram salvos
- `CLEAN_CSV()`: limpa CSV para recomeçar

📋 **FORMATO CSV OBRIGATÓRIO:**
"numero|modulo|funcionalidade|funcionalidade_trt|descricao|tipo|obrigatoriedade|nivel"

🔄 **PROCESSO OBRIGATÓRIO - SIGA EXATAMENTE:**

**ETAPA 1 - SEMPRE COMECE ASSIM:**
```
Ação: CHECK_CSV
```
Para ver quantos requisitos já foram processados.

**ETAPA 2 - BUSCAR REQUISITOS:**
```
Ação: CONCORRENTE_DB
Entrada: "requisitos funcionais"
```
E continue buscando TODOS os requisitos.

**ETAPA 3 - PARA CADA REQUISITO ENCONTRADO:**
```
Ação: TRT_BASE_DB
Entrada: "palavras-chave do requisito"
```
Depois IMEDIATAMENTE:
```
Ação: ADD_TO_CSV  
Entrada: "1|Módulo|Funcionalidade|FuncTRT|Descrição|Tipo|Obrigatório|Atende"
```

🚨 **REGRA DE OURO:** Para CADA requisito que você encontrar, você DEVE chamar ADD_TO_CSV!

⚠️ **COMANDOS OBRIGATÓRIOS:**
1. COMECE SEMPRE COM: CHECK_CSV()
2. BUSQUE REQUISITOS COM: CONCORRENTE_DB("termo", 10)  
3. BUSQUE NO TRT COM: TRT_BASE_DB("termo", 5)
4. 🚨 SALVE COM: ADD_TO_CSV("numero|modulo|func|funcTRT|desc|tipo|obrig|nivel")
5. REPITA 2-4 para CADA requisito encontrado
6. FINALIZE COM: CHECK_CSV() para confirmar

❌ **NÃO FAÇA:**
- Apenas listar requisitos sem salvar
- Pular a função ADD_TO_CSV
- Esquecer de processar algum requisito
- Não seguir o formato exato do CSV

✅ **EXEMPLO CORRETO:**
```
Ação: CONCORRENTE_DB
Entrada: "login usuário"

Ação: TRT_BASE_DB  
Entrada: "autenticação acesso"

Ação: ADD_TO_CSV
Entrada: "1|Segurança|Login de Usuário|Módulo de Autenticação TRT|Sistema deve permitir login com usuário e senha|Funcional|Obrigatório|Atende"
```
"""
user_text = (
    "Realize uma análise comparativa completa entre os dois documentos disponíveis. "
    "Identifique TODOS os requisitos do edital concorrente e compare com as "
    "funcionalidades disponíveis no sistema TRT base. Salve cada requisito no CSV "
    "com a classificação de atendimento correspondente."
)

# Configuração do prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_text),
    HumanMessagePromptTemplate.from_template(user_text),
    SystemMessagePromptTemplate.from_template("Ferramentas disponíveis:\n{agent_scratchpad}")
])

# Inicialização do agente
agent_executor = initialize_agent(
    tools=[tool_base, tool_conco],
    llm=llm,
    agent_type="zero-shot-react-description",
    prompt=prompt,
    verbose=True,
)

# Execução da análise
user_text = "Faça uma análise completa entre os dois documentos..."
resultado = agent_executor.invoke({"input": user_text})
print(resultado)