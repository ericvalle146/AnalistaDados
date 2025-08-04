from langchain.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain_ollama import ChatOllama
from vectorstore import ensure_chroma
from langchain.agents import initialize_agent, Tool
import os
from dotenv import load_dotenv
load_dotenv()

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

# --- Criar vectorstores, ou persistir ---
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

# --- Funções para tools --- 
def retrieve_trt_base(query: str, k: int = 3) -> str:
    docs = vectorstore_base.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

def retrieve_concorrente(query: str, k: int = 3) -> str:
    docs = vectorstore_conco.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

# --- Tools ---
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


# --- Prompt --- 
system_text = """
Você é um assistente especialista em Análise de Editais de Concorrência Pública, 
com foco específico na seção de Requisitos de Negócio, utilizando técnicas de 
processamento de linguagem natural, embeddings e análise semântica com LangChain 
e ChromaDB. Você receberá dois arquivos PDF:

- **TRT_BASE**: termo de referência que descreve todas as funcionalidades do sistema atual.  
- **CONCORRENTE**: edital contendo os requisitos exigidos por uma prefeitura para aquisição de um sistema.

Use as seguintes ferramentas, informando sempre ambos os parâmetros (`query` e `k`):

- `TRT_BASE_DB(query: str, k: int)`: recupera *k* trechos do Termo de Referência.  
- `CONCORRENTE_DB(query: str, k: int)`: recupera *k* trechos do edital concorrente.  

⚙️ **Fluxo de trabalho técnico e analítico**  
1. **Identificação de Seções**  
   - Liste todos os títulos de seção relacionados a requisitos em ambos os documentos (ex.:  
     “Requisitos de Negócio”, “Funcionalidades do Sistema”, “Especificação Funcional”, módulos etc.).  
   - Extraia integralmente as seções relevantes, estejam elas em listas, tabelas ou textos corridos.

2. **Pré-processamento**  
   - Os chunks do TRT_BASE já estão indexados em ChromaDB.  
   - Os chunks do CONCORRENTE serão recuperados on-demand pela ferramenta, um a um.

3. **Extração de Requisitos do CONCORRENTE**  
   Para cada requisito extraído de **CONCORRENTE_DB**:  
   - **Módulo**: área funcional à qual o requisito pertence (se indicado).  
   - **Funcionalidade**: nome/título da funcionalidade (ex.: “Cadastro de Usuário”).  
   - **Descrição**: texto explicativo ou detalhamento da funcionalidade.  
   - **Tipo do requisito**: “Funcional” ou “Não Funcional”.  
   - **Obrigatoriedade**: “Obrigatório” ou “Opcional”, se indicado.

4. **Busca e Comparação Semântica**  
   Para cada requisito do CONCORRENTE:  
   - Chame **TRT_BASE_DB** com a mesma `query` (ou um trecho-chave) e escolha um `k`.  
   - Identifique no chunk retornado a **Funcionalidade similar do TRT_BASE** com fidelidade contextual.  
   - Classifique o nível de similaridade:  
     - ✅ **Atende** → ≥ 90%  
     - ⚠️ **Atende Parcialmente** → 50–89%  
     - ❌ **Não Atende** → < 50%

5. **Numeração e Normalização**  
   - Atribua um número sequencial a cada requisito na ordem de aparecimento.  
   - Se algum campo não estiver claramente presente, use “Não informado”.

6. **Construção da Tabela Final**  
   Monte uma tabela Markdown (ou CSV) com colunas:  
   Nº | Módulo | Funcionalidade (CONCORRENTE_DB) | Funcionalidade Equivalente (TRT_BASE_DB) | Descrição | Tipo | Obrigatoriedade | Nível de Similaridade


7. **Saída**  
- Retorne **somente** a tabela final em Português Brasileiro, **sem** explicações adicionais.  

📌 **Instruções adicionais**  
- Priorize sempre a identificação correta da funcionalidade equivalente no TRT_BASE.  
- Mantenha fidelidade textual e preserve estruturas originais (listas, tabelas, enumerações).  
- Use “Não informado” para campos ausentes.  
- Siga este fluxo para qualquer novo par de arquivos TRT_BASE e CONCORRENTE.

8. **Exemplo de saída**

"item","módulo","funcionalidade","funcionalidade_similar_trt_base","descrição","tipo_requisito","obrigatoriedade","nível_similaridade"
"1","Requisitos Não Funcionais, Gerais","Responsabilidade pelo Sistema","REQUISITOS TÉCNICOS GERAIS (Item 6.1.1)","O sistema deve ser de responsabilidade da CONTRATADA, não podendo a licitante participar do certame com sistema pelo qual não responda, sendo a vencedora a única pessoa jurídica a prestar os serviços constantes no objeto do contrato.","Não Funcional","Obrigatório","Atende"
"2","Requisitos Não Funcionais, Gerais","Utilização de Software de Apoio","REQUISITOS TÉCNICOS GERAIS (Item 6.1.2)","Exclui-se da limitação do item anterior a possibilidade de utilização do software de apoio aos serviços prestados, não havendo nenhuma responsabilidade da CONTRATANTE com respeito aos direitos de propriedade.","Não Funcional","Obrigatório","Atende"
"3","Requisitos Não Funcionais, Gerais","Infraestrutura de Data Center","QUANTO A HOSPEDAGEM DA SOLUÇÃO DE SOFTWARE EM CENTRO DE DADOS (DATACENTER) (Item 5.1)","A CONTRATADA poderá contratar a infraestrutura especificada em um Data Center de terceiros, desde que atendendo aos requisitos estabelecidos no Termo de Referência, no tópico REQUISITOS DE INFRAESTRUTURA.","Não Funcional","Obrigatório","Atende"
"4","Requisitos Não Funcionais, Gerais","Modelo SaaS (Software as a Service)","PLATAFORMA TECNOLÓGICA E LICENCIAMENTO DA SOLUÇÃO DE SOFTWARE (Item 3.3)","O sistema deverá ser fornecido no modelo SaaS (Software as a Service) – Software como Serviço, sendo a CONTRATADA responsável em fornecer o sistema e toda a estrutura necessária para a sua disponibilização em Data Center.","Não Funcional","Obrigatório","Atende"
"5","Requisitos Não Funcionais, Gerais","Sistema Multiusuário","REQUISITOS TÉCNICOS GERAIS","O sistema deve ser multiusuário, sem limitação de número de usuários com acessos simultâneos, podendo mais de um usuário trabalhar simultaneamente numa mesma tarefa, desde que com dados diferentes, mantendo total integridade dos dados.","Não Funcional","Obrigatório","Atende"
"6","Requisitos Não Funcionais, Gerais","Acesso 'Somente Leitura' à Base de Dados","REQUISITOS TÉCNICOS GERAIS","Acesso, com privilégio de 'somente leitura', a base de dados do sistema da CONTRATADA, pelos técnicos definidos pela Secretaria demandante. O acesso será sob demanda da CONTRATANTE e de acordo com requisitos de segurança e da Lei Geral de Proteção de Dados (LGPD).","Não Funcional","Obrigatório","Atende"
"7","Requisitos Não Funcionais, Gerais","Fornecimento de Base de Dados ao Final do Contrato","Disponibilizar em meio digital e com acesso integral e irrestrito (Item 4.3.1.16)","Ao final do contrato, ou a qualquer tempo em que houver rescisão do contrato, ou sempre que solicitado, a CONTRATADA deverá fornecer, todas as bases de dados contidas no Sistema Gerenciador de Banco de Dados – SGBD.","Não Funcional","Obrigatório","Atende"
"8","Requisitos Não Funcionais, Gerais","Escalabilidade de Servidores","PLATAFORMA TECNOLÓGICA E LICENCIAMENTO DA SOLUÇÃO DE SOFTWARE","Os equipamentos servidores devem permitir escalabilidade visando atender aos aumentos de demanda de acesso concorrente ao sistema.","Não Funcional","Obrigatório","Atende"
"9","Requisitos Não Funcionais, Gerais","Compa  tibilidade com Navegadores Web","QUANTO A COMPATIBILIDADE COM NAVEGADORES DE INTERNET (WEB BROWSERS) (Item 6.6.1)","O sistema deve ser desenvolvido em linguagem nativamente para Web, e permitir o acesso através dos principais navegadores web (browsers): Mozilla Firefox, Google Chrome e Microsoft Edge, em suas últimas versões. Mas, sempre acessível via web browser.","Não Funcional","Obrigatório","Atende"
"""

# --- TESTE --- 
user_text = "Inicie a análise comparativa dos módulos de banco de dados."

# --- Esqueleto metadados ---
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_text),
    HumanMessagePromptTemplate.from_template(user_text),
])

# --- Inicializa o agente com as tools configuráveis ---
agent = initialize_agent(
    tools=[tool_conco, tool_base],
    llm=llm,
    agent_type="zero-shot-react-description",
    prompt=prompt,
    verbose=True,
)

# --- Execute e imprima o resultado ---
resultado = agent.run(user_text)
print(resultado)
