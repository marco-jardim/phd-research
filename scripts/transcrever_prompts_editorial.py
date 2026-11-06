"""Transcreve instrumentos literais sem importar ou alterar o módulo histórico."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def role_payloads(root=ROOT):
    module = ast.parse((root / 'gzcmd/llm_review.py').read_text(encoding='utf-8'))
    roles = next(ast.literal_eval(node.value) for node in module.body
                 if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                 and node.target.id == '_ROLE_INSTRUCTION')
    result = [(name, roles[name]) for name in ('Agent-A', 'Agent-B', 'Arbiter')]
    # A chamada efetiva do árbitro utiliza instrução própria, também transcrita.
    arbiter = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)
                   and node.name == '_build_arbiter_messages')
    payload = next(node.value for node in ast.walk(arbiter) if isinstance(node, ast.Constant)
                   and isinstance(node.value, str) and node.value.startswith('\n\nVocê é o Árbitro.'))
    result.append(('Árbitro: chamada efetiva', payload))
    batch = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'build_review_messages')
    payload = next(node.value for node in ast.walk(batch) if isinstance(node, ast.Constant)
                   and isinstance(node.value, str) and node.value.startswith('\n\nCRITICAL:'))
    result.append(('Modo em lote: instrução JSON', payload))
    return result


def generate(root=ROOT, output=None):
    output = output or root / 'text/latex/apendice_prompts.tex'
    text = r'''As instruções de revisão são transcritas integralmente dos arquivos-base v3 e v4, sem edição ou quebra de linhas no conteúdo. Nos arquivos-base, a designação v3.1 corresponde à revisão interna das instruções v3; nesta tese, v3.1 designa somente a política de decisão, e as instruções do primeiro arquivo são chamadas v3.

O carregador extrai a mensagem de sistema até a primeira separadora e a de usuário até a seguinte. Na v3, a política por banda está após a primeira separadora e não foi enviada, embora a mensagem de usuário peça a aplicação da política de decisão adequada à banda; na v4, a política única pelos campos integra o sistema. Títulos, avisos e exemplos do arquivo-base não integram as mensagens enviadas. Os complementos de papel provêm de \path{gzcmd/llm_review.py}: dicionário de papéis, construção da mensagem efetiva do árbitro e instrução adicional do modo em lote. A instrução do árbitro genérico é distinguida da chamada efetivamente usada para arbitragem.

O dossiê efetivamente enviado contém 14 dos 29 subescores do extrato, conforme \path{data/dossie_campos_enviados.csv} e a Tabela~\ref{tab:dossie-campos}. Faltam, entre outros, a penalidade por nome comum da mãe, inversões de data e subescores de UF. As instruções v4 solicitam evidência ausente: inversão de dia e mês, sexo, UF e datas de evento sem correspondência integral no dossiê; os nomes \texttt{PNOME}, \texttt{UNOME} e \texttt{NMAE} não são chaves literais do JSON. Essa incompatibilidade não foi corrigida retroativamente nos instrumentos ou nas execuções. Ausência de evidência não permite avaliar a regra que a solicita.

As faixas do motor são \texttt{low}, abaixo de 5; \texttt{grey\_low}, $[5,6)$; \texttt{grey\_mid}, $[6,7)$; \texttt{grey\_high}, $[7,8)$; \texttt{near\_high}, $[8,9)$; e \texttt{high}, a partir de 9. Os intervalos diferentes no arquivo v3 transcrito abaixo não foram enviados e não descrevem o motor. Os agentes A e B compartilham modelo e dossiê e diferem por uma frase de papel; essa distinção não estabelece independência. A instrução de lote em inglês integra o protocolo majoritariamente em português, sem avaliação de seu efeito. A transcrição preserva os instrumentos, não valida a qualidade dos códigos de razão ou de \texttt{evidence\_summary}.

'''.encode('utf-8')
    for version in ('v3', 'v4'):
        text += ('\\subsection*{Arquivo-base '+version+': transcrição literal}\n\n{\\fontsize{6}{7}\\selectfont\n\\begin{verbatim}\n').encode('utf-8')
        text += (root / f'gzcmd/gzcmd_{version}_llm_prompt.md').read_bytes()
        text += b'\\end{verbatim}\n}\n\n'
    for title, payload in role_payloads(root):
        text += ('\\subsection*{'+title+'}\n\n{\\fontsize{4.5}{6}\\selectfont\n\\begin{verbatim}\n').encode('utf-8')
        # A quebra final é enquadramento LaTeX, não parte da mensagem enviada.
        text += payload.encode('utf-8') + b'\n\\end{verbatim}\n}\n\n'
    output.write_bytes(text.rstrip(b'\n') + b'\n')


if __name__ == '__main__':
    generate()
