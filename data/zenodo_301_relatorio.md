# Publicação Zenodo 3.0.1

## Registro e verificação pública

Publicado em 4 de novembro de 2026:

- DOI da versão corrente: [10.5281/zenodo.23948459](https://doi.org/10.5281/zenodo.23948459).
- DOI conceito, mantido: [10.5281/zenodo.18652286](https://doi.org/10.5281/zenodo.18652286).
- Versão utilizada nos experimentos: [3.0.0, DOI 10.5281/zenodo.23928226](https://doi.org/10.5281/zenodo.23928226).
- GET público, sem autenticação: <https://zenodo.org/api/records/23948459>.

Campos confirmados pela resposta pública da API legada:

```json
{
  "id": 23948459,
  "doi": "10.5281/zenodo.23948459",
  "conceptdoi": "10.5281/zenodo.18652286",
  "metadata": {
    "version": "3.0.1",
    "publication_date": "2026-11-04",
    "license": {"id": "mit-license"}
  }
}
```

`mit-license` é a identificação da licença MIT na serialização legada do Zenodo.
O PUT do metadata usou `license: mit`. A descrição recebeu a nota:
"Versão 3.0.1: corrige a licença declarada no README (MIT); dados idênticos à 3.0.0."
Os demais metadados descritivos foram preservados; DOI e reserva de DOI da
3.0.0 não foram enviados como identificadores da nova versão.

## Arquivos do registro publicado

Inventário confirmado por GET público e comparado ao GET autenticado da 3.0.0.
Os oito arquivos assinalados como inalterados mantêm tamanho e MD5 originais.

| Arquivo | Bytes | MD5 | Situação frente à 3.0.0 |
|---|---:|---|---|
| `README.md` | 5.194 | `d273731d0834282a90fe85ece9bc17d1` | Licença corrigida para MIT e nota 3.0.1 |
| `baselines.json` | 97.309 | `f5c6bbd40ca89e00b1ad99bb313030a9` | Inalterado |
| `benchmark_sim_sinan_tb.csv` | 7.969.179 | `1faf19ed001621d80c6503304429b960` | Inalterado |
| `k_anonymity_report.json` | 4.877 | `d9361e108df6a869b10a49ebcc06cf5e` | Inalterado |
| `manifest.json` | 357 | `12e19d0c14661b97197b994786d71eeb` | Inalterado |
| `qw2.zip` | 791.273 | `e0c5ac07171156b96d9a10870b7ecb18` | Inalterado |
| `qw3.zip` | 257.322 | `f2494e9f66b80042803ca10a676bc1e8` | Inalterado |
| `resultados.zip` | 10.139.396 | `40d6bedd004433196cc67c33622bc7a1` | Inalterado |
| `splits.csv` | 772.446 | `71f1899045cf93f1761384bd14e33f38` | Inalterado |
| `v2_gzcmd_v31.zip` | 16.956.846 | `c63c07a8258323d66a2ffc272fbcfa86` | README, LICENSE, metadata e manifest internos |
| `zenodo_metadata.json` | 1.340 | `05acf4fd712f44216e57dc4735b8a848` | Licença MIT, versão 3.0.1 e nota da correção |

O manifest de nível superior contém somente uma lista de caminhos, sem
checksums, e foi preservado integralmente como documento histórico do pacote-base.
Não houve exclusão em bloco de arquivos herdados. Apenas os três arquivos
assinalados como corrigidos foram substituídos no rascunho.

## Integridade do ZIP GZ-CMD

O link `files[].links.download` da API de depósito retornou HTTP 404. Utilizou-se
a cópia local do ZIP original, cuja identidade com o arquivo publicado foi
confirmada por tamanho, MD5 e SHA-256:

| Propriedade | ZIP 3.0.0 | ZIP 3.0.1 |
|---|---|---|
| Bytes | 16.955.928 | 16.956.846 |
| MD5 | `018dd2cab8f59ee3337eb93080d9e6de` | `c63c07a8258323d66a2ffc272fbcfa86` |
| SHA-256 | `e2c45997540162a65f21a584a19d856bfe12317b10f20c2ddd5efa104a396df1` | `b558f864b3e3dcb23f74c17a2e7d10ee2053f0ffc3550c0c237070b7a50bf2e1` |

O conteúdo original foi extraído para staging separado em
`$env:TEMP\zenodo_301\content\v2_gzcmd_v31`. Todos os caminhos, tamanhos e
SHA-256 do manifest original foram conferidos antes das alterações.

- 149 arquivos no ZIP original; 150 no corrigido.
- 146 arquivos originais mantidos byte a byte, incluindo todos os dados,
  resultados, prompts e documentos de auditoria.
- Alterados somente `README.md`, `zenodo_metadata.json` e `manifest.json`.
- Acrescentado `LICENSE`, texto MIT, titular Marco Elisio Oliveira Jardim, 2026.
- Manifest interno atualizado com SHA-256 e tamanho de cada arquivo, excluindo
  o próprio manifest, conforme o formato original.
- Nenhuma alteração de `pair_id` ou de qualquer arquivo de dados.
- ZIP sem diretório `scripts/`, arquivos executáveis `.py`, `.pyc`, `.log`,
  `.tmp` ou `__pycache__`. A cópia documental `privacy_scan_source.py.txt`,
  já publicada na 3.0.0, permanece idêntica.

O diretório local extraído diferia do ZIP original somente no README, por uma
nota de publicação posterior ao empacotamento. Para evitar incorporar diferenças
não previstas, a correção partiu do ZIP original validado, não desse diretório.

## Proveniência e limites dos scripts

Os helpers não versionados em `data/zenodo/` foram inspecionados, não executados:

- `prepare_v2.py` seleciona artefatos, chama `scan_v2.py`, produz README,
  metadata e manifest específicos do GZ-CMD e cria `v2_gzcmd_v31.zip`.
- `scan_v2.py` implementa a auditoria heurística descrita no pacote. Não foi
  necessário repetir a auditoria dos dados, pois seus bytes foram preservados.
- `upload_v2.py` cria nova versão do depósito 18674440, envia esse ZIP, mantém
  arquivos herdados e grava `v2_published.json`; é compatível com a publicação
  da 3.0.0 observada no inventário e no recibo local, sem constituir por si só
  prova histórica de execução.
- `scripts/qw4_prepare_zenodo.py` gera o pacote-base 2.0.0, não o metadata
  específico desta correção, e permaneceu inalterado.
- `scripts/qw4_zenodo_upload.py` varre todo `data/zenodo/` e apaga arquivos
  herdados; não foi usado.

A publicação empregou um helper temporário fora do repositório, autenticação
por cabeçalho e validação integral do inventário antes do publish e no GET
público posterior. Nenhum token foi impresso ou gravado.

Após a publicação, foram baixados novamente, sem autenticação, o ZIP corrigido
e os dois documentos substituídos de nível superior. Os bytes desses documentos
e os hashes do ZIP coincidiram com o staging. A comparação direta entre o ZIP
público e o original confirmou exclusivamente as três alterações internas e o
acréscimo do LICENSE, sem mudança de dados. Os quatro documentos internos
sincronizados no diretório local também coincidiram com o ZIP público.

## Arquivos locais não versionados

Em `D:\git\phd-research\data\zenodo\v2_gzcmd_v31\`, foram sincronizados com o
staging `README.md`, `zenodo_metadata.json`, `LICENSE` e `manifest.json`.
O recibo `D:\git\phd-research\data\zenodo\v301_published.json` registra DOI,
data, tamanho e hashes do ZIP, inventário público e auditoria de integridade.
O recibo da 3.0.0 não foi substituído. O ZIP local original da 3.0.0 foi mantido;
o ZIP publicado corrigido encontra-se no staging.

As respostas e evidências sem credenciais estão em `$env:TEMP\zenodo_301\`:
`source_deposition.json`, `audit.json`, `draft_verified.json`,
`publish_response.json`, `public_record.json`, `v301_published.json` e
`public_download_verification.json`.

## Citações da tese

A entrada `Jardim2026zenodoGzcmd301` identifica a versão corrente corrigida.
`Jardim2026zenodoGzcmd` continua identificando a 3.0.0 usada nos experimentos;
`Jardim2026zenodo` continua apontando ao DOI conceito. As alterações nos capítulos
de dados, conclusão e Estudo 4 explicitam que a correção não alterou os dados.
O README do repositório não apresentava referência à versão ou à licença do
depósito que exigisse atualização.

## Verificação da tese e testes

- `python scripts/build_pdf.py --jobname tese_zen --outname tese_zen`: sucesso,
  estabilizado em quatro passadas; PDF alternativo de 957 KB, não versionado.
- Busca no log por `Citation .* undefined|Reference .* undefined|^! `: vazia.
- `tese_zen.bbl`: contém `Jardim2026zenodoGzcmd301`, versão 3.0.1,
  licença MIT, nota sobre dados idênticos e URL do registro 23948459.
- `git diff --check`: sem problemas.
- `python -m pytest tests -q`: **249 passed, 5 failed, 2 warnings**, em 14,49 s.
  Não se declara a suíte aprovada.

As cinco falhas decorrem de dependências ausentes no worktree limpo:

1. `test_row_archive_preserves_historical_table_bytes_outside_active_tex`:
   `ImportError: cannot import name 'preservar_m5_m7_row' from 'scripts'`.
2. `test_legacy_v3_cohort_and_guards[vigilancia-1410]`: falta
   `data/COMPARADORSEMIDENT.csv`.
3. `test_legacy_v3_cohort_and_guards[confirmacao-435]`: falta o mesmo extrato.
4. `test_runner_honors_configured_split_by`: falta o mesmo extrato.
5. `test_cascade_table_is_regenerated_from_cells_not_old_counts`: falta
   `data/funil_gzcmd_final_row.csv`, usado como fixture histórica.

Esses caminhos não constam dos arquivos versionados na base `dc25212`.
Não houve alteração de scripts Python, motor GZ-CMD ou testes nesta entrega.
Não foram copiados dados privados para satisfazer os testes, nem alterados ou
suprimidos testes alheios à correção documental. Os avisos foram de divisão
por zero e separação perfeita no teste de modelo loglinear M8.
