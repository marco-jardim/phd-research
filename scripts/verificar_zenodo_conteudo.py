"""Auditoria agregada do depósito público, sem exportar registros individuais."""
from pathlib import Path
import csv
import argparse
import difflib
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r'D:\git\phd-research\data\zenodo')
SUSPECT = re.compile(r'DT|DATA|NASC|OBITO|NOTI|DIAG|SEXO|BAIR|MUN|CODMUN|IDPESSOA|REFREC|COMPREC|NOME|ENDER|CEP', re.I)
rows = []
meta = {'data_utc': datetime.now(timezone.utc).isoformat(), 'metodo': 'Cabeçalhos CSV; contagens e cardinalidades sem valores individuais; groupby(dropna=False, observed=True); consulta pública Zenodo, sem alteração do registro.', 'seed': 20261104, 'erros': []}

def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Thesis-deposit-audit/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()

def frame(raw, header_only=False):
    text = io.TextIOWrapper(io.BytesIO(raw), encoding='utf-8-sig')
    first = text.readline()
    text.seek(0)
    sep = ';' if first.count(';') > first.count(',') else ','
    return pd.read_csv(text, sep=sep, dtype=str, nrows=0 if header_only else None)

def k_statistics(df, keys):
    counts = df.groupby(keys, dropna=False, observed=True).size()
    return {'esquema_k': '|'.join(keys), 'k_min_qw3': int(counts.min()), 'n_celulas_qw3': len(counts), 'n_celulas_k_menor_5': int((counts < 5).sum()), 'n_linhas_k_menor_5': int(counts[counts < 5].sum())}

def inventory(version, name, raw, size=None, header_only=False):
    row = {'versao': version, 'arquivo': name, 'tamanho_bytes': size if size is not None else len(raw), 'colunas': '', 'n_linhas': '', 'quase_identificadores': '', 'n_pair_id_distintos': '', 'k_min_qw3': '', 'n_celulas_qw3': '', 'n_celulas_k_menor_5': '', 'n_linhas_k_menor_5': '', 'esquema_k': ''}
    if name.lower().endswith('.csv'):
        head = frame(raw, True)
        row['colunas'] = '|'.join(head.columns)
        row['quase_identificadores'] = '|'.join(c for c in head if SUSPECT.search(c))
        # Full reads are confined to requested counts, never printed as records.
        if not header_only and ('pair_id' in head or name.endswith('qw3_analysis_frame.csv') or name.endswith('benchmark_sim_sinan_tb.csv')):
            df = frame(raw)
            row['n_linhas'] = len(df)
            if 'pair_id' in df:
                row['n_pair_id_distintos'] = df.pair_id.nunique()
            if name.endswith('qw3_analysis_frame.csv'):
                for keys in [['faixa_idade', 'sexo', 'bandeira', 'y_true'], ['faixa_idade', 'sexo', 'bandeira']]:
                    extra = dict(row, **k_statistics(df, keys))
                    rows.append(extra)
                    print('K_QW3', json.dumps({k: extra[k] for k in ['esquema_k','k_min_qw3','n_celulas_qw3','n_celulas_k_menor_5','n_linhas_k_menor_5']}))
                meta['qw3'] = {'n_linhas': len(df), 'colunas': list(df.columns), 'cardinalidades': {c: int(df[c].nunique(dropna=False)) for c in df}}
            if name.endswith('benchmark_sim_sinan_tb.csv'):
                meta.setdefault('benchmark', {})[version] = {'n_linhas': len(df), 'n_colunas': len(df.columns), 'PAR': df.PAR.value_counts(dropna=False).to_dict(), 'colunas': list(df.columns), 'campos_brutos_datas_sexo_municipio': [c for c in df if re.fullmatch(r'(?:[RC]_)?(?:DT\w*|SEXO|CODMUN\w*|MUNICIPIO)', c, re.I)], 'grupos_subescores_publicados': k_statistics(df, [c for c in df if c != 'PAR'])}
                print('BENCHMARK', version, json.dumps(meta['benchmark'][version], ensure_ascii=False))
    rows.append(row)
    print('ARQUIVO', version, name, row['tamanho_bytes'], 'COLUNAS', row['colunas'], 'SUSPEITAS', row['quase_identificadores'], 'PAIR_ID_DISTINTOS', row['n_pair_id_distintos'])

def run():
    records = []
    for url in ['https://zenodo.org/api/records/18652286/versions', 'https://zenodo.org/api/records/23948459/versions', 'https://zenodo.org/api/records?q=conceptrecid%3A18652286&allversions=true&size=100']:
        try:
            answer = json.loads(get(url))
            hits = answer.get('hits', {})
            records = hits.get('hits', []) if isinstance(hits, dict) else hits
            if records:
                break
        except Exception as exc:
            meta['erros'].append({'url': url, 'erro': str(exc)})
    if not records:
        # Known published records can still be verified if the versions endpoint fails.
        for record_id in ['18652286', '23928226', '23929639']:
            try:
                records.append(json.loads(get('https://zenodo.org/api/records/' + record_id)))
            except Exception as exc:
                meta['erros'].append({'registro': record_id, 'erro': str(exc)})
    records = list({record['id']: record for record in records}.values())
    meta['versoes'] = []
    archives = set()
    for record in sorted(records, key=lambda r: r['metadata'].get('version', '')):
        version = record['metadata'].get('version', 'não informada')
        files = record.get('files', [])
        summary = {'versao': version, 'id': record['id'], 'access_right': record['metadata'].get('access_right'), 'arquivos': [{'nome': f['key'], 'tamanho': f['size']} for f in files]}
        meta['versoes'].append(summary)
        print('VERSAO', json.dumps(summary, ensure_ascii=False))
        current = version == '3.0.1'
        for file in files:
            name = file['key']
            link = file['links'].get('self') or file['links'].get('content')
            local = DATA / name
            if not current:
                if name.endswith('.csv'):
                    request = urllib.request.Request(link, headers={'Range': 'bytes=0-65535', 'User-Agent': 'Thesis-deposit-audit/1.0'})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        raw = response.read(65536).splitlines()[0] + b'\n'
                    inventory(version, name, raw, file['size'], header_only=True)
                else:
                    inventory(version, name, b'', file['size'])
                continue
            if current:
                raw = get(link)
                if current and local.exists():
                    meta.setdefault('comparacao_local', {})[name] = {'igual_publicado': hashlib.sha256(local.read_bytes()).hexdigest() == hashlib.sha256(raw).hexdigest(), 'tamanho_local': local.stat().st_size, 'tamanho_publicado': len(raw)}
                inventory(version, name, raw, file['size'])
                if current and name.lower().startswith('readme'):
                    local_raw = local.read_bytes()
                    meta['readme_301'] = {'igual_local': local_raw == raw, 'sha256_local': hashlib.sha256(local_raw).hexdigest(), 'sha256_publicado': hashlib.sha256(raw).hexdigest(), 'diff': difflib.unified_diff(local_raw.decode('utf-8-sig').splitlines(), raw.decode('utf-8-sig').splitlines())}
                    meta['readme_301']['diff'] = list(meta['readme_301']['diff'])
                    print('README_301', json.dumps(meta['readme_301'], ensure_ascii=False))
                if current and name == 'k_anonymity_report.json':
                    meta['k_anonymity_report'] = json.loads(raw)
                    meta['k_anonymity_report']['selected_schema'].pop('smallest_groups_preview', None)
                    print('K_REPORT', json.dumps(meta['k_anonymity_report'], ensure_ascii=False))
                if current and name in ['qw3.zip', 'qw2.zip', 'resultados.zip', 'v2_gzcmd_v31.zip']:
                    archives.add(name)
                    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                        if name == 'v2_gzcmd_v31.zip' and local.exists():
                            with zipfile.ZipFile(local) as previous:
                                before = {m.filename: hashlib.sha256(previous.read(m)).hexdigest() for m in previous.infolist() if not m.is_dir()}
                            after = {m.filename: hashlib.sha256(archive.read(m)).hexdigest() for m in archive.infolist() if not m.is_dir()}
                            meta['zip_301_vs_local_300'] = {'adicionados': sorted(after.keys()-before.keys()), 'removidos': sorted(before.keys()-after.keys()), 'modificados': sorted(k for k in before.keys() & after.keys() if before[k] != after[k]), 'n_membros_301': len(after)}
                        for member in archive.infolist():
                            if not member.is_dir():
                                # Inspect only headers for unrequested CSV contents.
                                payload = archive.read(member) if member.filename.endswith('.csv') else b''
                                inventory(version, name + '/' + member.filename, payload, member.file_size)
                                if member.filename.endswith('privacy_scan_report.json'):
                                    scan = json.loads(archive.read(member))
                                    # Do not retain individual matches or sampled field values.
                                    meta['privacy_scan_report'] = {k: v for k, v in scan.items() if isinstance(v, (int, float, bool))}
                                    meta['privacy_scan_report']['chaves'] = list(scan)
    # Audit local requested archives even if network failed; do not copy them.
    for name in ['qw3.zip', 'qw2.zip', 'resultados.zip', 'v2_gzcmd_v31.zip']:
        if name not in archives:
            if not (DATA / name).exists():
                meta['erros'].append({'arquivo': name, 'erro': 'ZIP ausente localmente e não recuperado pelo GET público'})
                continue
            with zipfile.ZipFile(DATA / name) as archive:
                for member in archive.infolist():
                    if not member.is_dir():
                        inventory('local', name+'/'+member.filename, archive.read(member) if member.filename.endswith('.csv') else b'', member.file_size)
    extrato = pd.read_csv(args.extrato, sep=';', dtype=str)
    # DBF export includes field definitions in each header.
    extrato.columns = [c.split(',')[0].strip() for c in extrato.columns]
    stats = {}
    for col in ['COMPREC', 'REFREC', 'PASSO']:
        values = extrato[col].dropna().str.strip()
        digits = values.str.fullmatch(r'\d+')
        lengths = values[digits].str.len()
        stats[col] = {'n_distintos': int(values.nunique()), 'comprimento_digitos_min': int(lengths.min()), 'comprimento_digitos_max': int(lengths.max()), 'n_nao_numericos': int((~digits).sum()), 'espaco_por_comprimentos_sem_zeros_iniciais': sum(9*10**(k-1) for k in range(int(lengths.min()),int(lengths.max())+1)), 'espaco_por_comprimentos_com_zeros_iniciais': 10**int(lengths.max())}
    meta['identificadores'] = stats
    meta['espaco_observado_produto'] = stats['COMPREC']['n_distintos'] * stats['REFREC']['n_distintos'] * stats['PASSO']['n_distintos']
    meta['espaco_comprimentos_sem_zeros_iniciais'] = stats['COMPREC']['espaco_por_comprimentos_sem_zeros_iniciais'] * stats['REFREC']['espaco_por_comprimentos_sem_zeros_iniciais'] * stats['PASSO']['espaco_por_comprimentos_sem_zeros_iniciais']
    meta['espaco_comprimentos_com_zeros_iniciais'] = stats['COMPREC']['espaco_por_comprimentos_com_zeros_iniciais'] * stats['REFREC']['espaco_por_comprimentos_com_zeros_iniciais'] * stats['PASSO']['espaco_por_comprimentos_com_zeros_iniciais']
    print('IDENTIFICADORES_AGREGADOS', json.dumps(stats), 'PRODUTO_OBSERVADO', meta['espaco_observado_produto'], 'PRODUTO_COMPRIMENTOS', meta['espaco_comprimentos_sem_zeros_iniciais'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--extrato', type=Path, default=Path(r'D:\git\phd-research\data\COMPARADORSEMIDENT.csv'))
    args = parser.parse_args()
    DATA = args.data_dir
    try:
        run()
    finally:
        (ROOT/'data').mkdir(exist_ok=True)
        if rows:
            with (ROOT/'data/zenodo_conteudo_resumo.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (ROOT/'data/zenodo_conteudo_resumo_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        print('ERROS', json.dumps(meta['erros'], ensure_ascii=False))
