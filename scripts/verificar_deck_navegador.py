"""Verificação do deck de defesa (HTML) no Chrome real, pelo protocolo do DevTools (Playwright, canal "chrome").

Sem rede: o deck é aberto por `file://`. Checagens (código de saída 1 se alguma falhar):

* por quadro e por viewport (padrão 1280x720, 1920x1080 e 1024x768): estouro vertical e horizontal do corpo,
  descendentes fora da caixa do corpo, cabeçalho, rodapé (lembrete, ressalva e fonte) sem sobreposição
  entre si nem com a barra de navegação, o marcador de objetivo ou a faixa do contador; letra do lembrete
  nunca menor que a da linha "Fonte:"; número de linhas e tamanho da letra (px) do lembrete; comparação com
  a checagem embutida `?verificar=1` (`data-total`, `data-estouro`); captura PNG de cada quadro (1280x720);
* console e página: nenhuma mensagem de erro nem exceção de página em toda a sessão;
* navegação: setas, PageDown/PageUp, Home/End, botões, contador, um quadro visível por vez, hash (link
  direto e atualização), remissões a reservas e links "voltar";
* glossário: tecla g, "Neste quadro" igual a `data-termos`, setas bloqueadas com o painel aberto, Esc e foco,
  links de ocorrência, botão Glossário, barra de espaço com botão em foco, foco visível por teclado;
* impressão: PDF com um quadro por página (54) mais as páginas do glossário, sem página em branco, 16:9, e as
  mesmas medidas de sobreposição sob a mídia `print`.

`--calibrar` mede, quadro a quadro e em todos os viewports, o maior número de itens que cabe em cada degrau de letra
do lembrete (normal, compacto, mini) e imprime a tabela `AJUSTES` de scripts/lembretes_deck.py pronta para colar; sem
`--calibrar`, o script falha se a tabela deixar de ser a ótima (checagem de otimalidade). Saída: capturas, `relatorio.json`
e, com o PDF, `deck.pdf`, em `--saida` (padrão `$TEMP/Claude/deck_cdp`).
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DECK = ROOT / "text/slides/defesa.html"
VIEWPORTS = ("1280x720", "1920x1080", "1024x768")
TOL = 1.5

JS_DEF = r"""
window.__medirAtual = function (q) {
  const R = e => e.getBoundingClientRect();
  const fs = parseFloat(getComputedStyle(q).fontSize);
  const qb = R(q), corpo = q.querySelector('.corpo'), pe = q.querySelector('.pe'), cab = q.querySelector('header');
  const lem = q.querySelector('.lembrete'), res = q.querySelector('.ressalva'), fon = q.querySelector('.fonte');
  const marc = q.querySelector('.marcador');
  const ctl = document.querySelector('.controles');
  const falhas = [];
  const cb = R(corpo), pb = R(pe);
  if (q.scrollHeight > q.clientHeight + 1 || q.scrollWidth > q.clientWidth + 1) falhas.push('quadro com rolagem');
  if (corpo.scrollHeight > corpo.clientHeight + 1)
    falhas.push('corpo: scrollHeight ' + corpo.scrollHeight + ' > clientHeight ' + corpo.clientHeight);
  if (corpo.scrollWidth > corpo.clientWidth + 1)
    falhas.push('corpo: scrollWidth ' + corpo.scrollWidth + ' > clientWidth ' + corpo.clientWidth);
  const fora = [];
  corpo.querySelectorAll('*').forEach(e => {
    if (e.closest('svg') && e.tagName.toLowerCase() !== 'svg') return;
    if (e.closest('.sr')) return;
    const b = R(e);
    if (b.width === 0 && b.height === 0) return;
    if (b.bottom > cb.bottom + __TOL__ || b.right > cb.right + __TOL__ ||
        b.top < cb.top - __TOL__ || b.left < cb.left - __TOL__)
      fora.push(e.tagName.toLowerCase() + (e.className && e.className.baseVal === undefined ? '.' + e.className : ''));
  });
  if (fora.length) falhas.push('fora da caixa do corpo: ' + fora.slice(0, 5).join(', '));
  if (cab.scrollWidth > cab.clientWidth + 1) falhas.push('cabeçalho com estouro horizontal');
  if (R(cab).bottom > cb.top + __TOL__) falhas.push('cabeçalho sobre o corpo');
  if (cb.bottom > pb.top + __TOL__) falhas.push('corpo sobre o rodapé');
  const limite = qb.bottom - 2 * fs + __TOL__;
  if (pb.bottom > limite) falhas.push('rodapé invade a faixa do contador/navegação');
  Array.prototype.forEach.call(q.childNodes, n => {
    if (n.nodeType === 3 && n.textContent.trim()) falhas.push('texto solto no quadro');
  });
  const caixas = {};
  if (lem) caixas.lembrete = R(lem);
  if (res) caixas.ressalva = R(res);
  if (fon) caixas.fonte = R(fon);
  const nomes = Object.keys(caixas);
  const cruza = (a, b) => a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1;
  for (let i = 0; i < nomes.length; i++) for (let j = i + 1; j < nomes.length; j++)
    if (cruza(caixas[nomes[i]], caixas[nomes[j]])) falhas.push('sobreposição: ' + nomes[i] + ' x ' + nomes[j]);
  nomes.forEach(n => {
    if (caixas[n].left < pb.left - 1 || caixas[n].right > pb.right + 1) falhas.push(n + ' fora da largura do rodapé');
    if (caixas[n].bottom > limite) falhas.push(n + ' na faixa do contador');
  });
  const uniao = el => { const fs = Array.from(el.children).map(R); return { left: Math.min(...fs.map(r => r.left)), right: Math.max(...fs.map(r => r.right)), top: Math.min(...fs.map(r => r.top)), bottom: Math.max(...fs.map(r => r.bottom)) }; };
  const cx = ctl && getComputedStyle(ctl).display !== 'none' ? R(ctl) : null;
  if (cx) {
    nomes.concat(marc ? ['marcador'] : []).forEach(n => {
      const b = n === 'marcador' ? uniao(marc) : caixas[n];
      if (cruza(b, cx)) falhas.push('sobreposição com a barra de navegação: ' + n);
    });
  }
  const cont = (() => {
    // faixa do contador (::after): canto direito, última linha do quadro; o rodapé já está acima dela
    return null;
  })();
  let lemInfo = null;
  if (lem) {
    const est = getComputedStyle(lem);
    const util = R(lem).height - parseFloat(est.paddingTop) - parseFloat(est.borderTopWidth);
    const fonteFonte = fon ? parseFloat(getComputedStyle(fon).fontSize) : null;
    const px = parseFloat(est.fontSize);
    lemInfo = {
      px: Math.round(px * 100) / 100, linhas: Math.round(util / parseFloat(est.lineHeight)),
      classe: lem.className, px_fonte: fonteFonte,
      mostrados: Array.from(lem.querySelectorAll('.it')).filter(s => !/^demais termos/.test(s.textContent)).length,
      caracteres: lem.textContent.replace(lem.querySelector('.rot').textContent, '').trim().length,
      cortado: /demais termos/.test(lem.textContent), largura_ok: lem.scrollWidth <= lem.clientWidth + 1
    };
    if (fonteFonte !== null && px + 0.01 < fonteFonte) falhas.push('letra do lembrete (' + px + 'px) menor que a da fonte (' + fonteFonte + 'px)');
    if (!lemInfo.largura_ok) falhas.push('lembrete com estouro horizontal');
  }
  return { id: q.id, falhas: falhas, lembrete: lemInfo, corpo_px: Math.round(parseFloat(getComputedStyle(corpo).fontSize) * 100) / 100 };
};
window.__mostrar = async function (id) {
  const q = document.getElementById(id);
  if (getComputedStyle(q).display === 'none') {
    location.hash = '#' + id;
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  }
  return q;
};
window.__medir = async function (id) { return window.__medirAtual(await window.__mostrar(id)); };
window.__calibrar = async function (id, itens, fixo) {
  // Maior número de itens que cabe em cada degrau (a lista completa vem do glossário, por ordem de prioridade),
  // com a densidade atual do quadro e com um degrau de densidade a menos (só reservas).
  const q = await window.__mostrar(id);
  const lem = q.querySelector('.lembrete');
  if (!lem) return { id: id, sem_lembrete: true };
  const origClasse = lem.className, origHtml = lem.innerHTML, origSec = q.className;
  const rot = '<strong class="rot">Lembrete:</strong> ';
  const k = itens.length;
  const cabe = () => window.__medirAtual(q).falhas.length === 0;
  const montar = (d, n) => {
    lem.className = 'lembrete' + (d ? ' ' + d : '');
    lem.innerHTML = rot + itens.slice(0, n).join(' ') + (n < k ? ' ' + fixo : '');
  };
  const nmax = () => {
    const out = {};
    [['normal', ''], ['compacto', 'compacto'], ['mini', 'mini']].forEach(par => {
      let n = k, ok = false;
      for (; n >= 1; n--) { montar(par[1], n); if (cabe()) { ok = true; break; } }
      out[par[0]] = ok ? n : 0;
    });
    return out;
  };
  const res = { id: id, k: k, atual: nmax(), proxima: null, proxima_nmax: null };
  const seq = ['', 'denso', 'denso1', 'denso2', 'denso3'];
  const cur = seq.find(c => c && q.classList.contains(c)) || '';
  if (q.getAttribute('data-tipo') === 'reserva' && seq.indexOf(cur) < seq.length - 1) {
    const prox = seq[seq.indexOf(cur) + 1];
    if (cur) q.classList.remove(cur);
    q.classList.add(prox);
    res.proxima = prox;
    res.proxima_nmax = nmax();
  }
  q.className = origSec; lem.className = origClasse; lem.innerHTML = origHtml;
  return res;
};
window.__estado = function () {
  const visiveis = Array.from(document.querySelectorAll('#palco > .slide')).filter(s => getComputedStyle(s).display !== 'none').map(s => s.id);
  const a = document.activeElement;
  return { contador: document.getElementById('contador').textContent, hash: location.hash, visiveis: visiveis,
           foco: a ? (a.id || a.tagName.toLowerCase()) : null,
           glossario: !document.getElementById('glossario').hidden, indice: !document.getElementById('indice').hidden };
};
""".replace("__TOL__", str(TOL))


class Relatorio:
    def __init__(self):
        self.checagens = []

    def reg(self, nome, ok, detalhe=""):
        self.checagens.append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})
        print(f"[{'OK' if ok else 'FALHA'}] {nome}" + (f": {detalhe}" if detalhe else ""), flush=True)

    @property
    def falhou(self):
        return any(not c["ok"] for c in self.checagens)


def _esperar(page, ms=60):
    page.wait_for_timeout(ms)


def estado(page):
    return page.evaluate("window.__estado()")


def ir(page, qid):
    page.evaluate("async (id) => { location.hash = '#' + id; await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))); }", qid)


def abrir(browser, url, vw, vh, erros):
    ctx = browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=1)
    page = ctx.new_page()
    page.on("console", lambda m: erros.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: erros.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: erros.append(f"requestfailed: {r.url} {r.failure}"))
    page.goto(url)
    page.wait_for_load_state("load")
    page.evaluate(JS_DEF)
    return ctx, page


def ids_dos_quadros(page):
    return page.evaluate("Array.from(document.querySelectorAll('#palco > .slide')).map(s => s.id)")


# ------------------------------------------------------------------ medidas por quadro e viewport

def medidas(browser, url, vw, vh, saida, capturar, rel, erros):
    ctx, page = abrir(browser, url + "?verificar=1", vw, vh, erros)
    page.wait_for_function("document.documentElement.dataset.total !== undefined", timeout=20000)
    emb = page.evaluate("({total: document.documentElement.dataset.total, estouro: document.documentElement.dataset.estouro})")
    ids = ids_dos_quadros(page)
    linhas = {}
    resultados = []
    for qid in ids:
        m = page.evaluate("(id) => window.__medir(id)", qid)
        resultados.append(m)
        if capturar:
            page.screenshot(path=str(saida / "capturas" / f"{qid}.png"))
        linhas[qid] = page.evaluate("(id) => document.getElementById(id).getAttribute('data-linhas-lembrete')", qid)
    tag = f"{vw}x{vh}"
    com_falha = [m for m in resultados if m["falhas"]]
    rel.reg(f"[{tag}] medidas próprias: nenhum quadro com estouro ou sobreposição ({len(ids)} quadros)", not com_falha,
            "; ".join(f"{m['id']}: {m['falhas']}" for m in com_falha))
    rel.reg(f"[{tag}] checagem embutida ?verificar=1: data-total={emb['total']}, data-estouro={emb['estouro']}",
            emb["total"] == str(len(ids)) and emb["estouro"] == "nenhum")
    discord = [m["id"] for m in resultados if m["lembrete"] and str(m["lembrete"]["linhas"]) != linhas[m["id"]]]
    rel.reg(f"[{tag}] linhas do lembrete iguais às da checagem embutida", not discord, ", ".join(discord))
    ctx.close()
    return resultados


# ------------------------------------------------------------------ calibração

ORDEM = ("normal", "compacto", "mini")
CLASSE = {"normal": "", "compacto": "compacto", "mini": "mini"}


def recomendar(nmax, k):
    """(degrau, itens mantidos): o MAIOR número de itens que cabe em algum degrau (None = lista inteira); em empate,
    o degrau de maior letra. None se nem um item cabe em mini."""
    melhor = max(min(nmax[d], k) for d in ORDEM)
    if melhor < 1:
        return None
    degrau = next(d for d in ORDEM if min(nmax[d], k) == melhor)
    return (CLASSE[degrau], None if melhor >= k else melhor)


def itens_da_recomendacao(rec, k):
    return 0 if rec is None else (k if rec[1] is None else rec[1])


def calibrar(browser, url, viewports, html_texto, erros):
    """qid -> {"k", "rec", "descer"}: medição em todos os viewports (vale o pior caso de cada degrau)."""
    from scripts import lembretes_deck as lem
    completos = lem.itens_completos(html_texto)
    agreg = {}
    for vp in viewports:
        vw, vh = (int(x) for x in vp.split("x"))
        ctx, page = abrir(browser, url, vw, vh, erros)
        for qid, itens in completos.items():
            r = page.evaluate("([id, itens, fixo]) => window.__calibrar(id, itens, fixo)", [qid, itens, lem.FIXO_HTML])
            a = agreg.setdefault(qid, {"k": r["k"], "atual": {d: 10 ** 6 for d in ORDEM}, "proxima": r["proxima"], "prox": {d: 10 ** 6 for d in ORDEM}})
            for d in ORDEM:
                a["atual"][d] = min(a["atual"][d], r["atual"][d])
                if r["proxima_nmax"]:
                    a["prox"][d] = min(a["prox"][d], r["proxima_nmax"][d])
        ctx.close()
    saida = {}
    for qid, a in agreg.items():
        rec = recomendar(a["atual"], a["k"])
        descer = None
        if rec is not None and rec[1] is not None and a["proxima"] and qid not in lem.DESCIDAS:
            alt = recomendar(a["prox"], a["k"])
            if itens_da_recomendacao(alt, a["k"]) > itens_da_recomendacao(rec, a["k"]):
                descer = {"para": a["proxima"], "rec": alt}
        saida[qid] = {"k": a["k"], "rec": rec, "descer": descer, "atual": a["atual"]}
    return saida


def tabela_pronta(cal):
    linhas = ["AJUSTES = {"]
    for qid in sorted(cal):
        rec = cal[qid]["rec"] if not cal[qid]["descer"] else cal[qid]["descer"]["rec"]
        linhas.append(f'    "{qid}": {rec!r},' if rec is not None else f'    "{qid}": None,  # nem um item cabe em mini')
    linhas.append("}")
    return "\n".join(linhas)


def checar_otimalidade(cal, rel):
    """A tabela AJUSTES tem de ser a que a medição recomenda: nenhum quadro pode subir de degrau, ganhar item ou
    descer a densidade sem estourar."""
    from scripts import lembretes_deck as lem
    ruins = []
    for qid, c in sorted(cal.items()):
        tab = lem.ajuste_do_quadro(qid)
        if c["descer"]:
            ruins.append(f"{qid}: lista inteira cabe com a densidade {c['descer']['para']} (tabela {tab})")
        elif c["rec"] is None:
            ruins.append(f"{qid}: nem um item cabe em mini")
        elif tuple(c["rec"]) != tuple(tab):
            ruins.append(f"{qid}: tabela {tab}, medição recomenda {tuple(c['rec'])} (lista de {c['k']})")
    sobras = sorted(q for q in lem.AJUSTES if q not in cal)
    if sobras:
        ruins.append(f"tabela com quadros sem lembrete: {sobras}")
    rel.reg(f"otimalidade: a tabela AJUSTES é a recomendada pela medição em todos os viewports ({len(cal)} quadros)", not ruins, "; ".join(ruins))


# ------------------------------------------------------------------ navegação

def checar_navegacao(browser, url, rel, erros):
    ctx, page = abrir(browser, url + "#S16", 1280, 720, erros)
    ids = ids_dos_quadros(page)
    total = len(ids)
    e = estado(page)
    rel.reg("link direto #S16", e["visiveis"] == ["S16"] and e["contador"] == f"S16 · {ids.index('S16') + 1} / {total}", str(e))
    ir(page, "R08")
    e = estado(page)
    rel.reg("link direto #R08", e["visiveis"] == ["R08"] and e["contador"] == f"R08 · {ids.index('R08') + 1} / {total}", str(e))
    ir(page, "R10")
    e = estado(page)
    rel.reg("contador de R10", e["contador"] == "R10 · 38 / 54", e["contador"])

    page.keyboard.press("Home")
    e = estado(page)
    rel.reg("Home leva a S01", e["visiveis"] == ["S01"] and e["hash"] in ("#S01", ""), str(e))
    ruins = []
    for i in range(1, total):
        page.keyboard.press("ArrowRight")
        e = estado(page)
        if e["visiveis"] != [ids[i]] or e["contador"] != f"{ids[i]} · {i + 1} / {total}" or e["hash"] != "#" + ids[i]:
            ruins.append((ids[i], e))
    rel.reg(f"ArrowRight percorre os {total} quadros: contador, hash e um quadro visível por vez", not ruins, str(ruins[:3]))
    page.keyboard.press("ArrowRight")
    e = estado(page)
    rel.reg("ArrowRight no último quadro não avança", e["visiveis"] == [ids[-1]], str(e))
    ruins = []
    for i in range(total - 2, -1, -1):
        page.keyboard.press("ArrowLeft")
        e = estado(page)
        if e["visiveis"] != [ids[i]] or e["contador"] != f"{ids[i]} · {i + 1} / {total}" or e["hash"] != "#" + ids[i]:
            ruins.append((ids[i], e))
    rel.reg("ArrowLeft volta pelos quadros até S01", not ruins, str(ruins[:3]))
    page.keyboard.press("ArrowLeft")
    rel.reg("ArrowLeft em S01 não recua", estado(page)["visiveis"] == ["S01"])

    page.keyboard.press("PageDown")
    ok1 = estado(page)["visiveis"] == ["S02"]
    page.keyboard.press("PageUp")
    ok2 = estado(page)["visiveis"] == ["S01"]
    rel.reg("PageDown e PageUp", ok1 and ok2)
    page.keyboard.press("End")
    e = estado(page)
    rel.reg("End leva ao último quadro (R26, documentado)", e["visiveis"] == [ids[-1]] and e["contador"].startswith(ids[-1]), e["contador"])
    page.keyboard.press("Home")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowUp")
    rel.reg("ArrowDown e ArrowUp", estado(page)["visiveis"] == ["S02"])
    page.keyboard.press("Home")

    # botões
    page.click("#prox")
    rel.reg("botão Próximo", estado(page)["visiveis"] == ["S02"])
    page.click("#ant")
    rel.reg("botão Anterior", estado(page)["visiveis"] == ["S01"])
    rel.reg("botão Anterior desabilitado em S01", page.evaluate("document.getElementById('ant').disabled"))
    page.keyboard.press("End")
    rel.reg("botão Próximo desabilitado no último quadro", page.evaluate("document.getElementById('prox').disabled"))
    page.keyboard.press("Home")
    page.click("#botao-indice")
    e = estado(page)
    rel.reg("botão Índice abre o índice", e["indice"], str(e))
    page.keyboard.press("ArrowRight")
    rel.reg("com o índice aberto as setas não mudam de quadro", estado(page)["visiveis"] == ["S01"])
    page.keyboard.press("Escape")
    e = estado(page)
    rel.reg("Esc fecha o índice e devolve o foco ao botão Índice", (not e["indice"]) and e["foco"] == "botao-indice", str(e))
    page.keyboard.press("i")
    ab = estado(page)["indice"]
    page.keyboard.press("i")
    rel.reg("tecla i abre e fecha o índice", ab and not estado(page)["indice"])
    page.click("#botao-indice")
    page.click('#indice a[href="#S10"]')
    e = estado(page)
    rel.reg("link do índice leva ao quadro e fecha o painel", e["visiveis"] == ["S10"] and not e["indice"], str(e))

    # espaço
    page.keyboard.press("Home")
    page.evaluate("document.activeElement.blur()")           # o foco ficou no botão Índice: o espaço o acionaria
    page.keyboard.press(" ")
    rel.reg("barra de espaço sem foco em controle avança um quadro", estado(page)["visiveis"] == ["S02"], str(estado(page)))
    page.focus("#prox")
    page.keyboard.press(" ")
    e = estado(page)
    rel.reg("barra de espaço com o botão Próximo em foco aciona o botão (avança exatamente um quadro)", e["visiveis"] == ["S03"], str(e))
    page.focus("#botao-glossario")
    page.keyboard.press(" ")
    e = estado(page)
    rel.reg("barra de espaço com o botão Glossário em foco abre o painel e não muda de quadro", e["glossario"] and e["visiveis"] == ["S03"], str(e))
    page.keyboard.press("Escape")

    # remissões e "voltar"
    pares = page.evaluate("""Array.from(document.querySelectorAll('#palco > .slide[id^="S"] a[href^="#R"]')).map(a => [a.closest('.slide').id, a.getAttribute('href').slice(1)])""")
    ruins = []
    for origem, destino in pares:
        ir(page, origem)
        page.click(f'#{origem} a[href="#{destino}"]')
        e1 = estado(page)
        page.click(f"#{destino} a.voltar")
        e2 = estado(page)
        if e1["visiveis"] != [destino] or e2["visiveis"] != [origem] or e2["hash"] != "#" + origem:
            ruins.append((origem, destino, e1["visiveis"], e2["visiveis"], e2["hash"]))
    rel.reg(f"remissões dos principais às reservas e retorno por \"← voltar\" ({len(pares)} remissões)", pares and not ruins, str(ruins[:3]))
    # voltar sem histórico de remissão (entrada direta na reserva)
    voltares = page.evaluate("""Array.from(document.querySelectorAll('#palco > .slide[id^="R"]')).map(s => { const a = s.querySelector('a.voltar'); return [s.id, a ? a.getAttribute('href').slice(1) : null]; })""")
    ruins = []
    for rid, alvo in voltares:
        if alvo is None:
            ruins.append((rid, "sem link voltar"))
            continue
        ir(page, rid)
        page.evaluate("history.replaceState(null, '', location.hash)")
        page.click(f"#{rid} a.voltar")
        e = estado(page)
        if e["visiveis"] != [alvo]:
            ruins.append((rid, alvo, e["visiveis"]))
    rel.reg(f"\"← voltar\" de cada reserva leva ao quadro de origem ({len(voltares)} reservas)", not ruins, str(ruins[:3]))
    # histórico do navegador após remissão
    if pares:
        origem, destino = pares[0]
        ir(page, origem)
        page.click(f'#{origem} a[href="#{destino}"]')
        page.go_back()
        _esperar(page, 120)
        rel.reg("voltar do navegador após remissão retorna ao quadro de origem", estado(page)["visiveis"] == [origem], str(estado(page)))
    ctx.close()


# ------------------------------------------------------------------ glossário

def checar_glossario(browser, url, rel, erros):
    ctx, page = abrir(browser, url, 1280, 720, erros)
    ids = ids_dos_quadros(page)
    esperados = page.evaluate("""(() => { const m = {};
      document.querySelectorAll('#palco > .slide').forEach(s => {
        const termos = (s.getAttribute('data-termos') || '').split('|').filter(Boolean);
        m[s.id] = termos.map(c => { for (const dt of document.querySelectorAll('#lista-glossario dt'))
          if (dt.getAttribute('data-chave') === c) return dt.textContent.replace(/\\s+/g, ' ').trim(); return '??' + c; }); });
      return m; })()""")
    for qid in ("S01", "S10", "S16", "S23", "R08", "R19"):
        ir(page, qid)
        page.focus("#prox")
        page.keyboard.press("g")
        e = estado(page)
        lista = page.evaluate("""Array.from(document.querySelectorAll('#neste-quadro li')).map(li => { const s = li.querySelector('strong'); return s ? s.textContent.replace(/\\s+/g, ' ').trim() : li.textContent.trim(); })""")
        titulo = page.evaluate("document.getElementById('neste-quadro-titulo').textContent")
        esperado = esperados[qid] or ["Nenhum termo da lista neste quadro."]
        rel.reg(f"tecla g em {qid}: painel aberto e \"Neste quadro\" = data-termos ({len(esperados[qid])} termos)",
                e["glossario"] and lista == esperado and titulo == f"Neste quadro ({qid})", f"{titulo}; lista={lista}; esperado={esperado}" if lista != esperado else titulo)
        page.keyboard.press("ArrowRight")
        page.keyboard.press("End")
        rel.reg(f"painel aberto em {qid}: setas e End não mudam de quadro", estado(page)["visiveis"] == [qid])
        page.keyboard.press("Escape")
        e = estado(page)
        rel.reg(f"Esc em {qid} fecha o painel e devolve o foco ao botão que o tinha", (not e["glossario"]) and e["foco"] == "prox", str(e))
        page.keyboard.press("g")
        a = estado(page)["glossario"]
        page.keyboard.press("g")
        rel.reg(f"g de novo em {qid} fecha o painel", a and not estado(page)["glossario"])
    # botão Glossário
    ir(page, "S10")
    page.click("#botao-glossario")
    e = estado(page)
    lista = page.evaluate("Array.from(document.querySelectorAll('#neste-quadro li strong')).map(s => s.textContent.replace(/\\s+/g, ' ').trim())")
    rel.reg("botão Glossário faz o mesmo que a tecla g", e["glossario"] and lista == esperados["S10"], str(e))
    page.click("#fechar-glossario")
    e = estado(page)
    rel.reg("botão Fechar do glossário fecha e devolve o foco ao botão Glossário", (not e["glossario"]) and e["foco"] == "botao-glossario", str(e))
    page.click("#botao-glossario")
    page.mouse.click(5, 5)
    rel.reg("clique no fundo do painel fecha o glossário", not estado(page)["glossario"])
    # links de ocorrência
    for origem, destino in (("S01", "S16"), ("S10", "R08"), ("R19", "S02")):
        ir(page, origem)
        page.keyboard.press("g")
        page.eval_on_selector(f'#glossario dd.ocor a[href="#{destino}"]', "a => a.scrollIntoView()")
        page.click(f'#glossario dd.ocor a[href="#{destino}"]')
        e = estado(page)
        rel.reg(f"link de ocorrência {origem} -> {destino}: vai ao quadro, fecha o painel e atualiza o hash",
                e["visiveis"] == [destino] and not e["glossario"] and e["hash"] == "#" + destino, str(e))
    # todos os links de ocorrência apontam para quadros existentes
    alvos = page.evaluate("Array.from(document.querySelectorAll('#glossario dd.ocor a')).map(a => a.getAttribute('href').slice(1))")
    ruins = sorted({a for a in alvos if a not in ids})
    rel.reg(f"links de ocorrência do glossário apontam para quadros existentes ({len(alvos)} links)", not ruins, str(ruins))
    # foco visível
    ctx.close()
    ctx, page = abrir(browser, url, 1280, 720, erros)
    vistos = []
    for _ in range(6):
        page.keyboard.press("Tab")
        info = page.evaluate("""(() => { const a = document.activeElement; if (!a || a === document.body) return null;
          const e = getComputedStyle(a); return {id: a.id || a.tagName.toLowerCase(), visivel: a.matches(':focus-visible'),
          estilo: e.outlineStyle, largura: parseFloat(e.outlineWidth), cor: e.outlineColor}; })()""")
        if info:
            vistos.append(info)
    ok = bool(vistos) and all(v["visivel"] and v["estilo"] != "none" and v["largura"] > 0 for v in vistos)
    rel.reg("foco de teclado visível (contorno) nos controles percorridos por Tab", ok, str(vistos))
    ctx.close()


# ------------------------------------------------------------------ impressão

def paginas_do_pdf(caminho):
    """(número de páginas, [(largura, altura)], [texto de cada página] ou None)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        dados = Path(caminho).read_bytes()
        n = len(re.findall(rb"/Type\s*/Page(?![a-z])", dados))
        return n, [], None
    leitor = PdfReader(str(caminho))
    paginas = leitor.pages
    return len(paginas), [(float(p.mediabox.width), float(p.mediabox.height)) for p in paginas], [p.extract_text() or "" for p in paginas]


def checar_impressao(browser, url, saida, rel, erros):
    ctx, page = abrir(browser, url, 1280, 720, erros)
    ids = ids_dos_quadros(page)
    page.emulate_media(media="print")
    medidas_print = [page.evaluate("(id) => window.__medir(id)", qid) for qid in ids]
    com_falha = [m for m in medidas_print if m["falhas"]]
    rel.reg("impressão (mídia print): nenhum quadro com estouro ou sobreposição, lembrete inteiro", not com_falha,
            "; ".join(f"{m['id']}: {m['falhas']}" for m in com_falha))
    caminho = saida / "deck.pdf"
    page.pdf(path=str(caminho), width="13.333in", height="7.5in", print_background=True, prefer_css_page_size=True)
    n, tamanhos, textos = paginas_do_pdf(caminho)
    glossario = n - len(ids)
    rel.reg(f"PDF: {len(ids)} páginas de quadro mais páginas do glossário (total {n}, glossário {glossario})", glossario >= 1, str(caminho))
    if tamanhos:
        razoes = {round(w / h, 3) for w, h in tamanhos}
        rel.reg(f"PDF: todas as páginas em 16:9 (razões {sorted(razoes)}; primeira página {tamanhos[0]})", razoes == {round(16 / 9, 3)} or all(abs(r - 16 / 9) < 0.01 for r in razoes))
    if textos is not None:
        faltam = [qid for i, qid in enumerate(ids) if i >= len(textos) or f"{qid} · {i + 1} / {len(ids)}".replace(" ", "") not in textos[i].replace(" ", "").replace("\u00a0", "")]
        rel.reg("PDF: a página i traz o contador do quadro i (um quadro por página, nenhuma em branco entre eles)", not faltam, str(faltam[:5]))
        brancas = [i + 1 for i, t in enumerate(textos) if not t.strip()]
        rel.reg("PDF: nenhuma página em branco", not brancas, str(brancas))
    else:
        rel.reg("PDF: texto das páginas não verificado (pypdf ausente)", True, "apenas contagem")
    ctx.close()


# ------------------------------------------------------------------ principal

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", type=Path, default=DECK, help="deck HTML (padrão: text/slides/defesa.html)")
    ap.add_argument("--saida", type=Path, default=Path(tempfile.gettempdir()) / "Claude" / "deck_cdp",
                    help="diretório de capturas, relatório e PDF")
    ap.add_argument("--viewport", action="append", metavar="LxA", help="viewport (repetível; padrão: %s)" % ", ".join(VIEWPORTS))
    ap.add_argument("--calibrar", action="store_true", help="mede cada degrau de letra do lembrete em cada quadro")
    ap.add_argument("--sem-interacao", action="store_true", help="pula navegação, glossário e impressão")
    args = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    viewports = args.viewport or list(VIEWPORTS)
    saida = Path(args.saida)
    (saida / "capturas").mkdir(parents=True, exist_ok=True)
    url = Path(args.html).resolve().as_uri()
    rel, erros = Relatorio(), []
    medidas_por_viewport = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        try:
            print("Chrome", browser.version)
            if args.calibrar:
                cal = calibrar(browser, url, viewports, Path(args.html).read_bytes().decode("utf-8"), erros)
                tab = tabela_pronta(cal)
                (saida / "ajustes.txt").write_text(tab + "\n", encoding="utf-8")
                print("\nCalibração (pior caso entre %s): quadro, itens, maior n que cabe por degrau (normal/compacto/mini)" % ", ".join(viewports))
                for qid, c in sorted(cal.items()):
                    a = c["atual"]
                    print(f"{qid} k={c['k']:2d} | n: {a['normal']}/{a['compacto']}/{a['mini']} | rec {c['rec']}" + (f" | DESCER DENSIDADE para {c['descer']['para']}: {c['descer']['rec']}" if c["descer"] else ""))
                print("\n" + tab + f"\n(gravada em {saida / 'ajustes.txt'})")
                return 0
            for i, vp in enumerate(viewports):
                vw, vh = (int(x) for x in vp.split("x"))
                medidas_por_viewport[vp] = medidas(browser, url, vw, vh, saida, i == 0, rel, erros)
            cal = calibrar(browser, url, viewports, Path(args.html).read_bytes().decode("utf-8"), erros)
            checar_otimalidade(cal, rel)
            totais = {q: c["k"] for q, c in cal.items()}
            if not args.sem_interacao:
                checar_navegacao(browser, url, rel, erros)
                checar_glossario(browser, url, rel, erros)
                checar_impressao(browser, url, saida, rel, erros)
        finally:
            browser.close()
    rel.reg("console e página: nenhuma mensagem de erro, exceção ou requisição falha em toda a sessão", not erros, "; ".join(erros[:5]))
    lembretes = {}
    for vp, ms in medidas_por_viewport.items():
        lembretes[vp] = {m["id"]: m["lembrete"] for m in ms if m["lembrete"]}
    primeiro = lembretes.get(viewports[0], {})
    cortados = sorted(q for q, l in primeiro.items() if l["cortado"])
    print(f"\nQuadros com lembrete cortado ({viewports[0]}): {len(cortados)} {cortados}")
    print(f"Lembrete por quadro a {viewports[0]} (degrau; px; linhas; itens mostrados/total):")
    for q, l in primeiro.items():
        print(f"  {q} {l['classe'].replace('lembrete', '').strip() or 'normal'} {l['px']}px {l['linhas']}L {l['mostrados']}/{totais.get(q, '?')}")
    (saida / "relatorio.json").write_text(json.dumps(
        {"html": str(args.html), "viewports": viewports, "checagens": rel.checagens, "lembretes": lembretes,
         "quadros_com_lembrete_cortado": cortados, "erros_de_console": erros}, ensure_ascii=False, indent=1), encoding="utf-8")
    falhas = [c for c in rel.checagens if not c["ok"]]
    print(f"\n{len(rel.checagens) - len(falhas)} checagens ok, {len(falhas)} falhas; relatório em {saida / 'relatorio.json'}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
