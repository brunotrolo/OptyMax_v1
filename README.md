# OptyMax — MVP corrigido (requisitos do usuário)

Este pacote contém a versão do MVP atualizada para cumprir os requisitos descritos pelo usuário.

## Principais cuidados
- Defina a variável de ambiente `OPLAB_TOKEN` com seu Access-Token. Ex:
  - `export OPLAB_TOKEN='SEU_TOKEN'`
- O app tenta buscar tickers em https://www.dadosdemercado.com.br/acoes e usar endpoints da OPLAB v3.
- Se a OPLAB não liberar a listagem da cadeia por ticker, o app usará um fallback sintético.
- Para calcular IV Rank e Beta, o app usa `yfinance`. Instale dependência e garanta acesso à internet.

## Rodar localmente
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPLAB_TOKEN='SEU_TOKEN'
streamlit run app.py
```
