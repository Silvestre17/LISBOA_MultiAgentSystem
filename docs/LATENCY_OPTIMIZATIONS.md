# 🚀 Sistema de Otimização de Latência

Este documento resume as otimizações implementadas para minimizar a latência de resposta do Lisbon Urban Assistant.

## 📊 Resumo das Otimizações

### 1. Execução Paralela de Agentes ✅ (Já existia)
- Os agentes `weather`, `transport` e `researcher` já correm em paralelo usando `ContextThreadPoolExecutor`
- Localização: `agent/graph.py` linha 845
- Benefício: Reduz o tempo total quando múltiplos agentes são necessários

### 2. Execução Paralela de Tools (NOVO) ✅
- Quando um agente precisa chamar múltiplos tools, eles agora correm em paralelo
- Implementado em: 
  - `agent/agents/base.py` - método `execute_tools_parallel()`
  - `agent/agents/transport_agent.py` - usa execução paralela
  - `agent/agents/researcher_agent.py` - usa execução paralela
- Benefício: Até 4x mais rápido quando múltiplos tools são chamados na mesma iteração

### 3. HTTP Connection Pooling (NOVO) ✅
- Implementado singleton `HTTPSessionPool` em `agent/utils/optimization.py`
- Reutiliza conexões HTTP para evitar overhead de TCP/TLS handshake
- Configuração:
  - `pool_connections=10` - Número de pools de conexão
  - `pool_maxsize=20` - Máximo de conexões por pool
  - `max_retries=2` - Retentativas automáticas
- Benefício: Reduz latência de cada request HTTP em ~50-100ms

### 4. Caching com TTL (NOVO) ✅
- Três caches especializados em `agent/utils/optimization.py`:
  - `weather_cache` - TTL 5 minutos (dados meteorológicos)
  - `transport_cache` - TTL 1 minuto (dados de transporte real-time)
  - `static_cache` - TTL 1 hora (dados que não mudam)
- APIs atualizadas:
  - `tools/ipma_api.py` - usa `weather_cache`
  - `tools/metrolisboa_api.py` - usa `transport_cache`
- Benefício: Evita chamadas redundantes à API em queries consecutivas

### 5. Timeouts Otimizados (NOVO) ✅
- Azure OpenAI: `request_timeout=60`, `max_retries=2`
- APIs externas: Timeout de conexão 3s, read 10s
- Benefício: Falha rápida em vez de espera indefinida

## 📁 Ficheiros Modificados

| Ficheiro | Alteração |
|----------|-----------|
| `agent/utils/__init__.py` | Novo - Package de utilitários |
| `agent/utils/optimization.py` | Novo - Connection pooling, caching, parallel execution |
| `agent/agents/base.py` | Adicionado `execute_tools_parallel()` |
| `agent/agents/transport_agent.py` | Usa execução paralela de tools |
| `agent/agents/researcher_agent.py` | Usa execução paralela de tools |
| `agent/llm_factory.py` | Timeouts e max_retries para Azure |
| `tools/ipma_api.py` | Connection pooling + caching |
| `tools/metrolisboa_api.py` | Connection pooling + caching |

## 📈 Impacto Esperado

| Cenário | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| Query simples (single agent) | ~3-5s | ~2-4s | ~20-30% |
| Query complexa (multi-agent) | ~8-12s | ~4-6s | ~40-50% |
| Queries repetidas (cache hit) | ~3-5s | ~1-2s | ~60-70% |
| Múltiplos tool calls | Sequential | Parallel | ~50-75% |

## 🔧 Configuração

### Variáveis de Ambiente
Não são necessárias novas variáveis. As otimizações são ativadas automaticamente.

### Para Desativar Cache (Debug)
```python
from agent.utils.optimization import weather_cache, transport_cache
weather_cache.clear()
transport_cache.clear()
```

## 🧪 Testar Otimizações

```powershell
$env:PYTHONIOENCODING='utf-8'; python agent/utils/optimization.py
```

Saída esperada:
```
============================================================
🧪 Optimization Utilities Test
============================================================

📡 Testing HTTP Session Pool...
   Request 1: 200 (335ms)
   Request 2: 200 (360ms)
   Request 3: 200 (370ms)

💾 Testing TTL Cache...
   Cached value: value1
   After TTL: None

🔄 Testing Optimized JSON Fetch...
   First call: 4ms (network)
   Second call: 0ms (cached)

✅ All optimization utilities working!
```

## ⚠️ Considerações

1. **Cache Invalidation**: O cache usa TTL, não invalidação baseada em eventos. Para dados ultra-real-time (como posição de autocarro), considere reduzir o TTL.

2. **Memory Usage**: O cache é in-memory. Em produção com alto tráfego, considere usar Redis.

3. **Parallel Tool Execution**: Limitado a 4 workers por defeito. Ajustável em `execute_tools_parallel(max_workers=N)`.
