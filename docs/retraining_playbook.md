# Playbook de Retreinamento

Este playbook documenta como o projeto deve evoluir quando o monitoramento indicar drift relevante ou degradacao da qualidade operacional do modelo.

## Quando considerar retreinamento

- drift moderado ou alto em features criticas
- aumento relevante da taxa prevista de churn
- perda de desempenho observada em lote rotulado futuro
- mudanca de negocio que altere o comportamento dos clientes

## Fluxo recomendado

1. Coletar um novo lote de dados validado.
2. Rodar as validacoes estruturais e de qualidade de entrada.
3. Comparar o novo lote com a base de referencia usando o monitoramento.
4. Confirmar se o alerta e pontual ou persistente.
5. Reexecutar o pipeline de treino com o novo conjunto.
6. Comparar o novo modelo com o modelo atual em:
   - ROC AUC
   - Average Precision
   - Recall
   - F1
   - estabilidade das probabilidades
7. Validar se o novo modelo melhora o ranking ou recupera degradacao.
8. Persistir uma nova versao do artefato se a troca for justificada.
9. Atualizar o registro de modelos no banco local.
10. Atualizar a base de referencia do monitoramento.

## Observacoes

- O retreinamento nao deve ser automatico apenas porque houve drift.
- O monitoramento e um gatilho de investigacao, nao uma ordem de deploy.
- Para portfolio, a clareza desse processo ja demonstra maturidade de MLOps, mesmo sem automatizacao completa.
