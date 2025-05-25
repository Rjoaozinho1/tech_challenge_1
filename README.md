# Projeto: Previsão de Custos de Seguro Saúde

Este projeto utiliza dados de seguros de saúde para treinar um modelo de machine learning capaz de prever o custo do plano de saúde de um paciente. O modelo é disponibilizado via uma API desenvolvida com FastAPI e pode ser executado em um container Docker.

## 1. Modelagem e Pipeline

- **Base de Dados:** `insurance.csv` (contém informações como idade, sexo, IMC, filhos, tabagismo e região)
- **Pré-processamento:**
  - As variáveis categóricas (`sex`, `smoker`, `region`) são convertidas para valores numéricos usando `LabelEncoder`.
  - Os dados são padronizados com `StandardScaler`.
- **Modelo:**
  - O modelo final é um `RandomForestRegressor` com os seguintes hiperparâmetros:
    - `random_state=0`, `oob_score=True`, `n_jobs=-1`, `n_estimators=100`, `min_samples_split=10`, `min_samples_leaf=4`, `max_features=1.0`, `max_depth=5`, `criterion='squared_error'`
  - O pipeline (`sklearn.pipeline.Pipeline`) encapsula o scaler e o modelo, permitindo que o mesmo arquivo (`pipeline_rf.joblib`) seja usado para predição sem necessidade de pré-processamento manual.
- **Exportação:**
  - O pipeline é salvo em `model/pipeline_rf.joblib`.

## 2. API (FastAPI)

- **Localização:** `app/main.py`
- **Framework:** FastAPI
- **Endpoint:**
  - `POST /predict`
  - Recebe um JSON com os seguintes campos:
    - `age` (int): Idade do paciente
    - `sex` (int): Sexo (0 = female, 1 = male)
    - `bmi` (float): Índice de Massa Corporal
    - `children` (int): Número de filhos
    - `smoker` (int): Fumante (0 = não fumante, 1 = fumante)
    - `region` (int): Região (0 = northeast, 1 = northwest, 2 = southeast, 3 = southwest)
  - Exemplo de requisição:
    ```json
    {
      "age": 35,
      "sex": 1,
      "bmi": 28.5,
      "children": 2,
      "smoker": 0,
      "region": 2
    }
    ```
  - Resposta:
    ```json
    { "charges": 12345.67 }
    ```
- **Carregamento do Modelo:**
  - O pipeline é carregado de `model/pipeline_rf.joblib`.
  - O endpoint faz a predição diretamente usando o pipeline.

## 3. Docker

- **Arquivo:** `Dockerfile`
- **Como funciona:**
  - Usa a imagem base `python:3.11-slim`.
  - Instala as dependências listadas em `requirements.txt`.
  - Copia a pasta `model/` (com o pipeline salvo) e a pasta `app/` (com o código da API).
  - Expõe a porta 8000.
  - Comando de inicialização: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Build e Execução

1. **Build da imagem:**
   ```sh
   docker build -t insurance-api .
   ```
2. **Execução do container:**
   ```sh
   docker run -p 8000:8000 insurance-api
   ```
3. **Acesso à documentação automática:**
   - Acesse `http://localhost:8000/docs` para testar a API via Swagger UI.

---

### Observações
- Certifique-se de que o arquivo `model/pipeline_rf.joblib` foi gerado antes de buildar a imagem Docker.
- O modelo espera que os dados estejam já codificados conforme o treinamento (veja os valores possíveis para cada campo).
- Para treinar novamente, utilize o notebook `Tech_Challange_1.ipynb`.