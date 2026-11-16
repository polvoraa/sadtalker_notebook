# SadTalker Colab

Este pacote contém um notebook pronto para instalar e iniciar a interface Gradio do repositório oficial [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) em um runtime Google Colab com GPU T4.

## Arquivos

- `SadTalker_Colab.ipynb` — notebook para abrir no Colab e executar as células na ordem.
- `build_notebook.py` — fonte que gera o notebook dentro desta pasta.

## Uso rápido

1. Envie `SadTalker_Colab.ipynb` para o Google Drive ou abra-o diretamente no Google Colab.
2. No Colab, escolha **Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU**.
3. Execute as células do notebook de cima para baixo. A instalação e o download dos modelos levam alguns minutos.
4. Abra o link `.gradio.live` que aparece na célula de inicialização.
5. Envie a imagem e o áudio. O enquadramento `full`, o modo `Still` e batch size 1 são os padrões para manter a composição e reduzir o uso de memória.
6. Para monitorar a geração, clique em **Generate** na interface e execute a última célula do notebook.

## O que o notebook corrige

- Usa Python 3.10 em um ambiente virtual separado do Python padrão do Colab.
- Instala PyTorch com CUDA 12.1 e NumPy 1.23.5; não instala o PyAV opcional.
- Instala BasicSR sem build isolation e corrige o import removido `torchvision.transforms.functional_tensor`.
- Fixa FastAPI, Starlette, HTTPX e Pydantic v1 em versões compatíveis com Gradio 3.50.2, evitando erro HTTP 500 ao abrir a página.
- Define `MPLBACKEND=Agg` antes de importar Gradio/Matplotlib.
- Desativa o bloco opcional de Coqui TTS; o fluxo usa áudio enviado pelo usuário.
- Executa o downloader oficial dos modelos e verifica se os arquivos foram baixados.
- Inicia a interface Gradio com `share=True`.

O monitoramento apresenta uma porcentagem estimada, pois o SadTalker não expõe o progresso global real da inferência. O link `gradio.live` é público enquanto a sessão Colab estiver ativa; não use com mídias confidenciais.
