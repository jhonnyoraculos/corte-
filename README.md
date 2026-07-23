# CNC Image Vectorizer

Aplicativo Streamlit em português brasileiro para transformar uma imagem de peça
em contornos vetoriais calibrados em milímetros, revisá-los e exportá-los como DXF.
Também inclui a base experimental de um pós-processador MPR por template, sempre
dependente de um perfil e de um arquivo real de referência.

> Confira o arquivo no software oficial da máquina e faça uma simulação antes de
> qualquer operação real. O aplicativo não substitui a validação técnica do operador.

O aplicativo não se conecta, não envia programas e não controla uma CNC.

## Recursos do MVP

- Upload validado de PNG, JPG, JPEG e BMP, limitado a 20 MB.
- Avisos de baixa resolução e possível desfoque.
- Calibração obrigatória por largura conhecida ou por dois pontos.
- Tons de cinza, inversão, blur, remoção de ruído, limiar manual, Otsu,
  adaptativo, Canny e morfologia configurável.
- Extração com `cv2.findContours(..., RETR_TREE, ...)`, preservando níveis,
  contornos externos, furos e ilhas.
- Filtros por área/perímetro, maior peça, fechamento de gaps, simplificação e
  suavização opcional.
- Seleção, exclusão, inversão, fechamento, translação, rotação, espelhamento e
  mudança de origem.
- Offset interno/externo com Shapely para raio de ferramenta, kerf ou valor livre.
- Prévia Plotly realmente vetorial, em milímetros, com IDs, início, sentido,
  origem, caixa delimitadora e comparação da linha compensada com a original.
- Validação geométrica e contra limites configurados da máquina.
- DXF R12 e R2000 em memória, com camadas distintas.
- Importação e exportação do projeto em JSON, sem incorporar a imagem.
- Análise conservadora de uma referência MPR textual.
- Pós-processador MPR por placeholders, bloqueado quando a configuração é
  insuficiente ou a geometria tem erros.
- 29 testes automatizados, sem depender de imagens externas.

## Instalação no Windows

Requer Python 3.11 ou superior. No PowerShell ou Prompt de Comando, entre na pasta
do projeto e execute:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se o comando `python` abrir apenas a Microsoft Store ou não for encontrado,
instale uma versão atual em [python.org](https://www.python.org/downloads/) e
marque a opção para adicionar o Python ao `PATH`.

Depois de iniciar, o Streamlit mostra o endereço local, normalmente
`http://localhost:8501`.

Em Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Fluxo de uso

1. Envie uma imagem nítida, ortogonal e com bom contraste.
2. Informe uma medida real. O aplicativo nunca cria uma escala automaticamente.
3. Ajuste o limiar e os filtros observando as quatro imagens intermediárias.
4. Detecte os contornos e confira tipo, área, perímetro, pontos, hierarquia e
   sentido.
5. Selecione os contornos úteis e aplique somente as transformações necessárias.
6. Confira a prévia vetorial e as dimensões em milímetros.
7. Registre a ferramenta e a profundidade.
8. Resolva os erros de validação; confirme avisos antes de gerar DXF.
9. Abra o DXF ou MPR no software oficial da máquina e simule o trabalho.

### Calibração

No método **Largura conhecida**, o MVP considera a largura inteira da imagem como
o trecho medido. Recorte a imagem exatamente nos limites da peça ou de uma
referência confiável. A escala é:

```text
pixels_por_mm = largura_da_imagem_em_pixels / largura_real_em_mm
```

No método **Dois pontos**, informe manualmente X/Y dos dois pontos em pixels e a
distância real entre eles. As coordenadas da imagem partem do canto superior
esquerdo. Depois da vetorização, o sistema converte o eixo Y para o sistema
cartesiano.

Fotografias com perspectiva produzem escala variável e não são corrigidas neste
MVP. Para precisão dimensional, use captura ortogonal, scanner ou imagem técnica.

## DXF

Camadas geradas:

- `CUT_OUTER`: contornos externos e ilhas;
- `CUT_INNER`: furos;
- `REFERENCE`: origem, caixa delimitadora e metadados opcionais;
- `START_POINTS`: pontos iniciais opcionais.

O R2000 usa `LWPOLYLINE`; o R12 usa `POLYLINE` 2D. O fechamento é representado
pela flag própria da polilinha. `$INSUNITS=mm` é gravado no R2000. O DXF R12 não
possui suporte padronizado a `$INSUNITS`; por isso o aplicativo registra
`Unidade=mm` no texto de metadados e mantém todas as coordenadas numericamente em
milímetros.

O arquivo é montado em memória e relido nos testes com `ezdxf`.

## MPR: limitações e configuração

MPR varia por fabricante, modelo, controlador, versão do software, macros e
cadastro de ferramentas. O projeto não contém uma sintaxe MPR genérica nem um
perfil pronto para produção.

O arquivo [example_machine.json](machine_profiles/example_machine.json) é
intencionalmente incompleto. Valores desconhecidos continuam como `null`, e
`contour_block_template` fica vazio para impedir uma exportação acidental.

### Criar um perfil de máquina

1. Copie `machine_profiles/example_machine.json`.
2. Preencha somente dados confirmados na documentação ou no software oficial.
3. Não estime limites, capacidades ou convenções de direção.
4. Em `contour_block_template`, copie a estrutura de uma coordenada comprovada
   no arquivo real e substitua somente os valores X/Y pelos placeholders.
5. Carregue o JSON na aba MPR e revise os erros apresentados.

Exemplo conceitual, que **não representa a sintaxe de uma máquina real**:

```json
{
  "profile_name": "Perfil validado pelo operador",
  "format": "mpr",
  "units": "mm",
  "decimal_separator": ",",
  "coordinate_precision": 3,
  "supports_open_contours": false,
  "contour_block_template": "ESTRUTURA_REAL X={{X}} Y={{Y}}"
}
```

O bloco é repetido para cada ponto. Além de `{{X}}` e `{{Y}}`, o bloco pode usar:

- `{{POINT_INDEX}}`
- `{{CONTOUR_INDEX}}`
- `{{CONTOUR_ID}}`
- `{{IS_HOLE}}`
- `{{CLOSED}}`

### Fornecer uma referência MPR

Use um `.mpr` válido, criado pelo software original da máquina, preferencialmente
com uma peça retangular simples e um furo conhecido. Remova dados confidenciais
sem alterar a estrutura.

O analisador:

- limita o arquivo a 2 MB;
- rejeita conteúdo binário e caracteres de controle excessivos;
- tenta preservar UTF-8, Windows-1252 ou Latin-1;
- detecta CRLF/LF/CR, separador decimal e linhas repetidas;
- mostra o original somente para leitura;
- rejeita caminhos locais, de rede ou URLs;
- nunca executa macros ou interpreta comandos como código.

Na interface, faça uma cópia editável do programa de referência e substitua a
região que contém os contornos por `{{CONTOUR_BLOCKS}}`. O template de programa
também aceita:

- `{{PROJECT_NAME}}`
- `{{LENGTH}}` — extensão total em X;
- `{{WIDTH}}` — extensão total em Y;
- `{{THICKNESS}}`

Qualquer placeholder restante bloqueia a saída. O arquivo MPR só é liberado
quando perfil, referência, templates e geometria passam nas validações. Mesmo
assim, ele permanece marcado como experimental e deve ser comparado e simulado no
software oficial.

Se o sistema não puder demonstrar uma geração confiável, use o DXF e faça a
conversão no software do fabricante.

## Projeto JSON

O JSON usa `schema_version: "1.0"` e armazena:

- calibração e parâmetros de imagem;
- contornos em milímetros e hierarquia;
- linha original quando houver compensação;
- ferramenta, espessura e perfil;
- histórico de transformações;
- resultados da última validação;
- metadados e hash da imagem.

A imagem completa não é salva. Ao importar um projeto, a imagem da sessão só é
mantida se o hash corresponder; caso contrário, os vetores continuam disponíveis,
mas uma nova extração exige reenviar a imagem correta.

## Validações

As verificações incluem escala, dimensões, seleção, contornos abertos, pontos
duplicados, segmentos curtos, auto-interseções, cruzamentos entre contornos,
polígonos inválidos/vazios, coordenadas negativas, furo fora da peça,
compensação destrutiva, excesso de pontos, limites e área útil da máquina.

- **Erro:** bloqueia DXF e MPR.
- **Aviso:** permite DXF depois de confirmação visual.
- **Informação:** descreve o estado ou as dimensões.

O MPR possui verificações adicionais e sempre é bloqueado sem perfil, referência
e templates válidos.

## Testes

Com o ambiente ativado:

```bat
python -m pytest -q
```

A suíte cria em memória retângulo, círculo, peça com furo, peças separadas,
polilinha aberta e imagem com ruído. Ela cobre escala, geometria, hierarquia,
offsets, DXF R12/R2000, limites, segurança de upload e o pós-processador MPR
fictício usado somente em teste.

Para verificar apenas a inicialização da interface:

```bat
streamlit run app.py --server.headless true
```

## Estrutura

```text
cnc_image_vectorizer/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── src/
│   ├── models.py
│   ├── image_processing.py
│   ├── contour_processing.py
│   ├── geometry.py
│   ├── validators.py
│   ├── preview.py
│   ├── exporters/
│   │   ├── base_exporter.py
│   │   ├── dxf_exporter.py
│   │   ├── mpr_exporter.py
│   │   └── mpr_template.py
│   └── utils/files.py
├── machine_profiles/example_machine.json
├── samples/
└── tests/
```

## Decisões técnicas

- A representação principal usa `CNCContour` em milímetros e independe do raster.
- Pontos fechados não repetem o primeiro ponto; o fechamento é um atributo.
- A hierarquia OpenCV define furo por nível ímpar e externo/ilha por nível par.
- Operações são aplicadas a cópias antes de substituir o projeto; uma falha de
  buffer não deixa transformações parciais.
- O processamento puro é cacheado pelo Streamlit a partir dos bytes e parâmetros.
- Nenhum upload é gravado em caminho fornecido pelo usuário; o MVP não precisa de
  arquivos temporários para processar ou exportar.
- Não há `eval`, `exec`, chamada de shell com dados enviados, execução de upload
  ou transmissão automática.

## Checklist antes da CNC

1. Confirme uma dimensão conhecida com instrumento calibrado.
2. Confira escala, origem, espelhamento, sentido e unidades.
3. Compare contornos externos e furos com o desenho original.
4. Verifique diâmetro real da ferramenta, kerf e lado da compensação.
5. Abra o arquivo no software oficial e confirme os limites da máquina.
6. Execute a simulação completa, incluindo profundidades e fixação.
7. Faça um teste seguro em material de descarte quando apropriado.
