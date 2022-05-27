import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load

st.set_page_config(page_title="Predição de Aluguel", page_icon="images/icone.png")

dados_brutos = pd.read_csv("data/raw/dados_brutos.csv")
dados_tratados = pd.read_csv("data/processed/dados_tratados.csv")

st.sidebar.markdown(
    """<p style="text-align: center;">
	<img src="https://i.ibb.co/bvV4kxg/u-http-icons-iconarchive-com-icons-paomedia-small-n-flat-1024-house-icon.png" alt="u-http-icons-iconarchive-com-icons-paomedia-small-n-flat-1024-house-icon" alt="Aluguel" width="150" height="150">
	</p>
	""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """<p style="font-family:Cambria; font-size: 30px; text-align: center">
	Previsão de Aluguel 
	<hr>
	</p>""",
    unsafe_allow_html=True,
)

pagina = st.sidebar.radio(
    label="",
    options=[
        "Apresentação",
        "Pré-Processamento",
        "Análise Descritiva",
        "Modelo de Predição",
    ],
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown(
    """<p style="text-align: center;">
	<a href="https://github.com/joao-vitor-souza/Prever-Aluguel">
	<img src="https://i.ibb.co/PYvPb4r/imagem-menu.png" alt="Github" width="200" height="75">
	</a>
	</p>
	""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """<p style="text-align: center;">
	Made with <a style="text-decoration: none" href='http://streamlit.io/'>Streamlit<a>
	</p>
	""",
    unsafe_allow_html=True,
)

if pagina == "Apresentação":
    "---"

    st.header("Problema de Negócio")

    texto_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Nesse projeto faremos uma Análise Descritiva dos aluguéis de casas e apartamentos nas cidades de São Paulo, Campinas, Rio de Janeiro, Belo
 	Horizonte e Porto Alegre, e ao final implementaremos um modelo de regressão capaz de prever o aluguel de certo imóvel. Isso auxiliará a 
  	estabelecer valores mais condizentes com a oferta e demanda do mercado imobiliário. A ideia é usar o valor do IPTU, do condomínio e as 
   	características físicas do lugar como a localização (cidade), quantidade de banheiros, quartos, vagas de estacionamento e etc., para prever 
    o seu aluguel.
	</p>"""

    texto_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Testaremos diversos modelos de regressão como 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Regressão Linear</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge">Regressão de Ridge</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic#sklearn.linear_model.LogisticRegression">Regressão Logística</a> e 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest#sklearn.ensemble.RandomForestRegressor">Floresta Aleatória</a> 
	com os <a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html?highlight=stratified#sklearn.model_selection.StratifiedShuffleSplit">dados estratificados</a> 
	por cidade. As métricas de desempenho usadas foram a Média do 
	<a style="text-decoration: none" href="https://pt.wikipedia.org/wiki/Coeficiente_de_determina%C3%A7%C3%A3o">Coeficiente de Determinação (R²)</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=rmse">Raiz do Erro Quadrático Médio (RMSE)</a> 
	e o <a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html?highlight=mean%20absolute#sklearn.metrics.mean_absolute_error">Erro Médio Absoluto (MAE)</a>. 
	O deploy do melhor modelo foi implementado nessa aplicação, mas para ver o tuning e o score de todos os modelos acesse este 
 	<a style="text-decoration: none" href='https://www.kaggle.com/joaovitorsilva/modelo-de-regress-o-para-prever-aluguel'>notebook<a>.
	<br><br>Os dados foram retirados de um banco de dados disponibilizado no Kaggle que pode ser baixado por 
	<a style="text-decoration: none" href="https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent">aqui<a>.
	</p>"""

    st.markdown(texto_1, unsafe_allow_html=True)

    # Imagem sem Copyright
    st.image("images/imagem.jpg")
    st.info(
        'Para pular diretamente para a aplicação, clique na opção "Modelo de Predição" no menu ao lado.'
    )
    st.markdown(texto_2, unsafe_allow_html=True)
    "---"

if pagina == "Pré-Processamento":
    "---"

    st.header("Pré-Processamento")

    texto_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Carregamos os dados para a variável <i>dados_brutos</i>, vamos dar uma olhada nos cinco primeiro elementos:
	</p>"""

    st.markdown(texto_1, unsafe_allow_html=True)

    with st.echo():
        st.dataframe(dados_brutos.head())

    st.subheader("Descrição")

    with st.echo():
        st.dataframe(dados_brutos.describe())

    st.subheader("Renomeação")

    texto_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Vamos renomear todas as colunas e campos do inglês para o português.
	</p>"""

    st.markdown(texto_2, unsafe_allow_html=True)

    st.code(
        """dados_brutos.rename({'city': 'Cidade', 
              'area': 'Área (m²)',
              'rooms': 'Quartos', 
              'bathroom': 'Banheiros', 
              'parking spaces': 'Vagas de Estacionamento', 
              'floor': 'Andar',  
              'animal': 'Animal', 
              'furniture': 'Mobília', 
              'hoa (R$)': 'Condomínio (R$)', 
              'rent amount (R$)': 'Aluguel (R$)', 
              'property tax (R$)': 'IPTU (R$)', 
              'fire insurance (R$)': 'Seguro de Incêndio (R$)', 
              'total (R$)': 'Total (R$)'}, axis='columns', inplace=True)

dados_brutos.replace({'acept': 'Aceita', 
               'not acept': 'Não Aceita', 
               'furnished': 'Mobiliado', 
               'not furnished': 'Não Mobiliado'}, inplace=True)"""
    )

    st.subheader("Removendo Valores Nulos e Duplicatas")

    st.code(
        """dados_brutos.dropna(inplace=True)
dados_brutos.drop_duplicates(inplace=True)"""
    )

    st.subheader("Outliers")

    texto_3 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Pelas informações da aba Descrição pode-se perceber que há uma grande disparidade de valores entre o 3º Quartil (75%) e os valores máximos 
 	em praticamente todas as colunas, isso indica que a maioria das características das instâncias são assimétricas à direita. Para remover 
  	outliers e ainda preservar pelo menos 90% dos dados usaremos a 
 	<a style="text-decoration: none" href="https://pt.wikipedia.org/wiki/Desigualdade_de_Chebyshev">Desigualdade de Chebyshev</a> que vale para 
  	qualquer distribuição. Segundo essa desigualdade pelo menos <i>x</i> porcentos dos dados estão dentro de <i>k</i> desvios, sendo <i>x</i> = 
   	1 - (1/<i>k²</i>). Logo para 3 desvios temos 1 - 1/9 = 8/9 ≅ 90%. Vamos definir uma função para remover os outliers a mais de 3 desvios e 
    aplicá-la aos nossos dados.
	</p>"""

    st.markdown(texto_3, unsafe_allow_html=True)

    st.code(
        """def outliers(dados, colunas): 

    indices_incomuns = []
    todos_indices = np.zeros(len(dados), dtype=bool)

    for coluna in colunas:

        dados_coluna = np.array(dados[coluna])
        
        desvio = dados_coluna.std()
        media = dados_coluna.mean()

        indice_incomum = dados_coluna > (media + 3 * desvio)

        indices_incomuns.append(indice_incomum)

    for i in range(len(indices_incomuns)):

        todos_indices = np.logical_or(todos_indices, indices_incomuns[i])

    return todos_indices

outliers_indices = outliers(dados, ['Área (m²)', 'Quartos', 'Banheiros', 
                                    'Vagas de Estacionamento', 'Condomínio (R$)', 
                                    'Aluguel (R$)', 'IPTU (R$)',
                                    'Seguro de Incêndio (R$)', 'Total (R$)'])

dados_tratados = dados_brutos[~outliers_indices]"""
    )

    texto_4 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Ao final, os <i>dados_tratados</i> ficaram da seguinte forma:
	</p>"""

    st.markdown(texto_4, unsafe_allow_html=True)

    with st.echo():
        st.dataframe(dados_tratados.head())
        st.dataframe(dados_tratados.describe())

    "---"

if pagina == "Análise Descritiva":
    "---"

    st.header("Análise Descritiva")

    st.subheader("Correlações")

    with st.echo():
        st.dataframe(dados_tratados.corr())

    texto_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Todas as colunas apresentam correlações positivas, isso confirma que quanto maiores são as grandezas espaciais como área, quantidade de 
 	quartos, banheiros e vagas de estacionamento, maior tende a ser as grandezas monetárias como aluguel, IPTU e seguro de incêndio.
	</p>"""

    st.markdown(texto_1, unsafe_allow_html=True)
    "---"
    st.subheader("Variações")

    with st.echo():
        desvio, desvio.name = (
            dados_tratados.groupby("Cidade")["Total (R$)"].std(),
            "Desvio (R$)",
        )
        media, media.name = (
            dados_tratados.groupby("Cidade")["Total (R$)"].mean(),
            "Média (R$)",
        )
        dados_var = pd.concat([media, desvio], axis="columns")
        dados_var["Coeficiente de Variação"] = (
            dados_var["Desvio (R$)"] / dados_var["Média (R$)"]
        )
        dados_var.sort_values(
            by="Coeficiente de Variação", inplace=True, ascending=False
        )

        st.dataframe(dados_var)

    texto_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Campinas é a cidade com mais variação das mensalidades e São Paulo é a com menos variação, embora esta última tenha a maior mensalidade 
 	média do conjunto de dados.	
	</p>"""

    st.markdown(texto_2, unsafe_allow_html=True)
    "---"
    st.subheader("Análise Gráfica")

    with st.echo():

        def barras(dados, x, y, hue, titulo):

            fig = px.bar(
                dados,
                x=y,
                y=x,
                color=hue,
                orientation="h",
                title=titulo,
                barmode="group",
            )
            fig.update_layout(font={"size": 18})

            return fig

    "---"
    st.markdown(
        "##### Média da mensalidade das propriedades que aceitam ou não animais"
    )

    with st.echo():
        grupo_animal_aluguel = (
            dados_tratados.groupby(["Cidade", "Animal"])["Total (R$)"]
            .mean()
            .reset_index()
        )

        fig = barras(
            grupo_animal_aluguel,
            "Cidade",
            "Total (R$)",
            "Animal",
            "Médias das mensalidades por cidade e se aceitam animais",
        )

        st.plotly_chart(fig, use_container_width=True)

    texto_3 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Em todas as cidades o aluguel é em média mais caro para aquelas propriedades que aceitam animais.
	</p>"""

    st.markdown(texto_3, unsafe_allow_html=True)
    "---"
    st.markdown(
        "##### Média da mensalidade das propriedades que aceitam ou não animais, por tipo de propriedade (casa ou apartamento)"
    )

    with st.echo():
        grupo_animal_propriedade = (
            dados_tratados.groupby(["Andar", "Animal"])["Total (R$)"]
            .mean()
            .reset_index()
        )
        casas = grupo_animal_propriedade[:2]
        casas.replace({"-": "Casa"}, inplace=True)
        casas.rename({"Andar": "Tipo"}, axis="columns", inplace=True)
        apartamentos = (
            grupo_animal_propriedade[2:]
            .groupby(["Animal"])["Total (R$)"]
            .mean()
            .reset_index()
        )
        apartamentos["Tipo"] = ["Apartamento"] * 2
        grupo_animal_propriedade = casas.append(apartamentos)

        fig = barras(
            grupo_animal_propriedade,
            "Tipo",
            "Total (R$)",
            "Animal",
            "Média de mensalidade por tipo de propriedade e se aceitam animais",
        )

        st.plotly_chart(fig, use_container_width=True)

    texto_4 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Apartamentos têm uma mensalidade média maior do que as casas em ambos os casos.
	</p>"""

    st.markdown(texto_4, unsafe_allow_html=True)
    "---"
    st.markdown(
        "##### Média da mensalidade das propriedades mobiliadas e não mobiliadas"
    )

    with st.echo():
        grupo_mobilia_aluguel = (
            dados_tratados.groupby(["Cidade", "Mobília"])["Total (R$)"]
            .mean()
            .reset_index()
        )

        fig = barras(
            grupo_mobilia_aluguel,
            "Cidade",
            "Total (R$)",
            "Mobília",
            "Médias das mensalidades por cidade e por mobília",
        )

        st.plotly_chart(fig, use_container_width=True)

    with st.echo():
        st.dataframe(
            dados_tratados.groupby("Cidade")["IPTU (R$)"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename({"IPTU (R$)": "IPTU Médio (R$)"}, axis="columns")
        )

    texto_5 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Em todas as cidades as propriedades mobiliadas são em média mais caras do que as não mobiliadas. Podemos ver também que a cidade de São Paulo 
	geralmente apresenta as maiores mensalidades e o maior IPTU na média.
	</p>"""

    st.markdown(texto_5, unsafe_allow_html=True)
    "---"
    st.markdown("##### Distribuição das 20% propriedades mais caras")
    with st.echo():

        def setor(dados, titulo):

            fig = px.pie(
                dados.reset_index(), values="Total (R$)", names="Cidade", title=titulo
            )
            fig.update_traces(pull=[0.1, 0, 0, 0, 0])
            fig.update_layout(font={"size": 18})

            return fig

    with st.echo():
        top_20 = dados_tratados.sort_values(by="Total (R$)")[
            -int(0.2 * dados_tratados.shape[0]) :
        ]
        grupo_top = top_20.groupby("Cidade")["Total (R$)"].count()
        grupo_top.sort_values(ascending=False, inplace=True)

        fig = setor(
            grupo_top, "20% das propriedades mais caras distribuídas por cidade"
        )

        st.plotly_chart(fig, use_container_width=True)

    texto_6 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	A alta concentração de propriedades caras em São Paulo acaba por elevar o aluguel e o imposto médio da cidade.
	</p>"""

    st.markdown(texto_6, unsafe_allow_html=True)
    "---"
    st.markdown("##### Distribuição das 20% propriedades mais baratas")

    with st.echo():
        sub_20 = dados_tratados.sort_values(by="Total (R$)")[
            : int(0.2 * dados_tratados.shape[0])
        ]
        grupo_sub = sub_20.groupby("Cidade")["Total (R$)"].count()
        grupo_sub.sort_values(ascending=False, inplace=True)

        fig = setor(
            grupo_sub, "20% das propriedades mais baratas distribuídas por cidade"
        )

        st.plotly_chart(fig, use_container_width=True)

    texto_7 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Diferente das propriedades mais caras que estão concentradas em São Paulo, as propriedades mais baratas estão bem mais distribuídas pelas 
 	cidades.
	</p>"""

    st.markdown(texto_7, unsafe_allow_html=True)
    "---"
    st.markdown("##### Mensalidade média por intervalos de andar")

    with st.echo():
        total_andares = dados_tratados.groupby("Andar")["Total (R$)"].mean()
        total_andares.drop(["301", "-"], inplace=True)
        total_andares.index = total_andares.index.values.astype(int)
        total_andares.sort_index(inplace=True)

        intervalos_andares = pd.cut(
            total_andares.index,
            [0, 10, 20, 30, 40, np.inf],
            labels=[
                "Até 9 andares",
                "Entre 10 e 20",
                "Entre 20 e 30",
                "Entre 30 e 40",
                "Mais de 40",
            ],
        )

        df = (
            pd.DataFrame(
                {"Total Andares": total_andares, "Intervalos": intervalos_andares}
            )
            .groupby("Intervalos")
            .agg(["mean", "std"])
        )

        fig = px.bar(
            x=[i for i in df.index],
            y=df["Total Andares"]["mean"],
            labels={"x": "Intervalo de Andares", "y": "Mensalidade Média (R$)"},
            title="Mensalidade média por intervalos de andar",
        )
        fig.update_layout(font={"size": 18})

        st.plotly_chart(fig, use_container_width=True)

    with st.echo():
        st.dataframe(df)

    with st.echo():
        st.write(
            total_andares.reset_index()["index"].corr(
                total_andares.reset_index()["Total (R$)"]
            )
        )

    texto_8 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Os andares entre 20 e 30, inclusive, são os mais caros na média. Os andares acima de 40 são os que mais variam de preço e há uma correlação
 	moderada (~0.55) entre o número do andar e sua mensalidade, isto é, quanto mais alto for o andar maior tende a ser a sua mensalidade.
	</p>"""

    st.markdown(texto_8, unsafe_allow_html=True)
    "---"

if pagina == "Modelo de Predição":
    "---"

    st.header("Modelo de Predição")

    texto_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	O modelo usado para predição que teve o melhor desempenho em teste foi o de 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest#sklearn.ensemble.RandomForestRegressor">Floresta Aleatória</a>. 
	O modelo base foi otimizado usando o framework <a style="text-decoration: none" href="https://optuna.org/">Optuna</a>. O número ideal de 
 	estimadores, que nesse caso são 
  	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier">Árvores de Decisão</a>, 
	é de 115, e a profundidade máxima de cada Árvore é 27. Definiu-se também que a Floresta Aleatória pode usar todos os núcleos do processador 
 	enquanto realiza as estimativas de aluguel. Essa Floresta apresentou o seguinte desempenho:
	</p>"""

    st.markdown(texto_1, unsafe_allow_html=True)

    metricas = ["RMSE (R$)", "MAE (R$)", "Correlação - R (%)", "Determinação - R² (%)"]
    valores = [1736.12, 1111.96, 77.86, 60.62]
    desempenho = pd.DataFrame(valores, index=metricas, columns=["Valor"])

    st.table(desempenho)

    texto_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	Todas as métricas foram calculadas sobre dados transformados por um pipeline. Caso queira ver o código do pipeline e os cálculos por trás dessas métricas clique 
	<a style="text-decoration: none" href='https://www.kaggle.com/joaovitorsilva/modelo-de-regress-o-para-prever-aluguel'>aqui<a>.
	</p>"""

    st.markdown(texto_2, unsafe_allow_html=True)

    st.info(
        "Poderíamos ainda implementar uma função para o cálculo de uma estimativa intervalar com certo nível de confiança em vez de uma estimativa pontual. Para isso usaríamos a biblioteca forestci (Forest Confidence Interval). Para mais detalhes: http://contrib.scikit-learn.org/forest-confidence-interval/index.html e https://pubmed.ncbi.nlm.nih.gov/25580094/"
    )
    "---"

    dados_tratados["Andar"].replace({"-": "0"}, inplace=True)
    dados_tratados = dados_tratados[dados_tratados["Andar"] != "301"]
    dados_tratados["Andar"] = dados_tratados["Andar"].astype(int)

    col_1, col_2 = st.columns(2)
    cidade = col_1.selectbox(
        "Cidade",
        ["São Paulo", "Porto Alegre", "Rio de Janeiro", "Campinas", "Belo Horizonte"],
        index=1,
    )
    area = col_1.number_input(
        "Área (m²)", value=100, min_value=1, max_value=dados_tratados["Área (m²)"].max()
    )
    quartos = col_1.number_input(
        "Quartos", value=1, min_value=0, max_value=dados_tratados.Quartos.max()
    )
    banheiro = col_1.number_input(
        "Banheiros", value=1, min_value=0, max_value=dados_tratados.Banheiros.max()
    )
    iptu = col_2.number_input(
        "IPTU (R$)", value=100, min_value=0, max_value=dados_tratados["IPTU (R$)"].max()
    )
    condominio = col_2.number_input(
        "Valor do Condomínio (R$)",
        value=0,
        min_value=0,
        max_value=dados_tratados["Condomínio (R$)"].max(),
    )
    estacionamento = col_2.number_input(
        "Vagas de Estacionamento",
        value=0,
        min_value=0,
        max_value=dados_tratados["Vagas de Estacionamento"].max(),
    )
    andar = col_2.number_input(
        "Andar",
        value=0,
        min_value=0,
        max_value=dados_tratados["Andar"].max(),
        help="Se for casa então mantenha zero!",
    )
    animal = col_1.radio("Aceita Animal?", options=["Aceita", "Não Aceita"])
    mobilia = col_2.radio(
        "É Mobiliado?", options=["Mobiliado", "Não Mobiliado"], index=1
    )
    desconto = st.number_input(
        "Desconto",
        min_value=0.0,
        help="Visando atrair clientes, você pode dar um desconto sobre o valor de mercado previsto. Valores entre 0 e 1 são considerados porcentagens, e maiores que 1 são valores em R$ propriamente ditos.",
        value=0.05,
    )

    modelo = load("models/modelo.joblib")

    instancia = pd.DataFrame(
        [
            [
                cidade,
                area,
                quartos,
                banheiro,
                estacionamento,
                andar,
                animal,
                mobilia,
                iptu,
                condominio,
            ]
        ]
    )
    instancia.columns = [
        "Cidade",
        "Área (m²)",
        "Quartos",
        "Banheiros",
        "Vagas de Estacionamento",
        "Andar",
        "Animal",
        "Mobília",
        "IPTU (R$)",
        "Condomínio (R$)",
    ]

    "---"
    predicao = np.round(modelo.predict(instancia)[0], 2)

    if (desconto >= 0) and (desconto <= 1):
        valor_final = np.round(predicao * (1 - desconto), 2)
        desconto_t, desconto = "Desconto (%)", 100 * desconto
    else:
        valor_final = np.round(predicao - desconto, 2)
        desconto_t = "Desconto (R$)"

    info = [
        "Valor de mercado previsto (R$)",
        desconto_t,
        "Valor final oferecido ao cliente (R$)",
    ]
    info_val = [str(predicao), str(desconto), str(valor_final)]
    resultado = pd.DataFrame(info_val, index=info, columns=["Valores"])

    st.table(resultado)
