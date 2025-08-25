Act as a master student in the Master Degree Program in Data Science and Advanced Analytics, with a specialization in Data Science that will develop a thesis on "LLM-Powered Urban Exploration: A Framework for Adaptive Tourist and Mobility Route Planning"

# Proposta de Tese

## 🧠 Proposta de Tese de Mestrado

**Título (provisório):** *LLM-Powered Urban Exploration: A Framework for Adaptive Tourist Route Planning*

---

### 🎯 Objetivo

Desenvolver um agente inteligente baseado em LLM que crie **roteiros turísticos personalizados** por Lisboa ([AML](https://www.aml.pt/)) e **auxilie na mobilidade urbana**, combinando informação em tempo real sobre transporte, meteorologia, pontos de interesse, entre outras de relevância.

---

## **📂 Fontes de Dados (Gratuitas e Open Source)**

1. **Pontos Turísticos**
    - Monumentos, museus, exposições, atrações: dados de [Visitlisboa.com](https://www.visitlisboa.com/pt-pt/informacoes-viajante) e [Turismo de Lisboa](https://www.discoverportugal.info/reisefuehrer/wp-content/uploads//lisboacard_detail.pdf)
2. **Meteorologia**
    - Previsão IPMA (até 3 dias) - utilizada apenas em planeamento imediato ([API do IPMA](https://api.ipma.pt/))
3. **Transportes Públicos em Tempo Real**
    - Metro de Lisboa ([API Metro](https://api.metrolisboa.pt/store/apis/info?name=EstadoServicoML&version=1.0.1&provider=admin#tab1))
    - Carris Metropolitana ([API Metropolitana](https://github.com/carrismetropolitana/api?tab=readme-ov-file))
    - Comboios de Portugal ([Horários em Tempo Real CP](https://www.cp.pt/passageiros/pt/consultar-horarios))
4. **Serviços Essenciais**
    - Dados de farmácias, hospitais, entre outros serviços: [Lisboa Aberta CM](https://dados.cm-lisboa.pt/dataset?organization=camara-municipal-de-lisboa&res_format=GeoJSON)

---

### 🧰 Funcionalidades do Agente

- **Planeamento automático de rota personalizada**
    – Com base em preferências do utilizador, tempo disponível e condições (clima, tráfego dos transportes, horários).

- **Simulação de compra de bilhetes**
    – Transportes públicos e atrações turísticas (mesmo que apenas simulada).

- **Exportação para Calendário/Notion**
    – Gerar as tarefas com o itinerário no Google/Apple Calendar ou Notion.
    – Incluir notas breves (geradas por LLM) com história/contexto de cada ponto.

---

### ⚖️ Metodologia e Avaliação

- **Comparação entre o agente desenvolvido e o outros LLM’s (**ex.: ChatGPT/Grok/Gemini**)**
    – Identificar pontos onde os modelos sem contexto falham (p.ex. planeamento contextual com dados reais).
- **Testar múltiplos modelos LLM** (por exemplo: Llama, Mistral, outros open‑source ou utilizando APIs se disponível).
    – Avaliação de compreensão contextual e coerência das respostas.

- **Critérios de avaliação**:
  - Precisão e adequação das sugestões de itinerário e rotas de transportes
  - Usabilidade e experiência do utilizador
  - Integração e performance nas chamadas a APIs

As for text, always write in English, unless I tell you otherwise, and then write in well-written Portuguese without Brazilianisms (sempre que quiseres explicar alguma coisa e não seja o texto principal).

As for writing, you write in simple but academic English and follow the following rules:
- **Tone** - Be direct and helpful, without being overly optimistic or exuberant. Use an academic tone, but with simple English, and always keep in mind that the ultimate goal of this work is to be published in a scientific journal.
- **Concise and Clear**- Be concise and informative. Keep sentences and paragraphs short.
- **No AI Buzzwords**- Use clear, jargon-free language. Focus on clarity.
- **Active voice**- Write in a way that feels natural and engaging
- **No over explaining**- Just deliver the response, no unnecessary commentary.
- **Fresh research**- If available, pull in recent info from the web.
- **Criticism Allowed**- ChatGPT should push back when needed, not just agree
- **Prohibited Content**- Never use "-" in the text or other elements that AI generates.

Tem sempre em atenção que quero o melhor trabalho possível, por isso pensa bem nas tuas respostas e na forma como as estruturas.

O objetivo final deste primeiro "One Paper" é preencher este template do Word sendo que no máximo puderá ter 1 página o que irá equivaler a pouco mais do que 350 palavras:

Master Degree Program in Data Science and Advanced Analytics, with a specialization in Data Science
[Title/Topic]
André Filipe Gomes Silvestre & 20240502
Supervisor: Professor Bruno Jardim

Context:
[Paragraph #1: containing the context and importance of the topic; line spacing 1.15 points; font type; Times New Roman in font size [10 up to 12] for the body text.

Research gap and objectives:
[Paragraph #2: Containing a scientifically supported research gap, the research question(s), and the research objectives]

Methodological approach:
[Paragraph #3: Containing the scientifically supported methodological approach, namely the method(s), and instruments]

Expected results and contributions:
[Paragraph #4: Containing the expected results and expected contributions]



 
Bibliographical references:
[APA Style references, using a reference management system, e.g., Zotero, Mendeley, or Endnote]
________________________________________________________________________
Notes: Students should use one page, excluding references. References are mandatory.




JÁ DECIDI QUE VOU USAR ESTES ARTIGOS PARA O ONE PAPER 

    Final																
                                                                    
Date	Notas	 Year 	 Link 	 Paper Title 	 Authors 	 Abstract 	 Keywords 	 Methodology 	 Main Findings 	 Limitations 	 Future Work 	 Practical Applications 	 Tools Used 	 Models Tested 	 Evaluation Metrics 	 Data Sources 	 Implementation Details 
14-ago-2025		2025	https://link.springer.com/article/10.1007/s40558-025-00339-x	Lisa: a touristic chatbot for Lisbon	Miguel Cruz, Bruno Jardim, Miguel de Castro Neto	Proposes a methodology for developing generative touristic chatbots, demonstrated with a prototype for Lisbon using a web-scraped knowledge base and a RAG pipeline. The chatbot offers recommendations, information, and engages with visitors.	Chatbot, Transformer, Chatgpt, Tourism, Retrieval augmented generation	A two-stage evaluation process: 1) Automatic evaluation using synthetic datasets for Q&A and recommendations. 2) Manual evaluation by tourism experts assessing recommendation, information, and engagement.	The RAG-based chatbot achieved very strong results across all evaluated use cases, enhancing the tourism experience. The iterative optimization of the RAG pipeline and using ChatGPT-4o significantly improved performance.	The chatbot was not tested by actual tourists. ChatGPT-4o was only tested in two scenarios, so improvements may not be directly transferable from the ChatGPT-3.5 Turbo experiments.	Involve real users for evaluation. Expand optimization exploration with different models and data structures. Explore using LLMs as autonomous agents (e.g., for reservations) and integrate real-time data (strikes, weather).	A virtual concierge for tourists in Lisbon, providing recommendations, transit info, and engagement, enhancing visitor satisfaction and supporting destination management.	Chroma, Selenium, Streamlit, LlamaIndex	ChatGPT-3.5 Turbo, ChatGPT-4o. Retrieval algorithms: OpenAI embeddings, BM25, Hybrid Search.	Automatic: Semantic Answer Similarity (SAS), completion rate. Human: Likert scales on Recommendation, Information, Engagement, and Overall Experience.	Web-scraped data from Visit Lisboa, Agenda Cultural de Lisboa, Comboios de Portugal, Carris, Metro Lisboa, TripAdvisor, Reddit.	RAG pipeline on a knowledge base of over 2000 web pages. Used Chroma as vector DB and OpenAI's text-embedding-ada-002. Iterative prompt engineering was key.
2-ago-2025		2025	https://www.arxiv.org/pdf/2508.01432	TripTailor: A Real‑World Benchmark for Personalized Travel Planning	Kaimin Wang, Yuanzhe Shen, Changze Lv, Xiaoqing Zheng, Xuanjing Huang	Introduces TripTailor, a large-scale, real-world benchmark for personalized travel planning, to address limitations of existing benchmarks. Finds that <10% of LLM-generated itineraries achieve human-level performance.	Travel planning, Benchmark, LLM-based Agents, Personalization, Real-world data	Construction of a large-scale travel dataset (TripTailor) from real-world sources. An integrated evaluation framework assessing feasibility, rationality, and personalization using objective metrics, LLM-as-a-Judge, and a reward model.	Fewer than 10% of itineraries from SOTA LLMs achieve human-level performance. LLMs struggle with feasibility, rationality, and personalization. Merely satisfying constraints does not ensure high-quality plans.	Primarily focused on travel within China. Query generation uses LLMs, which may not fully reflect real-world user behavior. Evaluation methods depend on other LLMs, introducing potential bias.	Design more objective evaluation metrics. Train more robust evaluation models. Explore multi-turn dialogues for more authentic travel planning interactions.	A benchmark for evaluating and developing travel planning agents, driving progress towards agents that can generate practical, rational, and personalized itineraries.	Amap, TF-IDF	OpenAI GPT-4o, GPT-4o mini, DeepSeek-V3, Qwen2.5, OpenAI o1-mini. Planning methods: Direct, Zero-shot CoT, ReAct, Reflexion.	Feasibility Pass Rate, Rationality Pass Rate, Personalization Surpassing Rate, Average Route Distance Ratio, Final Surpassing Rate.	Data from open internet and online travel agencies for 40 Chinese cities.	Agents provided with pre-searched information. LLM evaluation uses DeepSeek-V3 and GPT-4o. Reward model fine-tunes Qwen2.5-1.5B-Instruct.
14-jul-2025		2025	https://arxiv.org/pdf/2507.10382v1	Leveraging RAG‑LLMs for Urban Mobility Simulation and Analysis	Yue Ding, Conor McCarthy, Kevin O’Shea, Mingming Liu	Presents a cloud-based, LLM-powered shared e-mobility platform with a mobile app. Introduces a RAG-based approach for querying structured traffic simulation and user travel data via natural language.	Smart Mobility, Shared E-Mobility, Traffic Simulation, Route Optimization, RAG, Cloud-Based Platform	A cloud-based platform using SUMO for simulation and Google Cloud Bigtable for storage. A two-stage Text-to-SQL framework using schema-aware RAG with M-Schema.	The schema-level RAG framework achieves high execution accuracy for Text-to-SQL (0.98 for user queries, 0.81 for system operators), making complex simulation data accessible.	Generated SQL is not always perfect and outputs should be used as supportive aids. The platform currently uses predefined traffic flows, not dynamic real-world data.	Extend the system for multi-objective optimization. Integrate public transportation networks. Enhance RAG with external knowledge. Move towards digital twin modeling.	A platform for simulating and analyzing shared e-mobility systems for urban planners, and a user app for personalized, energy-aware route recommendations.	SUMO, Google Cloud Platform, Docker, Android, NetworkX, PuLP, Chroma	XiYanSQL, GPT-4, GPT-4o, Gemini 1.5 Pro, Gemini 2.0 Flash. Embedding: all-MiniLM-L6-v2.	Execution Accuracy, Component Match (F1), BLEU Scores (1-4), ROUGE Scores (1, 2, L).	Dublin's SCATS system data, Transport Infrastructure Ireland data, 2022 Irish census data.	Multi-layered architecture (Cloud, Application, LLM). RAG module uses M-Schema to represent database schemas in a model-friendly JSON format for context.
1-jul-2025		2025	https://arxiv.org/pdf/2507.00914	Large Language Model Powered Intelligent Urban Agents: Concepts, Capabilities, and Applications	Jindong Han, Yansong Ning, Zirui Yuan, et al.	A comprehensive survey on Urban LLM Agents. It introduces their concept and unique capabilities, and reviews the research landscape from agent workflows to application domains like planning and transport.	Urban LLM Agents, Spatio-temporal reasoning, Urban intelligence, LLM-powered agents	A systematic literature survey, categorizing existing works by agent workflows (sensing, memory, reasoning, execution, learning) and application domains.	Urban LLM agents are a new class of intelligent systems for urban operations, but the field is nascent. They have core capabilities for spatio-temporal reasoning and collaboration but need significant development.	Agents lack sufficient domain-specific knowledge and reasoning. Large-scale, cross-regional collaboration is underexplored. Trustworthiness and evaluation are major open problems.	Build a full-stack ecosystem for urban LLM agents, focusing on multimodal fusion, rehearsal-based reasoning, toolchain building, self-evolution, and value alignment.	A conceptual framework and roadmap for researchers and developers working on AI solutions for smart cities, transportation, and urban planning.	N/A (Survey)	N/A (Survey)	Discusses existing evaluation approaches: agent-centric (utility, efficiency), task-specific, and spatio-temporal generalization.	N/A (Survey)	N/A (Survey)
23-abr-2025		2025	https://arxiv.org/pdf/2504.16505	TraveLLaMA: Facilitating Multimodal LLMs to Understand Urban Scenes and Provide Travel Assistance	Meng Chu, Yukang Chen, Haokun Gui, et al.	Introduces TraveLLaMA, a specialized multimodal language model for urban scene understanding and travel assistance, trained on a new 220k QA dataset to improve contextual travel recommendations.	Multimodal learning, Large language model, Urban scene understanding, Human-centered AI	Creation of a large-scale, multimodal dataset (TravelQA). Fine-tuning of state-of-the-art vision-language models (LLaVA, Qwen-VL, Shikra).	TraveLLaMA significantly outperforms general-purpose models in travel-specific tasks (6.5%-9.4% improvement). It excels at providing contextual recommendations and interpreting maps/images.	The model exhibits geographical biases, performing better on Western cities than Asian ones, likely due to training data imbalances.	Developing a more robust agent architecture for online assistance, handling multi-day planning, and integrating real-time information.	An AI travel assistant that can process both text and image queries to provide contextual responses, localization, and personalized recommendations.	Google Maps API, GPT-4	TraveLLaMA (fine-tuned LLaVA, Qwen-VL, Shikra). Baselines: BLIP-2, InstructBLIP, Shikra, Qwen-VL, LLaVA-1.5.	Accuracy on multiple-choice questions for Pure Text, VQA, and Full tasks. Qualitative analysis and System Usability Scale (SUS).	TravelQA dataset created from travel forums, review sites, tourism portals, and mapping services.	Dataset created using GPT-4 to generate QA pairs. Models were fine-tuned on this dataset. Agent architecture uses the ReAct framework.
2-nov-2024		2024	https://arxiv.org/abs/2403.09059	LAMP: A Language Model on the Map	Pasquale Balsebre, Weiming Huang, Gao Cong	Introduces LAMP, a framework to fine-tune a pre-trained LLM on city-specific data to provide accurate, conversational geospatial recommendations while minimizing hallucinations.	datasets, neural networks, gaze detection, text tagging, geospatial, POI retrieval	Fine-tuning a pre-trained LLM (LLaMa-2-7B-Chat) on a self-supervised task using an automatically generated dataset via a RAG-like approach.	The fine-tuned model (LAMP) shows significantly improved spatial awareness (92%) and truthfulness (86%) compared to baselines like ChatGPT and can handle complex conversational queries.	The model is specific to one city (Singapore). Performance depends heavily on the quality of the initial POI data collected.	Extending the framework to more cities and incorporating more diverse data sources. Exploring more complex conversational planning capabilities.	A city-specific conversational AI for POI retrieval, day planning, and other geospatial tasks, useful in tourism and local search.	Nominatim (OSM API)	LAMP (fine-tuned LLaMa-2-7B-Chat), ChatGPT 3.5/4o/4, Claude-2/3 Sonnet, LLaMa-2 (7B, 13B, 70B).	Human evaluation by GIS experts based on Truthfulness, Spatial Awareness, and Semantic Relatedness.	18,390 Points of Interest (POIs) from Yelp Singapore.	Used 4-bit quantization and LoRA for efficient fine-tuning. Training data included negative samples to reduce hallucinations.
26-set-2024		2024	https://arxiv.org/abs/2409.18003	Enhancing Tourism Recommender Systems for Sustainable City Trips Using Retrieval‑Augmented Generation	Ashmi Banerjee, Adithi Satish, Wolfgang Wörndl	Proposes a novel RAG-based approach to enhance Tourism Recommender Systems for sustainability by incorporating a sustainability metric (SAR) to balance user preferences with sustainability goals.	Tourism Recommender Systems, Sustainability, Retrieval-Augmented Generation, Large Language Models	A modified RAG pipeline with a "Sustainability Augmented Reranking" (SAR) metric. Evaluation using open-source LLMs on synthetic test queries.	The SAR-enhanced approach consistently matches or outperforms the baseline, effectively integrating sustainability into recommendations without compromising answer quality.	The sustainability metric is limited to popularity and seasonality. The evaluation uses synthetic test data generated by other LLMs.	Expand knowledge base with real-time data. Explore additional sustainability metrics (e.g., carbon footprint). Develop a conversational recommender to address cold-start problem.	A sustainable tourism recommender system that guides tourists towards less crowded options, helping to manage over-tourism.	Wikivoyage, LanceDB, sentence-transformers	Llama-3.1-Instruct-8B, Mistral-Instruct-7B. GPT-4, Gemini-1.5-Pro, GPT-4o-mini, Claude-3-5-sonnet used for data generation/evaluation.	Answer Relevance, Sustainability (Accuracy and Frequency), Faithfulness (% of out-of-context responses).	Wikivoyage, Tripadvisor API, whereandwhen.net.	SAR metric is a weighted combination of normalized popularity and seasonality. LLM instructed via a role-playing prompt to balance preferences and sustainability.
4-set-2024		2024	https://arxiv.org/pdf/2409.00494	GenAI-powered Multi-Agent Paradigm for Smart Urban Mobility: Opportunities and Challenges for Integrating Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) with Intelligent Transportation Systems	Haowen Xu, Jinghui Yuan, Anye Zhou, et al.	Proposes a conceptual framework for a GenAI-powered multi-agent system for smart urban mobility, integrating LLMs and RAG with Intelligent Transportation Systems (ITS).	Retrieval-Augmented Generation, Large Language Models, Intelligent Transportation System, Multi-agent System	Proposal of a conceptual framework for a multi-agent system. The architecture includes LLM-based, retrieval, and task-specific agents.	Integrating LLMs and RAG into a multi-agent system can overcome limitations of traditional ITS, leading to a more scalable, intuitive, and human-centric paradigm.	The work is conceptual and does not present an implemented system or empirical results.	Focus on refining the proposed technologies and expanding applications. Address challenges like task coordination, data sovereignty, and AI accountability.	The framework could build advanced ITS solutions for traffic management, public transit optimization, and personalized driver assistance.	N/A (Conceptual)	N/A (Conceptual)	N/A (Conceptual)	Discusses data from IoT infrastructure, Connected Vehicles (CVs), and Social Media.	Framework designed to be flexible, leveraging modern DevOps technologies and APIs for modular agent deployment.
24-jun-2024		2024	https://arxiv.org/abs/2407.11003	Using Large Language Models in Public Transit Systems: San Antonio as a Case Study	Ramya Jonnala, Gongbo Liang, Jeong Yang, Izzat Alsmadi	Examines the impact of LLMs on San Antonio's public transit system, using GTFS data to evaluate their potential in enhancing route planning and providing personalized assistance.	Large Language Models, GTFS, Public Transit Systems	A two-task evaluation: 1) "Understanding" task using a pre-trained LLM on MCQs. 2) "Information Retrieval" task providing the LLM with GTFS data before questioning.	LLMs have acquired significant transportation knowledge but show large performance gaps, suggesting imbalanced training data. GPT-4o outperforms GPT-3.5-Turbo in information retrieval.	The study is confined to a single city's public transit system (San Antonio) and its specific data format (GTFS).	Address performance gaps with more balanced datasets. Continued research is needed to address ethical concerns before real-world deployment.	Using LLMs to create more user-friendly and efficient public transportation networks, providing a model for other cities to follow.	OpenAI API	ChatGPT (GPT-3.5-Turbo and GPT-4o)	Accuracy on multiple-choice questions and a question-answer benchmark.	San Antonio VIA GTFS feed. A custom-designed set of 275 questions.	"Understanding" task used 195 original MCQs plus an augmented set. "GTFS Retrieval" benchmark used a QA format on a trimmed dataset.
4-mai-2024	Justificar o pq deste trabalho - ChatGPT só por si não consegue ter o potencial todo para itenerários 	2024	https://link.springer.com/content/pdf/10.1007/978-3-031-58839-6_38.pdf	ChatGPT as a Travel Itinerary Planner	Katerina Volchek, Stanislav Ivanov	Analyzes ChatGPT as an itinerary planner by comparing its generated itineraries for three destinations with those from tourism experts. It finds ChatGPT creates accessible but less accurate and specific itineraries.	ChatGPT, generative AI, itinerary, e-tourism, service automation	Comparative analysis of ChatGPT-4 generated itineraries versus human-expert itineraries for three destinations. Evaluation based on a conceptual framework of 11 quality criteria.	ChatGPT is a good starting point for travel inspiration but cannot be an exclusive planning tool, as it creates less useful itineraries that lack detail and real-world context compared to human experts.	Small sample size (three destinations). Did not deeply explore prompt engineering techniques to improve the quality of the generated itineraries.	Involve more destinations and validate itineraries with more travel experts. Research practical challenges for travel companies implementing generative AI.	ChatGPT can be used for drafting preliminary travel itineraries, which then require validation and further planning by the user.	ChatGPT-4	ChatGPT-4	A set of 11 quality criteria covering Intrinsic Quality/Accuracy, Contextual Quality, Representational Quality, and Accessibility.	Human-generated itineraries collected from reputable travel websites.	Used "3-day travel itinerary" + "destination" as search terms to find expert itineraries and then prompted ChatGPT-4 for the same destinations.













E A IDEIA É:

1. Context: Em 2024, o Turismo continou a ser um dos pilares da ecomonia Portuguese (Fonte: https://www.turismodeportugal.pt/pt/Turismo_Portugal/visao_geral/Paginas/default.aspx) com uma evolução positiva a rondar 8,8% nas receitas turísticas. Simultaneo a esse crescimento, seundo a Pordata (https://www.pordata.pt/pt/estatisticas/transportes/mobilidade/deslocacoes-em-transportes-publicos) as Deslocações em transportes públicos incrementou nos ultimos anos, sendo Lisboa área com maior diversidade e oferta de transportes. Neste sentido, é crucial explorar soluções viradas (MELHORA ESTA PALAVRA) para o cidadão de forma a otimizar a sua experiência, seja turista ou residente, recorrendo a tecnologias emergentes como os LLM's (Large Language Models) aliadas ao trafego de dados em tempo real dos transportes, condições atmosfericas e outros fatores diversos que podem influenciar na escolha do itinerário mais adequado

2. Research gap and objectives: Apesar do avanço significativo dos LLM's e disponibilização do ChatGPT, a sua aplicação no planeamento urbano e turístico ainda é limitada. [Aqui existe um dos artigos que já falta das lacunas do ChatGPT nos itenários - RESUME NUMA FRASE O QUE O ARTIGO MOSTRA]. Estudos recentes [aqui é todos os estudos que abordam a utilização de LLM's no contexto de transportes e turismo, destacando as suas potencialidades e limitações - póe apenas o nome dos LLMs e resume tudo ao máximo em grupo].

3. Methodological approach: A abordagem metodológica envolverá o desenvolvimento de um agente inteligente baseado em LLM's, integrando dados em tempo real de várias fontes, como transportes públicos (Metro de Lisboa, Carris Metropolitana, Comboios de Portugal), meteorologia (IPMA) e pontos turísticos (Visitlisboa.com, Turismo de Lisboa). Serão testados múltiplos modelos LLM's para avaliar a sua eficácia na criação de roteiros personalizados. A avaliação será realizada através de critérios como precisão das sugestões, usabilidade e performance na integração com APIs.

4. Expected results and contributions: Espera-se que o projeto contribua para a melhoria da experiência do utilizador em contextos urbanos e turísticos, através da personalização de itinerários e da otimização do uso dos transportes públicos. Os resultados poderão ser utilizados por entidades públicas e privadas para a promoção de soluções de mobilidade mais eficientes e sustentáveis.




Tem ainda como referência um OnePaper que tinha feito para outro tema que não este.

# Master Degree Program in Data Science and

# Advanced Analytics, with a specialization in Data

# Science


![](https://web-api.textin.com/ocr_image/external/4c3d338af2c46795.jpg)

LLM-based Agentic Framework for Analytics and Data 

Visualization (Oeste CIM)

André Filipe Gomes Silvestre & 20240502 

Supervisor: Professor Bruno Jardim

# Context: 

The West region of Portugal generates diverse datasets, from Waze traffic and E‑REDES energy consumption logs to SIBS ATM transactions and NOS mobility records. Combining these heterogeneous sources could improve urban planning, resource allocation, and economic policy, but presents challenges in fusion and interpretation. While deep-learning methods that merge spatio-temporal, visual, and text data reveal new smart-city patterns (Zou et al., 2024) and AI-driven predictive models support sustainable urban development (Cina et al., 2025), a tool is needed to translate natural language prompts into analytical workflows for decision support. 

# Research Gap and Objectives:

Recent work such as MatPlotAgent (Yang et al., 2024), PlotGen (Goswami et al., 2025), nvAgent (Ouyang et al., 2025), and IDM-GPT (Yang et al., 2025) show how multi-agent LLMs can automate charting and analysis, but they’ve only been tested on narrow or synthetic datasets. This research addresses this gap by designing and evaluating a framework purpose-built for real-world, heterogeneous regional data. The primary objective is to develop a system that not only generates accurate visualizations but also supports complex, cross-domain analytical queries posed in natural language. 

# Methodological Approach:

To structure development, this research will follow the CRISP-DM methodology (Provost & Fawcett, 2013). In the Business and Data Understanding phases involve the inventory and profile each dataset. Data Preparation will includes cleaning and schema alignment via SQL. In the Modelling phase, a multi-agent architecture will be designed using LangChain, incorporating prompt engineering and fine-tuning LLMs to handle complex queries. Visualization agents will generate Matplotlib and Seaborn charts, with an iterative feedback loop for refinement. The final solution will be deployed as a Streamlit application, offering a web interface where users submit natural language requests and customize visual styles, ensuring consistency in colour palettes and typography across municipalities or organizations. 

## Expected Results and Contributions:

The expected outcome is an application enabling users to generate visualizations from multiple data sources using conversational prompts. Contributions include: (1) a novel LLM-based agentic framework optimized for regional economic and mobility data; (2) an open-source, end-to-end tool bridging natural language with data visualization; (3) empirical insights into regional patterns of traffic,energy consumption, financial transactions, and population flows in the West region of Portugal; and (4) a scientific article detailing the framework and its findings. 

# Bibliographical References:

Cina, E., Elbasi, E., Elmazi, G., & AlArnaout, Z. (2025). The Role of AI in Predictive Modelling for Sustainable Urban Development: Challenges and Opportunities. Sustainability, 17(11), 5148. https://doi.org/10.3390/su17115148 

Goswami, K., Mathur, P., Rossi, R., & Dernoncourt, F. (2025). PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback. ArXiv.org. https://arxiv.org/abs/2502.00988 

Ouyang, G., Chen, J., Nie, Z., Gui, Y., Wan, Y., Zhang, H., & Chen, D. (2025). nvAgent: Automated Data Visualization from Natural Language via Collaborative Agent Workflow. ArXiv.org. https://arxiv.org/abs/2502.05036 

Provost, F., & Fawcett, T. (2013). Data Science for Business: What You Need to Know About Data Mining and Data-Analytic Thinking. O’Reilly. 

Yang, F., Liu, X. C., Lu, L., Wang, B., & Liu, C. D. (2025). Independent Mobility GPT (IDM-GPT): A Self-Supervised Multi-Agent Large Language Model Framework for Customized Traffic Mobility Analysis Using Machine Learning Models. ArXiv.org. http://arxiv.org/abs/2502.18652 

Yang, Z., Zhou, Z., Wang, S., Cong, X., Han, X., Yan, Y., Liu, Z., Tan, Z., Liu, P., Yu, D., Liu, Z., Shi,X., & Sun, M. (2024). MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization. ArXiv.org. https://arxiv.org/abs/2402.11453 

Zou, X., Yan, Y., Hao, X., Hu, Y., Wen, H., Liu, E., Zhang, J., Li, Y., Li, T., Zheng, Y., & Liang, Y.(2024). Deep Learning for Cross-Domain Data Fusion in Urban Computing: Taxonomy,Advances, and Outlook. ArXiv.org. https://doi.org/10.1016/j.inffus.2024.102606. 



