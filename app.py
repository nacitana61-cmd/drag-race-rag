import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ============================================================
# DOCUMENTS - RuPaul's Drag Race Knowledge Base
# ============================================================

DOCUMENTS = [
    """RuPaul's Drag Race is an American reality competition television series. 
    It was created by and stars RuPaul, who searches for "America's next drag superstar." 
    The show first aired on Logo TV in 2009 and later moved to VH1 in 2017. 
    It has become one of the most critically acclaimed reality shows in television history, 
    winning numerous Emmy Awards over the years.""",

    """The competition format of RuPaul's Drag Race follows a group of drag queens 
    competing in various challenges each episode. There is typically a mini challenge 
    and a main challenge. The bottom two contestants must lip sync for their lives, 
    and RuPaul decides who stays and who gets eliminated with the famous phrase 
    'Sashay away.' The winner of each episode receives prizes and advantages.""",

    """RuPaul Charles is the host, judge, and executive producer of Drag Race. 
    Born on November 17, 1960, RuPaul is one of the most successful drag queens 
    in history. He is also a singer, known for the iconic song 'Supermodel (You Better Work)' 
    released in 1993. RuPaul has used the show as a platform to promote self-expression, 
    LGBTQ+ visibility, and the message that 'We're all born naked and the rest is drag.'""",

    """Season 5 of RuPaul's Drag Race is widely considered one of the best seasons ever. 
    It featured iconic queens such as Alaska, Roxxxy Andrews, Detox, and Jinkx Monsoon, 
    who ultimately won the season. The season is remembered for its strong cast, 
    memorable lip syncs, and the powerful episode where queens read their own letters 
    from home. Jinkx Monsoon became a fan favorite for her unique vintage style 
    and theatrical performances.""",

    """The Snatch Game is one of the most beloved recurring challenges in RuPaul's Drag Race. 
    It is a parody of the classic game show 'Match Game' where contestants impersonate 
    celebrities. A great Snatch Game performance requires comedic timing, knowledge of 
    the celebrity being portrayed, and the ability to improvise. Legendary Snatch Game 
    performances include BenDeLaCreme as Maggie Smith, Bianca Del Rio as Judge Judy, 
    and Alaska as Mae West.""",

    """Drag terminology is a colorful and rich vocabulary used within the drag community. 
    'Serving looks' means presenting a stunning visual appearance. 'Reading' means 
    playfully insulting someone in a witty way. 'Shade' is a more subtle form of insult. 
    'Tucking' refers to the technique of hiding male genitalia for a feminine silhouette. 
    'Gagging' means being amazed or shocked by something. These terms have roots in 
    ballroom culture and LGBTQ+ communities dating back decades.""",

    """RuPaul's Drag Race has inspired numerous international versions around the world. 
    These include Canada's Drag Race, Drag Race UK, Drag Race Holland, Drag Race España, 
    Drag Race France, Drag Race Italia, and many more. Each version celebrates the local 
    drag culture while maintaining the core format of the original show. Drag Race UK 
    is particularly popular and has produced beloved queens like Lawrence Chaney 
    and The Vivienne.""",

    """All Stars is a spin-off series of RuPaul's Drag Race where former contestants 
    return to compete again for a spot in the Drag Race Hall of Fame. The format differs 
    slightly from the main show — in earlier seasons, eliminated queens chose who to 
    send home rather than RuPaul. All Stars has produced some of the most dramatic 
    and memorable moments in Drag Race herstory, including Alaska's iconic win in 
    All Stars 2 and Trixie Mattel's win in All Stars 3.""",

    """The fashion and beauty aspects of RuPaul's Drag Race are central to the competition. 
    Queens are judged on their runway looks each episode, which must align with a specific 
    theme. Judges look for creativity, construction, and the ability to transform. 
    Some queens are known for making their own costumes, like Nina Bonina Brown and 
    Shea Couleé. The main stage runway has featured everything from avant-garde high 
    fashion to camp comedy looks.""",

    """Lip syncing is one of the most iconic elements of RuPaul's Drag Race. When two 
    queens are in the bottom, they must perform a lip sync to a pre-selected song, 
    usually a pop anthem or dance track. Memorable lip syncs include Tatiana and 
    Shangela's performance to 'Telephone' by Lady Gaga, and Roxxxy Andrews surprising 
    everyone by revealing a second outfit mid-performance. The phrase 'Lip sync for 
    your life' has become a pop culture catchphrase.""",

    """The judges panel of RuPaul's Drag Race has featured many notable personalities 
    over the years. The permanent judges include RuPaul, Michelle Visage, Ross Mathews, 
    and Carson Kressley. Michelle Visage is known for her sharp critiques and passion 
    for drag. Guest judges have included superstars like Lady Gaga, Nicki Minaj, 
    Christina Aguilera, and many more. The judging panel evaluates queens on 
    charisma, uniqueness, nerve, and talent — the four qualities that spell C.U.N.T.""",

    """Trixie Mattel and Katya are two of the most beloved queens to come out of 
    RuPaul's Drag Race. Both appeared on Season 7 and All Stars 3, where Trixie won. 
    They are known for their deep friendship and comedic chemistry, which led to 
    their own web series and TV show called 'UNHhhh' and 'The Trixie & Katya Show.' 
    Trixie is known for her Barbie-inspired aesthetic and country music, while Katya 
    is known for her Russian character and surreal humor.""",
]

# ============================================================
# CUSTOM STYLING
# ============================================================

def apply_custom_styling():
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #1a0033 0%, #2d0057 50%, #1a0033 100%);
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2d0057 0%, #4a0080 100%);
        }
        
        /* Sidebar text */
        [data-testid="stSidebar"] * {
            color: #ffb3ff !important;
        }

        /* All general text */
        .stApp, .stMarkdown, p, li {
            color: #f0e6ff !important;
        }

        /* Headers */
        h1 {
            color: #ff69b4 !important;
            text-shadow: 0 0 20px #ff69b4;
            font-size: 2.5em !important;
        }
        h2, h3 {
            color: #da70d6 !important;
        }

        /* Search input box */
        .stTextInput input {
            background-color: #2d0057 !important;
            color: #ffffff !important;
            border: 2px solid #ff69b4 !important;
            border-radius: 10px !important;
        }

        /* Buttons and expanders */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #4a0080, #800080) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }

        /* Success/info boxes */
        .stAlert {
            background-color: #2d0057 !important;
            border: 1px solid #ff69b4 !important;
            color: #f0e6ff !important;
        }

        /* Radio buttons */
        .stRadio label {
            color: #ffb3ff !important;
        }

        /* Slider */
        .stSlider {
            color: #ff69b4 !important;
        }

        /* Expander content */
        .streamlit-expanderContent {
            background-color: #1a0033 !important;
            border: 1px solid #800080 !important;
            color: #f0e6ff !important;
            border-radius: 0 0 8px 8px !important;
        }

        /* Selectbox */
        .stSelectbox select {
            background-color: #2d0057 !important;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================
# APP SETUP
# ============================================================

st.set_page_config(
    page_title="Drag Race Knowledge Base",
    page_icon="👑",
    layout="wide"
)

apply_custom_styling()

@st.cache_resource
def setup_vectorstore(chunk_size, chunk_overlap):
    """Creates a searchable vector database from our documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.create_documents(DOCUMENTS)
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)

# ============================================================
# PAGES
# ============================================================

def home_page():
    st.title("👑 RuPaul's Drag Race Knowledge Base")
    st.subheader("Your ultimate guide to the world of drag!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        Welcome to the **Drag Race Knowledge Base** — an AI-powered semantic search engine 
        built around the iconic TV show RuPaul's Drag Race! 🏳️‍🌈
        
        ### What can you find here?
        - 🎭 Information about queens, seasons, and challenges
        - 💄 Drag terminology and ballroom culture
        - 👗 Fashion, runway looks, and iconic moments
        - 🌍 International versions of the show
        - 💋 Lip syncs, judging panel, and All Stars
        """)

    with col2:
        st.markdown("""
        ### How to use this app
        Head over to the **🔍 Search** page using the sidebar,
        type any question about Drag Race, and get instant answers!
        
        ### Example questions to try:
        - *"Who won Season 5?"*
        - *"What is the Snatch Game?"*
        - *"Tell me about lip syncing"*
        - *"What does reading mean in drag?"*
        - *"Who are the judges?"*
        - *"Tell me about All Stars"*
        """)

    st.markdown("---")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("📄 Documents", "12")
    with col4:
        st.metric("🧠 Embedding Model", "all-MiniLM-L6-v2")
    with col5:
        st.metric("🔍 Search Type", "Semantic")

    st.info("💡 This app uses AI-powered semantic search — it understands the *meaning* of your question, not just keywords!")

def search_page():
    st.title("🔍 Search the Drag Race Knowledge Base")

    st.sidebar.markdown("### ⚙️ Chunking Settings")
    chunk_size = st.sidebar.selectbox(
        "Chunk Size",
        options=[150, 300, 500],
        index=1,
        help="How large each text chunk is. Smaller = more precise, Larger = more context."
    )
    chunk_overlap = st.sidebar.selectbox(
        "Chunk Overlap",
        options=[20, 50, 100],
        index=1,
        help="How much chunks overlap. More overlap = less chance of cutting off important info."
    )

    with st.spinner("⏳ Loading knowledge base..."):
        vectorstore, num_chunks = setup_vectorstore(chunk_size, chunk_overlap)

    st.success(f"✅ Knowledge base ready! Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap} → {num_chunks} total chunks created.")

    query = st.text_input(
        "💬 Ask anything about RuPaul's Drag Race:",
        placeholder="e.g. Who is RuPaul? What is a lip sync?"
    )

    num_results = st.slider("Number of results to show", 1, 5, 3)

    if query:
        results = vectorstore.similarity_search(query, k=num_results)
        st.markdown(f"### Results for: *{query}*")
        for i, result in enumerate(results):
            with st.expander(f"📄 Result {i+1}", expanded=True):
                st.write(result.page_content)

def about_page():
    st.title("ℹ️ About This App")

    st.markdown("""
    ## What is this app?
    This is a **Retrieval-Augmented Generation (RAG)** application built as part of a 
    university assignment. It uses semantic search to let users explore a curated 
    knowledge base about **RuPaul's Drag Race**.

    ## How does it work?
    
    ### 1. 📄 Documents
    The app contains 12 hand-curated text documents covering different aspects of 
    RuPaul's Drag Race — from the show's history and format, to iconic queens, 
    challenges, terminology, and international versions.
    
    ### 2. ✂️ Chunking
    Each document is split into smaller overlapping pieces called **chunks** using 
    LangChain's `RecursiveCharacterTextSplitter`. You can experiment with different 
    chunk sizes directly on the Search page!
    
    - **Small chunks (150)** → More precise results, less context per result
    - **Medium chunks (300)** → Balanced precision and context ✅ Default
    - **Large chunks (500)** → More context, but sometimes less precise
    
    ### 3. 🧠 Embeddings
    Each chunk is converted into a list of numbers (called an **embedding**) using 
    the `all-MiniLM-L6-v2` model from HuggingFace. These numbers capture the 
    *meaning* of the text, not just the words.
    
    ### 4. 🗄️ Vector Database
    All embeddings are stored in **ChromaDB**, a vector database. When you search, 
    your query is also converted to an embedding and compared against all chunks 
    to find the most similar ones.
    
    ### 5. 🔍 Semantic Search
    Unlike keyword search (like Ctrl+F), semantic search understands *meaning*. 
    For example, searching "who got eliminated first?" will find results about 
    queens being sent home, even if those exact words aren't in the documents.

    ---
    
    ## 🛠️ Tech Stack
    | Tool | Purpose |
    |------|---------|
    | Streamlit | Web application framework |
    | LangChain | Text splitting and RAG pipeline |
    | ChromaDB | Vector database |
    | HuggingFace | Embedding model |
    | Render.com | Cloud deployment |
    | GitHub | Version control |
    
    ---
    
    ## 👩‍💻 Built by
    Jana jovanovic, a student passionate about both AI and RuPaul's Drag Race! 👑
    """)

# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("👑 Drag Race RAG")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", ["🏠 Home", "🔍 Search", "ℹ️ About"])

if page == "🏠 Home":
    home_page()
elif page == "🔍 Search":
    search_page()
elif page == "ℹ️ About":
    about_page()

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ and drag")