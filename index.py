import os
from lib import utils
from lib.streaming import StreamHandler
import streamlit as st

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

st.set_page_config(page_title="ChatPDF", page_icon="📄")
LOGO_PATH = "logo.jpg"

# Add custom HTML and CSS for upper-left logo placement without the white background
st.markdown(
    f"""
    <style>
        .logo-container {{
            display: flex;
            align-items: center;
            position: fixed;
            top: 0;
            left: 0;
            padding: 10px;
            z-index: 1000;
        }}
        .logo-container img {{
            height: 150px; /* 3x bigger logo */
        }}
    </style>
    <div class="logo-container">
        <img src="data:image/jpeg;base64,{st.image(LOGO_PATH, output_format='jpeg', width=150)}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for tracking whether a question has been asked
if "question_asked" not in st.session_state:
    st.session_state["question_asked"] = False


st.header('Sitcom Knowledge Base - Explore Your Favorite TV Shows')
st.write('This application provides detailed descriptions and insights into popular sitcoms. Users can query the database to learn more about their favorite shows, including cast information, plot summaries, and unique trivia.')


st.subheader('Features of the Application')
st.markdown('''
Search by Sitcom Name: Quickly access descriptions of your favorite sitcoms.

Explore Cast Details: Get insights into the actors who brought these iconic characters to life.

Get Recommendations: Find sitcoms based on your preferences.

Use the search bar or navigate through the list to explore the sitcom world!
''')


# Only show additional content if a question has been asked
if st.session_state["question_asked"]:
    st.subheader('Features of the Application')
    st.markdown('''
    Search by Sitcom Name: Quickly access descriptions of your favorite sitcoms.

    Explore Cast Details: Get insights into the actors who brought these iconic characters to life.

    Get Recommendations: Find sitcoms based on your preferences.

    Use the search bar or navigate through the list to explore the sitcom world!
    ''')

class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()
        self.current_action = "Ask another question"  # Default action

    def import_source_documents(self):
        """Load, split, and vectorize documents."""
        docs = []
        files = []
        for file in os.listdir("data"):
            if file.endswith(".txt"):
                with open(os.path.join("data", file), "r", encoding="utf-8") as f:
                    docs.append(f.read())
                    files.append(file)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = []
        for i, doc in enumerate(docs):
            for chunk in text_splitter.split_text(doc):
                splits.append(Document(page_content=chunk, metadata={"source": files[i]}))

        vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 2, 'fetch_k': 11}
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are a chatbot tasked with responding to questions about sitcoms based on the detailed descriptions and data in the attached knowledge base.
            
            {context}
            
            Using only the content provided above, answer the following question accurately and concisely:
            {question}
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return qa_chain, docs, splits

    def analyze_documents(self, docs):
        """Perform statistical analysis on the documents."""
        st.subheader("Text Analysis")
        num_docs = len(docs)
        word_counts = [len(doc.split()) for doc in docs]
        total_words = sum(word_counts)
        avg_words = total_words / num_docs if num_docs > 0 else 0
        st.write(f"Number of documents: {num_docs}")
        st.write(f"Total words: {total_words}")
        st.write(f"Average words per document: {avg_words:.2f}")

        word_freq = Counter(" ".join(docs).split())
        most_common_words = word_freq.most_common(10)
        st.write("Most common words:")
        st.table(pd.DataFrame(most_common_words, columns=["Word", "Frequency"]))

    def visualize_embeddings(self, splits):
        """Visualize embeddings in 2D space using PCA."""
        embeddings = [self.embedding_model.embed_query(doc.page_content) for doc in splits]
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        fig, ax = plt.subplots()
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        ax.set_title("Visualization of Text Embeddings")
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        st.pyplot(fig)

    def handle_next_action(self, docs, splits):
        """Handle the next action based on user choice."""
        if self.current_action == "View text analysis and embeddings":
            self.analyze_documents(docs)
            self.visualize_embeddings(splits)
        elif self.current_action == "Ask another question":
            st.write("You can ask another question using the chat input below.")

    @utils.enable_chat_history
    def main(self):
        qa_chain, docs, splits = self.import_source_documents()

        # Check for user query
        user_query = st.chat_input(placeholder="Ask for information from documents")
        if user_query:
            # Mark that a question has been asked
            st.session_state["question_asked"] = True

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                result = qa_chain.invoke(
                    {"question": user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(CustomDocChatbot, user_query, response)

                # Show references
                for doc in result['source_documents']:
                    filename = os.path.basename(doc.metadata['source'])
                    ref_title = f":blue[Source document: {filename}]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)

        # Provide options for further actions only if a question has been asked
        if st.session_state["question_asked"]:
            self.current_action = st.radio(
                "What would you like to do next?",
                ["Ask another question", "View text analysis and embeddings"],
                key="user_action"
            )
            self.handle_next_action(docs, splits)


if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
