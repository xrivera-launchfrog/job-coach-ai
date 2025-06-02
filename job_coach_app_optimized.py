import streamlit as st
import pandas as pd
from llama_index.core import VectorStoreIndex, Document
import openai
import os

# Configure the page
st.set_page_config(
    page_title="Job Coach AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Job Coach AI")
st.subheader("Get data-backed career advice using real BLS employment projections")

# Add loading indicator
if 'index_loaded' not in st.session_state:
    st.session_state.index_loaded = False

# Load your data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('job_cards_streamlit.csv')
        st.success(f"✅ Loaded {len(df)} high-growth jobs")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def create_index():
    with st.spinner("🔄 Building job search index... This takes about 30 seconds"):
        try:
            df = load_data()
            if df is None:
                return None
                
            documents = []
            
            # Use your actual column names
            soc_col = 'SOC Code'
            title_col = 'Job Title'
            growth_col = 'Growth Rate % Employment change, percent, 2023–33'
            openings_col = 'Annual Job Openings - Occupational openings, 2023–33 annual average'
            education_col = 'Education Required'
            skills_col = 'Top_Skills'
            
            # Process only first 100 jobs for speed (you can increase this)
            for index, row in df.head(100).iterrows():
                if pd.isna(row[title_col]) or pd.isna(row[growth_col]):
                    continue
                    
                job_title = str(row[title_col]).strip()
                growth_rate = row[growth_col]
                annual_openings = str(row[openings_col]) if pd.notna(row[openings_col]) else "Data not available"
                education = str(row[education_col]) if pd.notna(row[education_col]) else "Not specified"
                skills = str(row[skills_col]) if pd.notna(row[skills_col]) else "Skills data not available"
                
                text_content = f"""
                Job Title: {job_title}
                SOC Code: {row[soc_col]}
                Growth Rate: {growth_rate}% projected growth from 2023 to 2033
                Annual Job Openings: {annual_openings}
                Education Required: {education}
                Key Skills Needed: {skills}
                
                {job_title} is experiencing {growth_rate}% growth through 2033. This role typically requires {education.lower()} and offers career prospects in a {"high-growth" if growth_rate > 15 else "stable"} field.
                """
                
                metadata = {
                    "soc_code": str(row[soc_col]),
                    "job_title": job_title,
                    "growth_rate": float(growth_rate),
                    "education": education,
                }
                
                doc = Document(text=text_content.strip(), metadata=metadata)
                documents.append(doc)
            
            if not documents:
                st.error("No documents created")
                return None
                
            index = VectorStoreIndex.from_documents(documents, show_progress=False)
            st.success(f"✅ Index created with {len(documents)} jobs")
            return index.as_query_engine(similarity_top_k=3)
            
        except Exception as e:
            st.error(f"Error creating index: {e}")
            return None

# Get API key
if 'openai_api_key' not in st.session_state:
    with st.expander("🔑 Setup Required", expanded=True):
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        st.markdown("Get your API key from: https://platform.openai.com/api-keys")
        
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            st.session_state.openai_api_key = api_key
            st.success("API key configured!")
            st.rerun()

if 'openai_api_key' in st.session_state:
    # Load data first
    df = load_data()
    
    if df is not None:
        # Show dataset stats
        growth_col = 'Growth Rate % Employment change, percent, 2023–33'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Jobs", len(df))
        with col2:
            st.metric("Avg Growth Rate", f"{df[growth_col].mean():.1f}%")
        with col3:
            st.metric("Max Growth Rate", f"{df[growth_col].max():.1f}%")
        
        # Create index
        query_engine = create_index()
        
        if query_engine:
            st.session_state.index_loaded = True
            
            # Main interface
            st.markdown("### 💬 Ask me about career opportunities:")
            
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What fast-growing jobs don't require a college degree?"
            )
            
            if question:
                with st.spinner("🔍 Finding the best career opportunities for you..."):
                    try:
                        response = query_engine.query(question)
                        st.markdown("### 💡 Career Advice:")
                        st.write(response.response)
                        
                        # Show data source
                        st.markdown("---")
                        st.markdown("*Based on real BLS employment projections and O*NET skills data*")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            # Sidebar with sample questions
            st.sidebar.markdown("### 💭 Try these questions:")
            sample_questions = [
                "What are the top 5 fastest growing jobs?",
                "Healthcare jobs that don't require a degree?",
                "Jobs with over 25% growth rate?",
                "Technology jobs without college degree?",
                "Best opportunities for career changers?"
            ]
            
            for sample in sample_questions:
                if st.sidebar.button(sample, key=f"btn_{sample[:20]}"):
                    st.session_state.sample_question = sample
                    st.rerun()
            
            # Handle sample question clicks
            if 'sample_question' in st.session_state:
                question = st.session_state.sample_question
                del st.session_state.sample_question
                
                with st.spinner("🔍 Finding answers..."):
                    try:
                        response = query_engine.query(question)
                        st.markdown(f"### 💭 Question: {question}")
                        st.markdown("### 💡 Answer:")
                        st.write(response.response)
                    except Exception as e:
                        st.error(f"Error: {e}")

else:
    st.info("👆 Please enter your OpenAI API key to get started")
    
# Footer
st.markdown("---")
st.markdown("Built with BLS Employment Projections data • Powered by LlamaIndex & OpenAI")
