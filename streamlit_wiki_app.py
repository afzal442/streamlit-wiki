import streamlit as st
import replicate
import wikipedia
import os
from transformers import AutoTokenizer

# Custom function to initialize the page
def initPage(title):
    st.set_page_config(page_title=title)

initPage("Search Wikipedia")

# Helper functions
def link(i, item):
    return f"**[{i + 1}. {item['title']}]({item['url']})**"

def aggregate(items):
    groups = {}
    for item in items:
        groups.setdefault(item["url"], []).append(item)
    results = []
    for group in groups.values():
        result = {}
        result["url"] = group[0]["url"]
        result["title"] = group[0]["title"]
        result["text"] = "\n\n".join([item["text"] for item in group])
        results.append(result)
    return results

def render_suggestions():
    def set_query(query):
        st.session_state.suggestion = query

    suggestions = [
        "Travel destinations known for their beaches",
        "Time travel movies with a twist",
        "Book authors explaining Physics",
    ]
    columns = st.columns(len(suggestions))
    for i, column in enumerate(columns):
        with column:
            st.button(suggestions[i], on_click=set_query, args=[suggestions[i]])

def render_query():
    st.text_input(
        "Search",
        placeholder="Search, e.g. 'Backpacking in Asia'",
        key="user_query",
        label_visibility="collapsed",
    )

def get_query():
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    user_query = st.session_state.suggestion or st.session_state.user_query
    st.session_state.suggestion = None
    st.session_state.user_query = ""
    return user_query

# Initialize API token and set up model parameters
if 'REPLICATE_API_TOKEN' in st.secrets:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']
else:
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
        st.warning('Please enter your Replicate API token.', icon='⚠️')
        st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")
        st.stop()

os.environ['REPLICATE_API_TOKEN'] = replicate_api
temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

user_query = get_query()
if not user_query:
    st.info(
        "Search Wikipedia and summarize the results using Snowflake Arctic. Type a query to start or pick one of these suggestions:"
    )
render_suggestions()
render_query()

if not user_query:
    st.stop()

MAX_ITEMS = 3

container = st.container()
header = container.empty()
header.write(f"Looking for results for: _{user_query}_")
placeholders = []
for i in range(MAX_ITEMS):
    placeholder = container.empty()
    placeholder.write("Searching...")
    placeholders.append(placeholder)

def search_wikipedia(query, limit=3):
    wikipedia.set_lang("en")
    search_results = wikipedia.search(query, results=limit)
    results = []
    for title in search_results:
        try:
            page = wikipedia.page(title)
            results.append({"url": page.url, "title": page.title, "text": page.summary})
        except wikipedia.PageError:
            continue
    return results

# Generate a response using Snowflake Arctic
def generate_arctic_response(prompt):
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokens = tokenizer.tokenize(prompt)
    if len(tokens) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.stop()

    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                                  input={"prompt": prompt, "temperature": temperature, "top_p": top_p}):
        yield str(event)

items = search_wikipedia(user_query, limit=10)
items = aggregate(items)[:MAX_ITEMS]

header.write(f"That's what I found about: _{user_query}_. **Summarizing results...**")
for i, item in enumerate(items):
    placeholders[i].markdown(f"{link(i, item)}")
    placeholders[i].markdown(f"**Content:** {item['text']}")

for i, item in enumerate(items):
    with placeholders[i]:
        prompt = f"Summarize the following text:\n\n{item['text']}"
        response = generate_arctic_response(prompt)
        summary = ''.join(response)
    placeholders[i].success(f"{link(i, item)}\n\n**Summary:** {summary}")

header.write("Search finished. Try something else!")
