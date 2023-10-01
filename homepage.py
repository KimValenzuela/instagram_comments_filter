import streamlit as st
from PIL import Image
import yaml
from yaml.loader import SafeLoader
from ig_api import show_posts, get_posts, get_comments, sentiment_analysis, res_comment
from vector_db import loader_docs, make_chain
import matplotlib.pyplot as plt
import pandas as pd

__version__ = "0.0.1"
app_name = 'Community Manager Bot'
models = ['text-davinci-003','text-curie-001'] #put here models available
embed_models = ['text-embedding-ada-002']

#we read prompt tasks from yaml file
with open('prompts.yml') as f:
	prompts = yaml.load(f, Loader=SafeLoader)

im = Image.open("images/ey2.png")
st.set_page_config(page_title=app_name, page_icon=im, layout="centered", initial_sidebar_state="auto", menu_items=None)
ss = st.session_state #this variable saves params of all widgets, the name of each param is the same that key value in widgets

def app_spacer(n=2, line=False, next_n=0):
	"""Function that create spaces between lines.

	Parameters:
		n (int): 
		line (boolean): 
		next_n (int): 

	Returns:
		None
	"""

	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')

def app_info():
	"""Function that fill the sidebar with information about author.

	Parameters:
		None

	Returns:
		None
	"""

	st.markdown(f"""
		# {app_name}
		version: {__version__}
		
		A system that classifies social media comments and suggests a response.
		""")
	app_spacer(1)
	st.write("Made by [Kim Valenzuela](https://github.com/KimValenzuela/). Supported by EY-Chile Training Team.", unsafe_allow_html=True)
	app_spacer(1)
	st.markdown("""
		This project was built as example for a training session framed in the Summit País Digital Hackatón by EY-Microsoft.
.
		""")
	app_spacer(1)
	st.markdown('')

def app_set_temperature():
	"""Function that allows to set temperature to a model.

	Parameters:
		None

	Returns:
		None
	"""

	st.slider('Temperature', 0.0, 1.0, 0.0, 0.01, key='temperature', format='%0.2f')
	print(ss['temperature'])
	#ss['temperature'] = 0.0

def app_llm_model():
	"""Function that allows to select the model using a list of models setted above (see parameters section).

	Parameters:
		None

	Returns:
		None
	"""

	st.selectbox('OpenAI model', models, key='model')
	st.selectbox('embedding model', embed_models, key='model_embed')



def insert_link(client):
	link = st.text_input('Insert post link: ')
	if link:
		media_pk = get_posts(client, link)

		post_ig(client, media_pk)



def post_ig(client, media_pk):

	img = show_posts(client, media_pk)
	if img:
		st.header(f'Post 1')
		st.image(img, caption='Post de instagram')
		model_embed = ss['model_embed']
		vectordb = loader_docs(model_embed)
		show_sent_analysis(client, media_pk)
		show_query_comments(vectordb)
			

def show_sent_analysis(client, media_pk):
	comments = get_comments(client, media_pk)
	#col1, col2 = st.columns(2)
	st.header('Sentiment Analysis')
	context = st.text_input('Describe your post to get comment analysis: ')
	if context:
		response = get_comment_clasification(client, media_pk, context)
		st.header('Comment statistics')
		create_chart(response)

		model = ss['model']
		temperature = ss['temperature']
		
		comment, answer = res_comment(context, comments, model, temperature)
		st.write(pd.DataFrame({
			'Comments': comment,
			'Answer': answer
		}))



def show_query_comments(vectordb):
	st.header('Check the comments of your publication')
	query = st.text_area('Enter a question: ')
	if query:
		response_query = query_comments_post(vectordb, query)
		st.header('Your followers say:')
		st.write(response_query)



def query_comments_post(vectordb, text):
	model = ss['model']
	temperature = ss['temperature']

	response = make_chain(model, temperature, vectordb, text)

	return response



def get_comment_clasification(client, pk, context):
	comments = get_comments(client, pk)

	model = ss['model']
	temperature = ss['temperature']

	sent_analysis = []

	for comment in comments:
		response = sentiment_analysis(context, comment, model, temperature)
		sent_analysis.append(response)



	return sent_analysis

def create_chart(c):
    labels = ['Positivo', 'Neutral', 'Negativo']
    
    bueno_count = c.count('Bueno')
    neutral_count = c.count('Neutral')
    malo_count = c.count('Malo')
    
    sizes = [bueno_count, neutral_count, malo_count]
    colors = ['#F7B800', '#00A6ED', '#EF3340']
    font_color = '#deece8'

    fig, ax = plt.subplots()
    ax.bar(labels, 
            sizes, 
            color=colors,
            alpha=0.7, edgecolor=font_color, linewidth=1.2
		)
	
    for i, size in enumerate(sizes):
        ax.text(i, size, str(size), ha='center', va='bottom', color=font_color, fontsize=10)

    ax.set_ylabel('Cantidad', fontsize=14, color=font_color)
    ax.set_title('Análisis de sentimientos', fontsize=14, color=font_color)
    plt.tight_layout()

    ax.set_xticks(labels)
    ax.set_xticklabels(labels, fontsize=12, color=font_color) 
    ax.set_ylim(0, max(sizes) + 1)  
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.grid(color='white', linestyle='--', linewidth=0.5)

    fig.savefig('pie.png', transparent=True)

    fig.savefig('pie.png', transparent=True)
    st.image('pie.png')


def main_screen(client):
	"""Function that creates the frontend layout.

	Parameters:
		None

	Returns:
		None
	"""	

	st.title(f"📝 :female-office-worker: :office_worker:  {app_name}")

	#SIDEBAR
	with st.sidebar:
		app_info()
		app_spacer(2)
		with st.expander('Advanced Parameters'):
			app_llm_model()
			app_set_temperature()
			# app_task_names()
			# app_task_modifier()

	insert_link(client)
	#post_ig(client, media_pk)