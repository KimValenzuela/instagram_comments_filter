import urllib.request
from PIL import Image
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def get_posts(client, link):
    media_pk = client.media_pk_from_url(link)
    return media_pk

def show_posts(client, media_pk):
    media_info = client.media_info(media_pk)

    try:
        url_image = media_info.image_versions2['candidates'][0]['url']
        urllib.request.urlretrieve(url_image, 'post.png')
        img_post = Image.open('post.png')

        return img_post
    except:
        print('url no encontrado')
        return None
    
def get_comments(client, pk_media):
    media_id = client.media_id(pk_media)
    comments = client.media_comments(media_id)

    media_comments = []

    for comment in comments:
        #if comment.user.username != client.username:
        media_comments.append(comment.text)
    #print(media_comments)

    with open('comments_file/comments.txt', 'w', encoding="utf-8") as f:
        for i in media_comments:
            f.write(i + '\n')

    return media_comments

def sentiment_analysis(context, comment, model, temperature):
    template_query = """
    Debes realizar un análisis de sentimiento a un comentario que hizo un usuario sobre mi post en instagram.
    El post es sobre el siguiente tema: {post_context} y necesito que clasifiques el comentario dentro de las siguientes opciones: Bueno, Neutral o Malo. 
    La diferencia entre un comentario Malo y uno de Odio, es que el comentario Malo puede ser alusivo a que el usuario no le gusta la publicación o no le parece bien, mientras 
    que un comentario de odio es un ataque a la persona.
    Comentario: {user_comment}
    Clasificación: 
    """
    prompt_temp = PromptTemplate(input_variables=['post_context', 'user_comment'], template=template_query)
    prompt_value = prompt_temp.format(post_context=context, user_comment=comment)

    llm = OpenAI(
        model = model,
        temperature = temperature
    )

    response = llm(prompt_value)

    return response


def response_comments(context, comment, model, temperature):

    template_query = """
    Este es un post de instagram sobre el siguiente tema: {post_context}.
    Necesito que respondas el comentario que ha dejado un usuario. Si el comentario no tiene sentido, no respondas nada. 
    Comentario: {user_comment}
    Respuesta: 
    """
    prompt_temp = PromptTemplate(input_variables=['post_context', 'user_comment'], template=template_query)
    prompt_value = prompt_temp.format(post_context=context, user_comment=comment)

    llm = OpenAI(
        model = model,
        temperature = temperature
    )

    response = llm(prompt_value)

    return response


def res_comment(context, comments, model, temperature):
    comment = []
    response = []

    for comm in comments:
        comment.append(comm)
        resp = response_comments(context, comm, model, temperature)
        response.append(resp)

    return comment, response