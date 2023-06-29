from instagrapi import Client
import yaml
from yaml.loader import SafeLoader
from homepage import main_screen
from dotenv import load_dotenv

def main():
    
    load_dotenv()

    with open('credentials.yml') as f:
        credentials = yaml.load(f, Loader=SafeLoader)

    client = Client()
    client.login(credentials['username'], credentials['password'])

    main_screen(client)

if __name__ == '__main__':
    main()