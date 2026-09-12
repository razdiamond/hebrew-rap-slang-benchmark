import os

import lyricsgenius
from dotenv import load_dotenv

load_dotenv()
token = os.getenv(key="GENIUS_ACCESS_TOKEN")
genius = lyricsgenius.Genius(access_token=token)
song = genius.search_song(title="Basbusa - בסבוסה")
print(type(song))
print(song)
