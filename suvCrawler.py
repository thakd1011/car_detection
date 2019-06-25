import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
import json
import os

SAVE_DIR = "input/directory/for/save/crawed file"
HEADERS = {}
HEADERS['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
COUNT = 256
# web engine = google
def make_url(search_word):
	# return word searching result page
	trans_img_text = "&source=lnms&tbm=isch"
	
	search_url = "http://www.google.com/search?q=" + search_word
	search_url += trans_img_text # only transparent img

	return search_url

def img_download(link, type, i):
	image = requests.get(link, stream=True)

	if len(type)==0:
		f = open(os.path.join(SAVE_DIR, "img"+"_"+str(i)+".jpg"), 'wb')
	else:
		f = open(os.path.join(SAVE_DIR, "img"+"_"+str(i)+"."+type), 'wb')

	for chunk in image.iter_content(chunk_size=256):  
		if chunk:
			f.write(chunk)


def get_img_link(web_url):
	# get html file using request lib
	global COUNT

	response = requests.get(web_url, headers=HEADERS)
	search_html = response.text

	soup = BeautifulSoup(search_html, 'html.parser')

	for a in soup.find_all("div", {"class":"rg_meta"}):

		link, type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
		print("link = ", link, ", type = ", type, "\n")
		img_download(link, type, COUNT)
		COUNT += 1


if __name__ == '__main__' :
	URL = make_url("suv back")
	get_img_link(URL)