#coding:utf-8
from urllib import request,parse
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import pycurl

#--------------------
# do this once at program startup
#--------------------
import socket
origGetAddrInfo = socket.getaddrinfo

def getAddrInfoWrapper(host, port, family=0, socktype=0, proto=0, flags=0):
	return origGetAddrInfo(host, port, socket.AF_INET, socktype, proto, flags)

# replace the original socket.getaddrinfo by our version
socket.getaddrinfo = getAddrInfoWrapper

#--------------------


def get_from_keyword():
	with open("/Users/Atsu/git/data/keyword_hatena.txt","r",encoding='utf-8') as file1:
		doc = file1.readlines()
	name_list = []
	for line in doc:
		#print(line)
		hiragana, name = line.split("	")
		name_list.append(name)
	count = 10
	category_list = []
	for name in name_list:
		if count > 0:
			print(name)
			name = parse.quote_plus(name)
			#開発ブログのIDつき
			url = "http://d.hatena.ne.jp/keyword/"+name
			#print(url)
			#urlにアクセス
			try:
				response = request.urlopen(url)
			except HTTPError:
				continue
			
			#soupに食わせる
			#soup = BeautifulSoup(response)
			#category = soup.find(class_ ="category").string
			#category_list.append(category)
			#count -= 1
	print(category_list)

	"""
	#[name,category]
	for line in doc:
		if category == "name":
			print(name, "PER")

	|tee /.txt
	"""
	return

def get_from_page():
	url_prefix = "http://d.hatena.ne.jp/keywordlist?r="
	count = 0
	url_suffix = "&s=created&cname="
	category = "geography"
	for count in range(0,100):
		url = url_prefix+str(count)+url_suffix+category
		print(url)
		count += 20 
	return
def main():
	get_from_keyword()
	#get_from_page()

if __name__ == '__main__':
	main()
