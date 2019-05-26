import urllib.request
from datetime import datetime

def find_between(s, first, last):

    try:
        start = s.index(first) + len( first )
        end = s.index(last, start )

        list.append(s[start:end])
        if s[start:end]!=None:
           temp = s[start:end]

           find_between(s[start:],first,last)
        else:
           return "azka"
    except ValueError:

        return "azka2"



def find_href(s, first, last):
    try:
        start = s.index(first) + len( first )
        end = s.index(last, start )

        return s[start:end]
    except ValueError:

        return "azka2"




html = ""

while True:

	with urllib.request.urlopen('https://blindsight.000webhostapp.com/?dir=pictures/') as response:
   		html = response.read()

	temp = ""
	list = []

	find_between(html,b'<tr>',b'</tr>')

	href = str(find_href(list[-1],b'href="',b'"'), 'utf-8')


	temp = str(find_href(href,'pictures/','.JPG'))

	datetime_object = datetime.strptime(temp, '%Y_%m_%d_%H_%M_%S')
	a = datetime.now()
	c = a - datetime_object
	print(c)
	print(c.total_seconds())
	if(c.total_seconds()<3*60):
		print(temp)
		print(datetime_object)
		href = "https://blindsight.000webhostapp.com/"+href
		urllib.request.urlretrieve(href, ""+temp+".JPG")  

		import recognize_faces_image

	else:
		continue


