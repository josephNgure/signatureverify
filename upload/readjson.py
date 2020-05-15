import json
f = open("file.json", "r")
for h in f:
    print(h)
    x=h.split("}")
    thelist=[]

    for doc in x[:-1]:
            print(doc)
            full=doc+'}'
            # print(full)
            y=json.loads(full)
            print(y)
            # print(y["companyName"])

            id=y["id"]
            userid=y["userid"]
            filepath=y["filepath"]
            # userId=y["UserId"]
            # del y["id"]
            print(userid)

            break
    print("hhhhhhhhhhhhhhhhhhhhhh")
    h=h[3:]
    print(h)
    print("mmmmmmmmMMMMMMMMMMMMMMMMMMMMMMMM") 
    # result = h.find('{') 
    # print ("Substring 'geeks' found at index:", result ) 

    if (h.find('{') != -1): 
        print ("Contains given substring ") 
        result = h.find('{') 
        print ("Substring 'geeks' found at index:", result )
        y=h[result:]

        f = open("file.json", "w")
        f.write(y)      

    else: 
        print ("Doesn't contains given substring")
        f = open("file.json", "w")
        f.write("") 
f.close()


