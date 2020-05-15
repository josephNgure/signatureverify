
from django.shortcuts import render
from .forms import UploadSignatureForm
from .models import UploadSignature
# ################## imports imported
import numpy as np
import os
import time
# mine
import json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg']




def signatureUpload(request):
    form = UploadSignatureForm()
    if request.method == 'POST':
        form = UploadSignatureForm(request.POST, request.FILES)
        if form.is_valid():
            user_pr = form.save(commit=False)
            user_pr.signature = request.FILES['signature']

            file_type = user_pr.signature.url.split('.')[-1]
            file_type = file_type.lower()
            if file_type not in IMAGE_FILE_TYPES:
                return render(request, 'upload/error.html')

            user_pr.save()
            print("ffffffffffffffffffffffffffffffff")
            

            # request.session['userid'] = user_pr.user_id
            # print(request.session['userid'])
            # request.session['filepath'] = user_pr.signature.url
            # print(request.session['filepath'])

            print(user_pr.id)
            print("vvvvvvvvvvvvvvvvvvvvv")

            fileDict={
                    "id":user_pr.id,
                    "userid": user_pr.user_id,
                    "filepath": user_pr.signature.url
                }
            print (fileDict)
            userjson=json.dumps(fileDict)
            print(userjson) 


            with open('upload/file.json','a') as f:
                json.dump(fileDict, f)


            
            print(user_pr)
###################
####################
            time.sleep(40)
            sys_id=user_pr.id
            response=""
            f = open("upload/response.json", "r")
            for x in f:
            # print(x)

                x=x.split("}")
                # print(x[:-1])
                thelist=[]

                for doc in x[:-1]:
                        # print(doc)
                        full=doc+'}'
                        print(full)
                        y=json.loads(full)
                        # print(y["companyName"])

                        id=y["id"]
                        if id==sys_id:
                            response=y["response"]
                            
                            break
            print(response)      

            f.close()

###################
#################



            return render(request, 'upload/details.html', {'user_pr': user_pr,'response':response})
    context = {"form": form,}
    return render(request, 'upload/upload.html', context)


