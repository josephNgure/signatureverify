from django.shortcuts import render
from django.http import  HttpResponse
from django.http import JsonResponse
from django.core import serializers
from .forms import ContactForm,SnippetForm
import json

# Create your views here.

def contact(request):
    if request.method == 'POST':
        form=ContactForm(request.POST)
        if form.is_valid():
            name=form.cleaned_data['name']
            email=form.cleaned_data['email']

            print(name, email)
            jj=json.dumps(name)
            print(jj)
    form=ContactForm()
    # jsonlist=serializers.serialize('json', form)
    return render(request,'form.html', {'form': form})
    # return HttpResponse(json.dumps(form))
    # return JsonResponse({'foo':'bar'})
    # return HttpResponse(jsonlist, content_type="text/json-comment-filtered")

# def posts(request):
#     posts = Post.objects.filter(published_at__isnull=False).order_by('-published_at')
#     post_list = serializers.serialize('json', posts)
#     return HttpResponse(post_list, content_type="text/json-comment-filtered")


def Snippet(request):
    if request.method == 'POST':
        form=SnippetForm(request.POST)

        if form.is_valid():
            name=form.cleaned_data['name']
            body=form.cleaned_data['body']

            print(name, body)
            print('valid')
            form.save()
    form=SnippetForm()
    return render(request,'form.html', {'form': form})
    



# from django.http import HttpResponseRedirect


# def get_name(request):
#     # if this is a POST request we need to process the form data
#     if request.method == 'POST':
#         # create a form instance and populate it with data from the request:
#         form = NameForm(request.POST)
#         # check whether it's valid:
#         if form.is_valid():
#             # process the data in form.cleaned_data as required
#             # ...
#             # redirect to a new URL:
#             return HttpResponseRedirect('/thanks/')

#     # if a GET (or any other method) we'll create a blank form
#     else:
#         form = NameForm()

#     return render(request, 'name.html', {'form': form})