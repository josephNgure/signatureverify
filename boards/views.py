from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from datetime import date

# create model to json
from django.core import serializers
from .models import Post

def home(*args, **kwargs):
    return HttpResponse("<h1>Hello Joseph django here</>")

# def datey(request):
def datey(request,year,month):
    # t=date.today()
    # month=date.strftime(t, '%b')
    # year=t.year
    title = "the socks brand launched- %s%s" %(month,year)
    return HttpResponse("<h1>%s</h1>" % title)
    # return HttpResponse("<h1>Hello Joseph django here</>")
def bootstrap(request):
    return render(request,'hello.html',{})




def posts(request):
    posts = Post.objects.filter(published_at__isnull=False).order_by('-published_at')
    post_list = serializers.serialize('json', posts)
    return HttpResponse(post_list, content_type="text/json-comment-filtered")