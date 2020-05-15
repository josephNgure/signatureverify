from django import forms
from .models import Snippet

class ContactForm(forms.Form):
    name = forms.CharField(required=False)
    email=forms.EmailField()
    category = forms.ChoiceField(choices=[('question','question'), ('other','other')])
    subject=forms.CharField(required=False)
    body= forms.CharField(widget=forms.Textarea)



class SnippetForm(forms.ModelForm):
    
    class Meta:
        model = Snippet
        fields = ("name","body")

    