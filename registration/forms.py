# from django import forms
# #DataFlair #Form
# class SignUp(forms.Form):
#     first_name = forms.CharField(initial = 'First Name', max_length=30 )
#     last_name = forms.CharField(max_length=30)
#     email = forms.EmailField(help_text = 'write your email', max_length=100)
#     Address = forms.CharField(required = False,max_length=100 )
#     Technology = forms.CharField(initial = 'Django', disabled = True,max_length=30 )
#     age = forms.IntegerField()
#     password = forms.CharField(widget = forms.PasswordInput)
#     re_password = forms.CharField(help_text = 'renter your password', widget = forms.PasswordInput)

from django import forms
from django.core import validators
#DataFlair #Form
class SignUp(forms.Form):
  first_name = forms.CharField(initial = 'First Name', )
  last_name = forms.CharField(required = False)
  email = forms.EmailField(help_text = 'write your email', required = False)
  Address = forms.CharField(required = False, )
  Technology = forms.CharField(initial = 'Django', disabled = True)
  age = forms.IntegerField(required = False, )
  password = forms.CharField(widget = forms.PasswordInput, validators = [validators.MinLengthValidator(6)])
  re_password = forms.CharField(widget = forms.PasswordInput, required = False)