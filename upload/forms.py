from django import forms
from .models import UploadSignature
#DataFlair #File_Upload
class UploadSignatureForm(forms.ModelForm):
    class Meta:
        model = UploadSignature
        fields = "__all__"





