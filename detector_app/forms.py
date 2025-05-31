# detector_app/forms.py
from django import forms

class NewsTextForm(forms.Form):
    news_article = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'cols': 70, 'placeholder': 'Enter news text here...'}),
        label="News Article Text",
        required=True
    )