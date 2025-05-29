from rest_framework import serializers
from django.db import models

class MyModel(models.Model):
    image = models.ImageField(upload_to='uploads/')


class InputSerializer(serializers.ModelSerializer):
    class Meta:
        model = MyModel
        fields = ['image']


class OutputSerializer(serializers.Serializer):
    prediction = serializers.CharField()
    prediction_label = serializers.IntegerField()
    probabilities = serializers.ListField()
    image_url = serializers.URLField(source='image.url', read_only=True)
    
    class Meta:
        fields = ['prediction', 'prediction_label', 'probabilities', 'image_url']