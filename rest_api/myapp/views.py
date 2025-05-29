from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import InputSerializer, OutputSerializer
from .utils.load_model import load_model, predict
from .utils.preprocessing import preprocess_image
import torch
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

class PredictionAPIView(APIView):

    @method_decorator(cache_page(300))  
    def post(self, request):
        labels = ['glioma', 'menin', 'tumor']
        serializer = InputSerializer(data=request.data)
        
        if serializer.is_valid():
            serializer.save()
            preprocessed = preprocess_image(serializer.validated_data['image'])
            
            try:
                preprocessed = preprocess_image(serializer.validated_data['image'])
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            logits = predict(preprocessed)
            probabilities = torch.sigmoid(logits)
            prediction_label = probabilities.argmax(dim=1).item()
            name_prediction = labels[prediction_label]
            probabilities = probabilities.tolist()[0]
            
            output_data = {
                "prediction": name_prediction,
                "prediction_label": prediction_label,
                "probabilities": probabilities,
                "image": serializer.validated_data['image']
            }
            output = OutputSerializer(output_data)

            return Response(output.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
    
