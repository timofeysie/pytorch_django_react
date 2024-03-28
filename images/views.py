from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Image
from .serializers import ImageSerializer
from image_classification.forms import ImageUploadForm
from image_classification.views import get_prediction

class Images(APIView):
    def get(self, request):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)

    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()

            # Process the image and get the predicted label
            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)
                predicted_label = "Prediction Error"

            # Return the predicted label in the response
            return Response({'predicted_label': predicted_label})
        else:
            return Response(form.errors, status=status.HTTP_400_BAD_REQUEST)