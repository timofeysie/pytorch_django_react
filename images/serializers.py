from rest_framework import serializers
from .models import Image

class ImageSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')

    class Meta:
        model = Image
        # to list all fields all in an array or set to '__all__'  
        fields = [
            'id', 'owner', 'created', 'title',
        ]