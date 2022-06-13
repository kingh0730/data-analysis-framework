from django.db import models

# Create your models here.


class DataFrameFile(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    cached_file = models.FileField()


class Job(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    input_data_frame = models.ForeignKey(
        DataFrameFile, on_delete=models.CASCADE, related_name="input", null=True
    )
    output_data_frame = models.ForeignKey(
        DataFrameFile, on_delete=models.CASCADE, related_name="output"
    )
