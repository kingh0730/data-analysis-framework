from django.db import models

# Create your models here.


class Month(models.Model):
    month_int = models.PositiveSmallIntegerField()
    month_str = models.CharField(max_length=7)

    def __str__(self) -> str:
        return f"{self.month_str} ({self.month_int})"


class DataFrameFile(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    cached_file = models.FileField()


class OneMonthGovsAndIndexes(DataFrameFile):
    month = models.ForeignKey(Month, on_delete=models.PROTECT)


class Job(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    input_data_frames = models.ManyToManyField(
        DataFrameFile, related_name="input", blank=True
    )
    output_data_frames = models.ManyToManyField(DataFrameFile, related_name="output")
