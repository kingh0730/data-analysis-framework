from django.contrib import admin

# Register your models here.


from . import models


@admin.register(models.InputToOutputDocument)
class InputToOutputDocumentAdmin(admin.ModelAdmin):
    list_display = [
        # pylint: disable=no-member, protected-access
        field.name
        for field in models.InputToOutputDocument._meta.get_fields()
    ]
