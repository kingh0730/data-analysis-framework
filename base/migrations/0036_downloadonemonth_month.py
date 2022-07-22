# Generated by Django 4.0.5 on 2022-06-21 08:36

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0035_downloadonemonth'),
    ]

    operations = [
        migrations.AddField(
            model_name='downloadonemonth',
            name='month',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.PROTECT, to='base.month'),
            preserve_default=False,
        ),
    ]