# Generated by Django 4.0.5 on 2022-06-15 05:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0011_alter_govlevelfiltering_output'),
    ]

    operations = [
        migrations.AddField(
            model_name='govlevelfiltering',
            name='gov_level',
            field=models.SmallIntegerField(choices=[(-1, 'GovLevel.ALL'), (0, 'GovLevel.NATION'), (1, 'GovLevel.PROVINCE'), (2, 'GovLevel.CITY_LEVEL'), (3, 'GovLevel.DISTRICT_LEVEL')], default=-1),
            preserve_default=False,
        ),
    ]