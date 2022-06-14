# Generated by Django 4.0.5 on 2022-06-14 03:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0003_month'),
    ]

    operations = [
        migrations.CreateModel(
            name='OneMonthGovsAndIndexes',
            fields=[
                ('dataframefile_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='base.dataframefile')),
                ('month', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='base.month')),
            ],
            bases=('base.dataframefile',),
        ),
    ]
