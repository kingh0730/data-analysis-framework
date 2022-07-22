# Generated by Django 4.0.5 on 2022-07-13 09:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0086_kmeans_use_cols'),
    ]

    operations = [
        migrations.CreateModel(
            name='FilterByValueInOneCol',
            fields=[
                ('job_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='base.job')),
                ('use_col', models.CharField(max_length=255)),
                ('min_value', models.FloatField(blank=True, null=True)),
                ('max_value', models.FloatField(blank=True, null=True)),
                ('input', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='input_filter_by_value_in_one_col', to='base.dataframefile')),
                ('output', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_filter_by_value_in_one_col', to='base.dataframefile')),
            ],
            options={
                'verbose_name_plural': '2.J. Filter by value in one col',
            },
            bases=('base.job',),
        ),
    ]