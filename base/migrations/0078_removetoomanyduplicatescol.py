# Generated by Django 4.0.5 on 2022-07-07 10:17

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0077_alter_dispersioncalculation_measure_of_dispersion_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='RemoveTooManyDuplicatesCol',
            fields=[
                ('job_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='base.job')),
                ('max_duplicates_percentage', models.FloatField()),
                ('input', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='input_remove_too_many_duplicates_col', to='base.dataframefile')),
                ('output', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_remove_too_many_duplicates_col', to='base.dataframefile')),
            ],
            options={
                'verbose_name_plural': '2.E. Remove too many duplicates col',
            },
            bases=('base.job',),
        ),
    ]