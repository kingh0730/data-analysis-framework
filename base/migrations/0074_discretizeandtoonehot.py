# Generated by Django 4.0.5 on 2022-07-06 10:41

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0073_appendinfotofeaturespairwiseflat_has_input_comments'),
    ]

    operations = [
        migrations.CreateModel(
            name='DiscretizeAndToOneHot',
            fields=[
                ('job_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='base.job')),
                ('input', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='input_discretize_and_to_one_hot', to='base.dataframefile')),
                ('output', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_discretize_and_to_one_hot', to='base.dataframefile')),
            ],
            options={
                'verbose_name_plural': '2.D. Discretize and to one hot',
            },
            bases=('base.job',),
        ),
    ]