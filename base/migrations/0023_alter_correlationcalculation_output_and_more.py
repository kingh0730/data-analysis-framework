# Generated by Django 4.0.5 on 2022-06-15 10:13

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0022_alter_eachcoloutliersidentification_output'),
    ]

    operations = [
        migrations.AlterField(
            model_name='correlationcalculation',
            name='output',
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_correlation_calculation', to='base.featurespairwise'),
        ),
        migrations.AlterField(
            model_name='govlevelfiltering',
            name='output',
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_gov_level_filtering', to='base.onemonthgovlevelandindexes'),
        ),
        migrations.AlterField(
            model_name='lowqualitycolsremoval',
            name='output',
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_low_quality_cols_removal', to='base.dataframefile'),
        ),
        migrations.AlterField(
            model_name='sortvaluesinfeaturespairwise',
            name='output',
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_sort_values_in_features_pairwise', to='base.featurespairwiseflat'),
        ),
        migrations.AlterField(
            model_name='twodataframescomparison',
            name='output',
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='output_two_data_frames_comparison', to='base.dataframefile'),
        ),
    ]