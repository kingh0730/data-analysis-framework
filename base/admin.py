from django.contrib import admin

# Register your models here.


from . import models


@admin.register(models.DataFrameFile)
class DataFrameFileAdmin(admin.ModelAdmin):
    readonly_fields = [
        "get_data_frame_file_type",
        "get_upper_job_id",
        "get_upper_job_type",
    ]
    list_display = [
        "id",
        "created_at",
        "get_data_frame_file_type",
        "get_upper_job_id",
        "get_upper_job_type",
        "file_exists",
    ]


@admin.register(models.Job)
class JobAdmin(admin.ModelAdmin):
    readonly_fields = [
        "get_job_type",
        "percentage_progress",
        "has_all_outputs",
    ]
    list_display = [
        "id",
        "created_at",
        "get_job_type",
        "percentage_progress",
        "progress",
        "has_all_outputs",
    ]


@admin.register(models.DownloadOneMonth)
class DownloadOneMonthAdmin(admin.ModelAdmin):
    readonly_fields = [
        # pylint: disable=no-member, protected-access
        field.name
        for field in models.DownloadOneMonth._meta.get_fields()
    ] + [
        "percentage_progress",
        "has_all_outputs",
    ]
    list_display = [
        "id",
        "created_at",
        "month",
        "percentage_progress",
        "progress",
        "has_all_outputs",
    ]


class CapitalJobsAdmin(admin.ModelAdmin):
    readonly_fields = [
        "input",
        "output",
        "percentage_progress",
        "has_all_outputs",
    ]
    list_display = [
        "id",
        "created_at",
        "percentage_progress",
        "progress",
        "has_all_outputs",
    ]


admin.site.register(
    [
        models.AssociationRules,
        models.DynamicTimeWarping,
        models.NormalizeEachCol,
        models.DiscretizeAndToOneHot,
        models.RemoveDispersionSmallCol,
        models.RemoveTooManyDuplicatesCol,
        models.AppendInfoToAssociationRules,
        models.FrequentItemSets,
        models.RemoveIndexesThatHaveSomeStringInNames,
        models.KMeans,
        models.FilterByValueInOneCol,
        models.DistrictsPopulationLevelFiltering,
        models.DistrictsIncomeLevelFiltering,
        models.DistrictsSevenAreasLevelFiltering,
        models.ProvincialCapitalsChildrenFiltering,
    ],
    CapitalJobsAdmin,
)

admin.site.register(
    # Info
    [
        models.Month,
    ]
    +
    # DataFrameFiles
    [
        models.OneMonthGovsAndIndexes,
        models.OneMonthGovLevelAndIndexes,
        models.FeaturesPairwise,
        models.FeaturesPairwiseFlat,
    ]
    +
    # Jobs
    [
        # models.Job,
        models.MultivariateOutliersRemoval,
        models.EachColOutliersIdentification,
        models.TwoDataFramesComparison,
        models.GovLevelFiltering,
        models.LowQualityColsRemoval,
        models.CorrelationCalculation,
        models.SortValuesInFeaturesPairwise,
        models.DispersionCalculation,
        models.FilterValuesInFeaturesPairwiseFlat,
        models.CountOccurrencesInFeaturesPairwiseFlat,
        models.PCA,
        models.EachColPerCapita,
        models.AppendInfoToFeaturesPairwiseFlat,
        # models.DownloadOneMonth,
        models.TimeSeriesSimilarities,
        models.EachColRelativeOrdering,
        models.DistanceBetweenColPairs,
        models.LowQualityRowsAndColsRemoval,
        models.FillNA,
        models.CalcValueOntologies,
        models.DownloadOneGovAcrossTime,
        models.FilterByInfo,
        models.ConcatDataFrames,
        models.DownloadOneGovMacroDataAcrossTime,
        models.AppendMacroDataInfoToFeaturesPairwiseFlat,
        models.DiscretizeEachCol,
    ]
)
