from django.http import HttpRequest, JsonResponse
from django.shortcuts import render


# Create your views here.


def test_all_techniques(request: HttpRequest) -> JsonResponse:
    return JsonResponse(
        {
            "interpretation": "文字解读",
            "samples": {
                "criteria": [
                    {
                        "name": "gov_level",
                        "class": "DISTRICT_LEVEL",
                        "count": 2861,
                    },
                    {
                        "name": "population",
                        "class": "low|3",
                        "count": 55,
                        "range": [150, 700],
                    },
                ],
                "count": 31,
                "interpretation": "文字解读",
            },
            "relations": [
                {
                    "index_id": 6,
                    "technique": "association-rules",
                    "values": [
                        {
                            "name": "lift",
                            "value": 1.5,
                            "interpretation": "文字解读",
                        },
                        {
                            "name": "confidence",
                            "value": 0.28,
                            "interpretation": "文字解读",
                        },
                    ],
                    "byproducts": {
                        "antecedents_ids": [125, 6, 80],
                        "consequents_ids": [101, 102],
                        "antecedents_ranges": [[35, 50], [-10.5, -2], [7.6, 199.8]],
                        "consequents_ranges": [[13, 15], [25, 89.4]],
                        "antecedents_classes": ["mid|3", "low|3", "high|3"],
                        "consequents_classes": ["high|3", "high|3"],
                        # "outliers_ids": [485, 2345, 1232, 2583, 2584],
                    },
                    "gov_specific": {
                        "is_outlier": True,
                        "antecedents_classes": ["mid|3", "low|3", "high|3"],
                        "consequents_classes": ["low|3", "high|3"],
                    },
                },
                {
                    "index_id": 7,
                    "technique": "correlation",
                    "values": [
                        {
                            "name": "pearson",
                            "value": -0.2,
                            "interpretation": "文字解读",
                        },
                        {
                            "name": "spearman",
                            "value": -0.3,
                            "interpretation": "文字解读",
                        },
                    ],
                    "byproducts": {
                        # "outliers_ids": [1, 5, 88, 234],
                    },
                    "gov_specific": {
                        "is_outlier": False,
                    },
                },
            ],
        }
    )
