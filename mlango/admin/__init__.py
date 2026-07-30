"""The mlango administration site.

from mlango import admin

@admin.register(Reviews)
class ReviewsAdmin(admin.ObjectAdmin):
    list_display = ("id", "text", "label")
"""

from mlango.admin.options import (
    AgentAdmin,
    DatasetAdmin,
    EvalAdmin,
    ModelAdmin,
    ObjectAdmin,
)
from mlango.admin.sites import AdminSite, autodiscover, register, site

__all__ = [
    "site",
    "register",
    "autodiscover",
    "AdminSite",
    "ObjectAdmin",
    "DatasetAdmin",
    "ModelAdmin",
    "AgentAdmin",
    "EvalAdmin",
]
